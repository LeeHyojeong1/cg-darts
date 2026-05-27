#!/usr/bin/env python
"""Generate device-conditioned CG-DARTS report and proposal-ready figures."""
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = ROOT / "cnn"
REPORT_DIR = ROOT / "reports" / "cg_darts_device"
FLOPS_SUMMARY = ROOT / "reports" / "cg_darts" / "summary.csv"
sys.path.insert(0, str(CNN_DIR))

import genotypes  # noqa: E402
import utils  # noqa: E402
from device_profiles import DEVICE_PROFILES, get_device_profile  # noqa: E402
from model import NetworkCIFAR  # noqa: E402

DEVICE_ORDER = ["jetson_orin", "rtx_pro_6000", "h100"]
DEVICE_COLORS = {
  "jetson_orin": "#2E86AB",
  "rtx_pro_6000": "#A23B72",
  "h100": "#F18F01",
}
DEVICE_SHORT = {
  "jetson_orin": "Edge\n(Orin)",
  "rtx_pro_6000": "Middle\n(RTX PRO 6000)",
  "h100": "High-end\n(H100)",
}
DEFAULT_LAMBDA = "1e-2"


def lambda_token(lmbda):
  return "lambda{}".format(lmbda.replace(".", "p"))


def newest_complete_dir(prefix):
  matches = sorted(
    (p for p in CNN_DIR.iterdir()
     if p.is_dir() and p.name.startswith(prefix) and (p / "weights.pt").exists()),
    key=lambda p: p.stat().st_mtime,
    reverse=True)
  return matches[0] if matches else None


def find_eval_dir(search_dir):
  suffix = search_dir.name[len("search-"):]
  for prefix in ("eval-eval-{}", "eval-{}"):
    found = newest_complete_dir(prefix.format(suffix))
    if found is not None:
      return found
  return None


def discover_device_experiments(lmbda=DEFAULT_LAMBDA):
  token = lambda_token(lmbda)
  experiments = []
  for device in DEVICE_ORDER:
    profile = get_device_profile(device)
    search_dir = newest_complete_dir("search-cg-device-{}-{}".format(device, token))
    if search_dir is None:
      continue
    eval_dir = find_eval_dir(search_dir)
    experiments.append({
      "device": device,
      "profile": profile,
      "lambda": lmbda,
      "search_dir": search_dir,
      "eval_dir": eval_dir,
    })
  return experiments


def read_last_cost_log(path):
  with path.open() as f:
    rows = list(csv.DictReader(f))
  if not rows:
    raise RuntimeError("empty cost log: {}".format(path))
  return rows[-1], rows


def last_metric(log_path, pattern):
  regex = re.compile(pattern)
  value = None
  with log_path.open(errors="replace") as f:
    for line in f:
      match = regex.search(line)
      if match:
        value = float(match.group(1))
  return value


def read_genotype(path):
  text = path.read_text().strip()
  return eval(text, {"Genotype": genotypes.Genotype, "range": range})


def count_flops(model, input_size=(1, 3, 32, 32)):
  totals = {"conv": 0, "linear": 0, "pool": 0}
  handles = []

  def conv_hook(module, inputs, output):
    batch = output.shape[0]
    out_channels, out_h, out_w = output.shape[1], output.shape[2], output.shape[3]
    kernel_h, kernel_w = module.kernel_size
    in_channels = module.in_channels
    groups = module.groups
    totals["conv"] += (
      batch * out_h * out_w * out_channels * (in_channels // groups) * kernel_h * kernel_w)

  def linear_hook(module, inputs, output):
    batch = inputs[0].shape[0]
    totals["linear"] += batch * module.in_features * module.out_features

  def pool_hook(module, inputs, output):
    if isinstance(module.kernel_size, tuple):
      kernel_h, kernel_w = module.kernel_size
    else:
      kernel_h = kernel_w = module.kernel_size
    totals["pool"] += output.numel() * kernel_h * kernel_w

  for module in model.modules():
    if isinstance(module, nn.Conv2d):
      handles.append(module.register_forward_hook(conv_hook))
    elif isinstance(module, nn.Linear):
      handles.append(module.register_forward_hook(linear_hook))
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
      handles.append(module.register_forward_hook(pool_hook))

  model.eval()
  model.drop_path_prob = 0.0
  with torch.no_grad():
    model(torch.zeros(*input_size))
  for handle in handles:
    handle.remove()
  return totals


def discrete_model_stats(search_dir):
  genotype = read_genotype(search_dir / "genotype.txt")
  model = NetworkCIFAR(36, 10, 20, True, genotype)
  params_m = utils.count_parameters_in_MB(model)
  flops = count_flops(model)
  total_macs = sum(flops.values())
  return params_m, total_macs, genotype


def load_flops_baselines():
  baselines = {}
  if not FLOPS_SUMMARY.exists():
    return baselines
  with FLOPS_SUMMARY.open() as f:
    for row in csv.DictReader(f):
      if row["label"] in ("Vanilla", "lambda=1e-2"):
        baselines[row["label"]] = row
  return baselines


def pct_reduction(value, baseline):
  if value is None or baseline is None or baseline == 0:
    return None
  return 100.0 * (baseline - value) / baseline


def setup_plot_style():
  import matplotlib.pyplot as plt
  plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
  })
  return plt


def plot_cost_weights(profiles, out_dir):
  plt = setup_plot_style()
  labels = [DEVICE_SHORT[d] for d in DEVICE_ORDER if d in profiles]
  devices = [d for d in DEVICE_ORDER if d in profiles]
  x = range(len(devices))
  width = 0.25
  w_f = [profiles[d].w_flops for d in devices]
  w_m = [profiles[d].w_mem for d in devices]
  w_l = [profiles[d].w_lat for d in devices]

  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.bar([i - width for i in x], w_f, width, label="FLOPs ($w_F$)", color="#4C72B0")
  ax.bar(list(x), w_m, width, label="Memory ($w_M$)", color="#55A868")
  ax.bar([i + width for i in x], w_l, width, label="Latency ($w_L$)", color="#C44E52")
  ax.set_xticks(list(x))
  ax.set_xticklabels(labels)
  ax.set_ylabel("Cost weight")
  ax.set_ylim(0, 0.75)
  ax.set_title("Device-tier cost mixing weights (Proposal v3 presets)")
  ax.legend(loc="upper right")
  ax.grid(axis="y", alpha=0.25)
  fig.tight_layout()
  fig.savefig(out_dir / "device_cost_weights.png")
  plt.close(fig)


def plot_cost_trajectory(rows, histories, out_dir):
  plt = setup_plot_style()
  fig, ax = plt.subplots(figsize=(8, 4.8))
  for row in rows:
    device = row["device"]
    history = histories[device]
    epochs = [int(r["epoch"]) for r in history]
    costs = [float(r["expected_cost"]) for r in history]
    ax.plot(epochs, costs, label=row["device_label"], color=DEVICE_COLORS[device], linewidth=2)
  ax.set_xlabel("Search epoch")
  ax.set_ylabel("Normalized expected device cost")
  ax.set_title("Expected device cost during architecture search ($\\lambda=10^{-2}$)")
  ax.legend()
  ax.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(out_dir / "device_expected_cost_over_epochs.png")
  plt.close(fig)


def plot_search_accuracy_vs_cost(rows, out_dir):
  plt = setup_plot_style()
  fig, ax = plt.subplots(figsize=(7.5, 5))
  for row in rows:
    device = row["device"]
    ax.scatter(
      row["expected_cost"], row["search_valid_acc"],
      s=120, color=DEVICE_COLORS[device], edgecolors="white", linewidths=0.8, zorder=3)
    ax.annotate(
      row["device_label"].replace(" (", "\n("),
      (row["expected_cost"], row["search_valid_acc"]),
      textcoords="offset points", xytext=(8, 4), fontsize=9)
  ax.set_xlabel("Final normalized expected device cost")
  ax.set_ylabel("Search validation accuracy (%)")
  ax.set_title("Search accuracy vs. device-conditioned cost")
  ax.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(out_dir / "device_search_accuracy_vs_cost.png")
  plt.close(fig)


def plot_retrain_pareto(rows, baselines, out_dir):
  plt = setup_plot_style()
  fig, ax = plt.subplots(figsize=(8, 5.5))

  if "Vanilla" in baselines:
    b = baselines["Vanilla"]
    ax.scatter(
      float(b["discrete_macs_m"]), float(b["retrain_test_acc"]),
      s=140, marker="D", color="#666666", label="Vanilla DARTS", zorder=2)
  if "lambda=1e-2" in baselines:
    b = baselines["lambda=1e-2"]
    ax.scatter(
      float(b["discrete_macs_m"]), float(b["retrain_test_acc"]),
      s=140, marker="s", color="#999999", label="CG-DARTS (FLOPs, $\\lambda=10^{-2}$)", zorder=2)

  for row in rows:
    if row["retrain_test_acc"] is None:
      continue
    device = row["device"]
    ax.scatter(
      row["discrete_macs_m"], row["retrain_test_acc"],
      s=160, color=DEVICE_COLORS[device], label=row["device_label"], zorder=3)
  ax.set_xlabel("Discrete MACs (M)")
  ax.set_ylabel("Retrain validation accuracy (%)")
  ax.set_title("Accuracy–efficiency trade-off after 100-epoch retrain")
  ax.legend(loc="lower right", framealpha=0.95)
  ax.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(out_dir / "device_retrain_accuracy_vs_macs.png")
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(8, 5.5))
  if "Vanilla" in baselines:
    b = baselines["Vanilla"]
    ax.scatter(
      float(b["params_m"]), float(b["retrain_test_acc"]),
      s=140, marker="D", color="#666666", label="Vanilla DARTS", zorder=2)
  if "lambda=1e-2" in baselines:
    b = baselines["lambda=1e-2"]
    ax.scatter(
      float(b["params_m"]), float(b["retrain_test_acc"]),
      s=140, marker="s", color="#999999", label="CG-DARTS (FLOPs, $\\lambda=10^{-2}$)", zorder=2)
  for row in rows:
    if row["retrain_test_acc"] is None:
      continue
    device = row["device"]
    ax.scatter(
      row["params_m"], row["retrain_test_acc"],
      s=160, color=DEVICE_COLORS[device], label=row["device_label"], zorder=3)
  ax.set_xlabel("Parameters (M)")
  ax.set_ylabel("Retrain validation accuracy (%)")
  ax.set_title("Accuracy vs. parameter count after retrain")
  ax.legend(loc="lower right", framealpha=0.95)
  ax.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(out_dir / "device_retrain_accuracy_vs_params.png")
  plt.close(fig)


def plot_tier_comparison(rows, baselines, out_dir):
  plt = setup_plot_style()
  metrics = [
    ("retrain_test_acc", "Retrain accuracy (%)", False),
    ("discrete_macs_m", "MACs (M)", True),
    ("params_m", "Parameters (M)", True),
  ]
  fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
  vanilla = baselines.get("Vanilla")

  for ax, (key, ylabel, lower_is_better) in zip(axes, metrics):
    xs = list(range(len(rows)))
    vals = [row[key] for row in rows]
    colors = [DEVICE_COLORS[row["device"]] for row in rows]
    bars = ax.bar(xs, vals, color=colors, width=0.62)
    ax.set_xticks(xs)
    ax.set_xticklabels([DEVICE_SHORT[row["device"]] for row in rows])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if vanilla and vanilla.get(key):
      ref = float(vanilla[key if key != "retrain_test_acc" else "retrain_test_acc"])
      ax.axhline(ref, color="#666666", linestyle="--", linewidth=1.2, label="Vanilla")
      ax.legend(loc="best", fontsize=8)
    for bar, val in zip(bars, vals):
      fmt = "{:.2f}" if key != "retrain_test_acc" else "{:.1f}%"
      ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height(),
        fmt.format(val), ha="center", va="bottom", fontsize=9)

  axes[0].set_title("Accuracy")
  axes[1].set_title("Compute (MACs)")
  axes[2].set_title("Model size")
  fig.suptitle("Device-conditioned CG-DARTS ($\\lambda=10^{-2}$, 50ep search / 100ep retrain)", y=1.02)
  fig.tight_layout()
  fig.savefig(out_dir / "device_tier_comparison.png", bbox_inches="tight")
  plt.close(fig)


def plot_cost_reduction(rows, out_dir):
  plt = setup_plot_style()
  first_cost = rows[0]["expected_cost"]
  fig, ax = plt.subplots(figsize=(8, 4.5))
  labels = [DEVICE_SHORT[r["device"]] for r in rows]
  reductions = [pct_reduction(r["expected_cost"], first_cost) or 0.0 for r in rows]
  colors = [DEVICE_COLORS[r["device"]] for r in rows]
  ax.bar(range(len(rows)), reductions, color=colors, width=0.62)
  ax.set_xticks(range(len(rows)))
  ax.set_xticklabels(labels)
  ax.set_ylabel("Expected cost change vs. Edge profile (%)")
  ax.set_title("Relative device-cost reduction at end of search")
  ax.axhline(0, color="#333333", linewidth=0.8)
  ax.grid(axis="y", alpha=0.25)
  fig.tight_layout()
  fig.savefig(out_dir / "device_expected_cost_reduction.png")
  plt.close(fig)


def write_markdown_report(rows, baselines, out_dir):
  vanilla = baselines.get("Vanilla", {})
  flops = baselines.get("lambda=1e-2", {})
  lines = [
    "# Device-Conditioned CG-DARTS Report",
    "",
    "CIFAR-10, seed=2, $\\lambda=10^{-2}$, search 50 epochs (warmup 10), retrain 100 epochs.",
    "",
    "## Method (per proposal)",
    "",
    "Each hardware tier uses a **device profile** with literature peak FP32 TFLOPS, memory",
    "bandwidth, and tier-specific mixing weights $(w_F, w_M, w_L)$:",
    "",
    "| Tier | Device | Peak FP32 | Mem BW | $(w_F, w_M, w_L)$ |",
    "|------|--------|-----------|--------|-------------------|",
  ]
  for device in DEVICE_ORDER:
    p = DEVICE_PROFILES[device]
    lines.append(
      "| {} | {} | {:.1f} TFLOPS | {:.0f} GB/s | ({:.2f}, {:.2f}, {:.2f}) |".format(
        DEVICE_SHORT[device].replace("\n", " "),
        p.sku, p.peak_tflops_fp32, p.mem_bandwidth_gbps,
        p.w_flops, p.w_mem, p.w_lat))

  lines.extend([
    "",
    "Architecture loss: $L_{arch} = L_{val} + \\lambda \\cdot \\mathbb{E}[cost(\\alpha)]$",
    "",
    "Per-op device cost combines edge-normalised FLOPs, activation-memory proxy, and",
    "roofline latency $\\max(F/\\mathrm{peak}, M/\\mathrm{BW})$ on the target device.",
    "",
    "## Results summary",
    "",
    "| Tier | Search val acc | E[cost] | MACs (M) | Params (M) | Retrain val acc |",
    "|------|----------------|---------|----------|------------|-----------------|",
  ])
  for row in rows:
    retrain = "-" if row["retrain_test_acc"] is None else "{:.2f}%".format(row["retrain_test_acc"])
    lines.append(
      "| {} | {:.2f}% | {:.3f} | {:.1f} | {:.3f} | {} |".format(
        row["device_label"], row["search_valid_acc"], row["expected_cost"],
        row["discrete_macs_m"], row["params_m"], retrain))

  if vanilla:
    lines.extend([
      "",
      "### Baselines (existing FLOPs CG-DARTS)",
      "",
      "- Vanilla DARTS retrain: {:.2f}%, {:.1f}M MACs, {:.3f}M params".format(
        float(vanilla.get("retrain_test_acc", 0)),
        float(vanilla.get("discrete_macs_m", 0)),
        float(vanilla.get("params_m", 0))),
    ])
  if flops:
    lines.append(
      "- CG-DARTS FLOPs $\\lambda=10^{{-2}}$: {:.2f}%, {:.1f}M MACs, {:.3f}M params".format(
        float(flops.get("retrain_test_acc", 0)),
        float(flops.get("discrete_macs_m", 0)),
        float(flops.get("params_m", 0))))

  lines.extend([
    "",
    "## Figures",
    "",
    "- `device_cost_weights.png` — tier mixing weights from proposal",
    "- `device_expected_cost_over_epochs.png` — cost trajectory during search",
    "- `device_search_accuracy_vs_cost.png` — search Pareto view",
    "- `device_tier_comparison.png` — accuracy / MACs / params by tier",
    "- `device_retrain_accuracy_vs_macs.png` — retrain trade-off vs Vanilla & FLOPs CG-DARTS",
    "- `device_retrain_accuracy_vs_params.png` — same vs parameter count",
    "",
  ])
  (out_dir / "REPORT.md").write_text("\n".join(lines))


def main():
  REPORT_DIR.mkdir(parents=True, exist_ok=True)
  experiments = discover_device_experiments()
  if len(experiments) < len(DEVICE_ORDER):
    found = [e["device"] for e in experiments]
    missing = [d for d in DEVICE_ORDER if d not in found]
    raise SystemExit("Missing completed device experiments: {}".format(", ".join(missing)))

  baselines = load_flops_baselines()
  rows = []
  histories = {}

  for exp in experiments:
    device = exp["device"]
    profile = exp["profile"]
    search_dir = exp["search_dir"]
    final_row, history = read_last_cost_log(search_dir / "cost_log.csv")
    params_m, macs, genotype = discrete_model_stats(search_dir)
    eval_dir = exp["eval_dir"]
    retrain_acc = None
    if eval_dir is not None:
      retrain_acc = last_metric(eval_dir / "log.txt", r"valid_acc ([0-9.]+)")

    row = {
      "device": device,
      "device_label": profile.label,
      "lambda": exp["lambda"],
      "peak_tflops_fp32": profile.peak_tflops_fp32,
      "mem_bandwidth_gbps": profile.mem_bandwidth_gbps,
      "w_flops": profile.w_flops,
      "w_mem": profile.w_mem,
      "w_lat": profile.w_lat,
      "search_valid_acc": float(final_row["valid_acc"]),
      "expected_cost": float(final_row["expected_cost"]),
      "discrete_macs": macs,
      "discrete_macs_m": macs / 1e6,
      "params_m": params_m,
      "retrain_test_acc": retrain_acc,
      "genotype": str(genotype),
      "search_dir": str(search_dir.relative_to(ROOT)),
      "eval_dir": "" if eval_dir is None else str(eval_dir.relative_to(ROOT)),
    }
    rows.append(row)
    histories[device] = history

  vanilla_macs = float(baselines["Vanilla"]["discrete_macs_m"]) if "Vanilla" in baselines else None
  vanilla_params = float(baselines["Vanilla"]["params_m"]) if "Vanilla" in baselines else None
  for row in rows:
    row["macs_reduction_pct_vs_vanilla"] = pct_reduction(row["discrete_macs_m"], vanilla_macs)
    row["params_reduction_pct_vs_vanilla"] = pct_reduction(row["params_m"], vanilla_params)

  fieldnames = [
    "device", "device_label", "lambda", "peak_tflops_fp32", "mem_bandwidth_gbps",
    "w_flops", "w_mem", "w_lat", "search_valid_acc", "expected_cost",
    "discrete_macs_m", "macs_reduction_pct_vs_vanilla", "params_m",
    "params_reduction_pct_vs_vanilla", "retrain_test_acc", "genotype",
    "search_dir", "eval_dir",
  ]
  with (REPORT_DIR / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

  profiles = {r["device"]: DEVICE_PROFILES[r["device"]] for r in rows}
  plot_cost_weights(profiles, REPORT_DIR)
  plot_cost_trajectory(rows, histories, REPORT_DIR)
  plot_search_accuracy_vs_cost(rows, REPORT_DIR)
  plot_retrain_pareto(rows, baselines, REPORT_DIR)
  plot_tier_comparison(rows, baselines, REPORT_DIR)
  plot_cost_reduction(rows, REPORT_DIR)
  write_markdown_report(rows, baselines, REPORT_DIR)

  print("Wrote report to {}".format(REPORT_DIR))
  for row in rows:
    print("{label}: search={sa:.2f}% cost={c:.3f} macs={m:.1f}M params={p:.3f}M retrain={r}".format(
      label=row["device_label"],
      sa=row["search_valid_acc"],
      c=row["expected_cost"],
      m=row["discrete_macs_m"],
      p=row["params_m"],
      r="-" if row["retrain_test_acc"] is None else "{:.2f}%".format(row["retrain_test_acc"])))


if __name__ == "__main__":
  main()
