#!/usr/bin/env python
import csv
import os
from pathlib import Path
import re
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = ROOT / "cnn"
sys.path.insert(0, str(CNN_DIR))

import genotypes  # noqa: E402
import utils  # noqa: E402
from cost_utils import build_search_costs, expected_cost  # noqa: E402
from model import NetworkCIFAR  # noqa: E402
from model_search import Network as SearchNetwork  # noqa: E402


CIFAR_CLASSES = 10
INPUT_SIZE = 32
REPORT_DIR = ROOT / "reports" / "cg_darts"


EXPERIMENTS = [
  {
    "method": "Vanilla",
    "lambda": "",
    "lambda_value": None,
    "search_dir": CNN_DIR / "search-first-order-50ep-20260517-222002",
    "eval_dir": CNN_DIR / "eval-eval-vanilla-seed2-100ep-20260518-185929",
  },
  {
    "method": "CG-DARTS",
    "lambda": "1e-3",
    "lambda_value": 1e-3,
    "search_dir": CNN_DIR / "search-cg-cg-flops-lambda1em3-seed2-20260517-233128",
    "eval_dir": None,
  },
  {
    "method": "CG-DARTS",
    "lambda": "5e-3",
    "lambda_value": 5e-3,
    "search_dir": CNN_DIR / "search-cg-cg-flops-lambda5em3-seed2-20260518-060004",
    "eval_dir": None,
  },
  {
    "method": "CG-DARTS",
    "lambda": "1e-2",
    "lambda_value": 1e-2,
    "search_dir": CNN_DIR / "search-cg-cg-flops-lambda1em2-seed2-20260518-105325",
    "eval_dir": CNN_DIR / "eval-eval-cg-lambda1e-2-seed2-100ep-20260519-004022",
  },
  {
    "method": "CG-DARTS",
    "lambda": "5e-2",
    "lambda_value": 5e-2,
    "search_dir": CNN_DIR / "search-cg-cg-flops-lambda5em2-seed2-20260518-110057",
    "eval_dir": CNN_DIR / "eval-eval-cg-lambda5e-2-seed2-100ep-20260519-004055",
  },
  {
    "method": "CG-DARTS",
    "lambda": "1e-1",
    "lambda_value": 1e-1,
    "search_dir": CNN_DIR / "search-cg-cg-flops-lambda1em1-seed2-20260518-183936",
    "eval_dir": None,
  },
]


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


def vanilla_expected_cost(search_dir):
  criterion = nn.CrossEntropyLoss()
  model = SearchNetwork(16, CIFAR_CLASSES, 8, criterion)
  state = torch.load(str(search_dir / "weights.pt"), map_location="cpu")
  model.load_state_dict(state)
  costs = build_search_costs(
    16, 8, steps=model._steps, input_size=INPUT_SIZE, metric="flops", normalize="edge")
  return float(expected_cost(model, costs.normal, costs.reduce))


def count_flops(model, input_size=(1, 3, 32, 32)):
  totals = {"conv": 0, "linear": 0, "pool": 0}
  handles = []

  def conv_hook(module, inputs, output):
    batch = output.shape[0]
    out_channels = output.shape[1]
    out_h = output.shape[2]
    out_w = output.shape[3]
    kernel_h, kernel_w = module.kernel_size
    in_channels = module.in_channels
    groups = module.groups
    totals["conv"] += batch * out_h * out_w * out_channels * (in_channels // groups) * kernel_h * kernel_w

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
  model = NetworkCIFAR(36, CIFAR_CLASSES, 20, True, genotype)
  params_m = utils.count_parameters_in_MB(model)
  flops = count_flops(model)
  total_macs = sum(flops.values())
  return params_m, total_macs, flops


def pct_reduction(value, baseline):
  if value is None or baseline is None:
    return None
  return 100.0 * (baseline - value) / baseline


def write_csv(path, rows, fieldnames):
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def plot_results(rows, histories):
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib is not installed; skipped plots")
    return

  labels = [row["label"] for row in rows]
  search_acc = [row["search_valid_acc"] for row in rows]
  expected = [row["expected_cost"] for row in rows]

  plt.figure(figsize=(7, 4.5))
  plt.scatter(expected, search_acc, s=70)
  for label, x, y in zip(labels, expected, search_acc):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)
  plt.xlabel("Normalized Expected Cost")
  plt.ylabel("Search Validation Accuracy (%)")
  plt.title("Search Accuracy vs. Expected Cost")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(REPORT_DIR / "search_accuracy_vs_expected_cost.png", dpi=200)
  plt.close()

  retrain_rows = [row for row in rows if row["retrain_test_acc"] is not None]
  plt.figure(figsize=(7, 4.5))
  plt.scatter(
    [row["params_m"] for row in retrain_rows],
    [row["retrain_test_acc"] for row in retrain_rows],
    s=70)
  for row in retrain_rows:
    plt.annotate(row["label"], (row["params_m"], row["retrain_test_acc"]),
                 textcoords="offset points", xytext=(6, 5), fontsize=9)
  plt.xlabel("Parameters (M, excluding auxiliary)")
  plt.ylabel("Retrain Test Accuracy (%)")
  plt.title("Retrain Accuracy vs. Model Size")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(REPORT_DIR / "retrain_accuracy_vs_params.png", dpi=200)
  plt.close()

  plt.figure(figsize=(7, 4.5))
  plt.scatter(
    [row["discrete_macs_m"] for row in retrain_rows],
    [row["retrain_test_acc"] for row in retrain_rows],
    s=70)
  for row in retrain_rows:
    plt.annotate(row["label"], (row["discrete_macs_m"], row["retrain_test_acc"]),
                 textcoords="offset points", xytext=(6, 5), fontsize=9)
  plt.xlabel("Final Discrete MACs (M)")
  plt.ylabel("Retrain Test Accuracy (%)")
  plt.title("Retrain Accuracy vs. Final Discrete MACs")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(REPORT_DIR / "retrain_accuracy_vs_macs.png", dpi=200)
  plt.close()

  plt.figure(figsize=(8, 4.5))
  x = range(len(rows))
  reductions = [row["expected_cost_reduction_pct"] or 0.0 for row in rows]
  plt.bar(x, reductions)
  plt.xticks(list(x), labels, rotation=25, ha="right")
  plt.ylabel("Expected Cost Reduction vs. Vanilla (%)")
  plt.title("Cost Reduction by Lambda")
  plt.tight_layout()
  plt.savefig(REPORT_DIR / "expected_cost_reduction.png", dpi=200)
  plt.close()

  plt.figure(figsize=(8, 4.5))
  for label, history in histories.items():
    if not history:
      continue
    epochs = [int(row["epoch"]) for row in history]
    costs = [float(row["expected_cost"]) for row in history]
    plt.plot(epochs, costs, label=label)
  plt.xlabel("Epoch")
  plt.ylabel("Normalized Expected Cost")
  plt.title("Expected Cost During Search")
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(REPORT_DIR / "expected_cost_over_epochs.png", dpi=200)
  plt.close()


def main():
  REPORT_DIR.mkdir(parents=True, exist_ok=True)

  rows = []
  histories = {}
  vanilla_cost = None

  for exp in EXPERIMENTS:
    label = exp["method"] if not exp["lambda"] else "lambda={}".format(exp["lambda"])
    search_dir = exp["search_dir"]
    if exp["method"] == "Vanilla":
      expected = vanilla_expected_cost(search_dir)
      search_acc = last_metric(search_dir / "log.txt", r"valid_acc ([0-9.]+)")
      histories[label] = []
      vanilla_cost = expected
    else:
      final_row, history = read_last_cost_log(search_dir / "cost_log.csv")
      expected = float(final_row["expected_cost"])
      search_acc = float(final_row["valid_acc"])
      histories[label] = history

    params_m, macs, flops_by_type = discrete_model_stats(search_dir)
    eval_dir = exp["eval_dir"]
    retrain_acc = None
    retrain_params_m = None
    if eval_dir is not None and (eval_dir / "log.txt").exists():
      retrain_acc = last_metric(eval_dir / "log.txt", r"valid_acc ([0-9.]+)")
      retrain_params_m = last_metric(eval_dir / "log.txt", r"param size = ([0-9.]+)MB")

    rows.append({
      "label": label,
      "method": exp["method"],
      "lambda": exp["lambda"],
      "search_valid_acc": search_acc,
      "expected_cost": expected,
      "expected_cost_reduction_pct": pct_reduction(expected, vanilla_cost) if vanilla_cost is not None else None,
      "discrete_macs": macs,
      "discrete_macs_m": macs / 1e6,
      "conv_macs_m": flops_by_type["conv"] / 1e6,
      "linear_macs_m": flops_by_type["linear"] / 1e6,
      "pool_ops_m": flops_by_type["pool"] / 1e6,
      "params_m": params_m,
      "params_reduction_pct": None,
      "retrain_test_acc": retrain_acc,
      "retrain_params_m_logged": retrain_params_m,
      "search_dir": str(search_dir.relative_to(ROOT)),
      "eval_dir": "" if eval_dir is None else str(eval_dir.relative_to(ROOT)),
    })

  baseline_params = rows[0]["params_m"]
  baseline_macs = rows[0]["discrete_macs"]
  for row in rows:
    row["params_reduction_pct"] = pct_reduction(row["params_m"], baseline_params)
    row["discrete_macs_reduction_pct"] = pct_reduction(row["discrete_macs"], baseline_macs)

  fieldnames = [
    "label", "method", "lambda", "search_valid_acc", "expected_cost",
    "expected_cost_reduction_pct", "discrete_macs_m", "discrete_macs_reduction_pct",
    "params_m", "params_reduction_pct", "retrain_test_acc", "retrain_params_m_logged",
    "conv_macs_m", "linear_macs_m", "pool_ops_m", "search_dir", "eval_dir",
  ]
  write_csv(REPORT_DIR / "summary.csv", rows, fieldnames)
  plot_results(rows, histories)

  print("Wrote {}".format(REPORT_DIR / "summary.csv"))
  print("")
  for row in rows:
    print("{label:>10s} search_acc={search_valid_acc:.3f} expected={expected_cost:.4f} "
          "macs={discrete_macs_m:.2f}M params={params_m:.3f}M retrain={retrain}".format(
            retrain="-" if row["retrain_test_acc"] is None else "{:.3f}".format(row["retrain_test_acc"]),
            **row))


if __name__ == "__main__":
  main()
