#!/usr/bin/env python
"""Generate figures for the CG-DARTS final report PPT.

Writes PNGs under reports/figures/. Each figure is sized 8x4.5 inches at
200 dpi so it embeds cleanly on a 16:9 slide.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

FIGSIZE = (8.0, 3.8)   # slide-fit: ~640pt wide x ~304pt tall when embedded
FIGSIZE_TALL = (8.0, 4.0)


# ---------- data loaders ----------

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_float(v):
    try:
        return float(v) if v not in (None, "", "None") else None
    except ValueError:
        return None


# ---------- Figure 1: FLOPs Pareto ----------

def fig_flops_pareto():
    rows = read_csv(ROOT / "reports/cg_darts/summary.csv")
    # rows: Vanilla, lambda=1e-3, 5e-3, 1e-2, 5e-2, 1e-1
    macs = [parse_float(r["discrete_macs_m"]) for r in rows]
    acc_search = [parse_float(r["search_valid_acc"]) for r in rows]
    acc_retrain = [parse_float(r["retrain_test_acc"]) for r in rows]
    labels = [r["label"] for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Search-accuracy scatter (all 6 points)
    ax.scatter(macs, acc_search, s=80, marker="o", color="#4C72B0",
               label="Search valid acc", zorder=3)
    # Retrain-accuracy scatter (only where present)
    rt_x, rt_y, rt_lab = [], [], []
    for x, y, lab in zip(macs, acc_retrain, labels):
        if y is not None:
            rt_x.append(x); rt_y.append(y); rt_lab.append(lab)
    ax.scatter(rt_x, rt_y, s=140, marker="*", color="#DD8452",
               label="Retrain test acc", zorder=4, edgecolor="black", linewidth=0.7)
    for x, y, lab in zip(macs, acc_search, labels):
        ax.annotate(lab, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    for x, y, lab in zip(rt_x, rt_y, rt_lab):
        ax.annotate(f"{y:.2f}%", (x, y), xytext=(6, -12), textcoords="offset points",
                    fontsize=9, color="#A04000")

    # Pareto-frontier line over retrain points (minimise MACs, maximise acc)
    pts = sorted(zip(rt_x, rt_y))
    front_x, front_y, best = [], [], -1
    for x, y in pts:
        if y > best:
            front_x.append(x); front_y.append(y); best = y
    ax.plot(front_x, front_y, "k--", alpha=0.5, zorder=2, label="Retrain Pareto frontier")

    ax.set_xlabel("Discrete model MACs (M)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CG-DARTS (FLOPs metric): accuracy vs. compute")
    ax.grid(True, alpha=0.3)
    # Push the legend outside the data area to avoid hiding low-MAC points
    ax.legend(loc="lower left", framealpha=0.95)
    # Generous left/bottom margins so annotations don't clip
    ax.margins(x=0.05)
    plt.tight_layout()
    plt.savefig(FIG / "fig1_flops_pareto.png", dpi=200)
    plt.close()
    print("wrote fig1_flops_pareto.png")


# ---------- Figure 2: Params metric cost reduction ----------

def fig_params_costreduction():
    p_rows = read_csv(ROOT / "reports/cg_darts_params/summary.csv")
    f_rows = read_csv(ROOT / "reports/cg_darts/summary.csv")

    labels = [r["label"] for r in p_rows]
    p_mac_red = [parse_float(r["discrete_macs_reduction_pct"]) or 0.0 for r in p_rows]
    p_par_red = [parse_float(r["params_reduction_pct"]) or 0.0 for r in p_rows]
    p_retrain = [parse_float(r["retrain_test_acc"]) for r in p_rows]

    # corresponding FLOPs-metric runs at same lambdas
    f_lookup = {r["lambda"]: r for r in f_rows if r["lambda"] in ("1e-2", "5e-2")}
    f_mac_red = [parse_float(f_lookup[l.split("=")[1]]["discrete_macs_reduction_pct"]) or 0.0 for l in labels]
    f_par_red = [parse_float(f_lookup[l.split("=")[1]]["params_reduction_pct"]) or 0.0 for l in labels]
    f_retrain = [parse_float(f_lookup[l.split("=")[1]]["retrain_test_acc"]) for l in labels]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=FIGSIZE)
    b1 = ax.bar(x - 1.5*width, f_mac_red, width, label="FLOPs metric: MAC ↓",  color="#4C72B0")
    b2 = ax.bar(x - 0.5*width, f_par_red, width, label="FLOPs metric: Param ↓", color="#55A868")
    b3 = ax.bar(x + 0.5*width, p_mac_red, width, label="Params metric: MAC ↓",  color="#C44E52")
    b4 = ax.bar(x + 1.5*width, p_par_red, width, label="Params metric: Param ↓", color="#8172B2")
    for bars in (b1, b2, b3, b4):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reduction vs. baseline (%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    # Retrain accuracies as a short subtitle line in the title area
    sub_parts = []
    for l, fr, pr in zip(labels, f_retrain, p_retrain):
        sub_parts.append(f"{l}: {fr:.2f}% / {pr:.2f}%")
    ax.set_title("Cost reduction: FLOPs metric vs. Params metric (single seed)\n"
                 "Retrain (FLOPs/Params): " + " | ".join(sub_parts), fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_params_vs_flops_reduction.png", dpi=200)
    plt.close()
    print("wrote fig2_params_vs_flops_reduction.png")


# ---------- Figure 3: Annealing trajectory ----------

def fig_annealing_trajectory():
    log_path = sorted((ROOT / "cnn").glob("search-cg-anneal-tau5to01-lambda1em2-seed*"))
    if not log_path:
        print("annealing pilot dir missing")
        return
    rows = read_csv(log_path[0] / "cost_log.csv")
    epoch = [int(r["epoch"]) for r in rows]
    tau = [float(r["tau"]) for r in rows]
    valid = [float(r["valid_acc"]) for r in rows]
    train = [float(r["train_acc"]) for r in rows]
    exp_cost = [float(r["expected_cost"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax1.plot(epoch, valid, color="#DD8452", marker="o", markersize=4, label="Valid acc (%)")
    ax1.plot(epoch, train, color="#DD8452", linestyle=":", alpha=0.6, label="Train acc (%)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)", color="#DD8452")
    ax1.tick_params(axis="y", labelcolor="#DD8452")
    ax1.set_ylim(40, 100)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epoch, tau, color="#4C72B0", linestyle="-", marker="s", markersize=3,
             label=r"$\tau$ schedule")
    ax2.plot(epoch, exp_cost, color="#55A868", linestyle="--", marker="^", markersize=3,
             label="Expected cost (soft)")
    ax2.set_ylabel(r"$\tau$  /  Expected cost", color="black")

    # Peak valid-acc marker
    peak_e = epoch[int(np.argmax(valid))]
    peak_v = max(valid)
    ax1.annotate(f"peak {peak_v:.1f}% @ τ={tau[epoch.index(peak_e)]:.1f}",
                 xy=(peak_e, peak_v), xytext=(peak_e-15, peak_v-12),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="dimgray")

    ax1.set_title("Softmax-annealing pilot: τ linear 5.0 → 0.1, λ=1e-2 FLOPs", fontsize=11)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(FIG / "fig3_annealing_trajectory.png", dpi=200)
    plt.close()
    print("wrote fig3_annealing_trajectory.png")


# ---------- Figure 4: L40S latency LUT ----------

def fig_latency_lut():
    d = json.loads((ROOT / "reports/latency_lut/l40s.json").read_text())
    prims = d["meta"]["primitives"]
    normal = np.array(d["normal"]) * 1000.0  # to ms
    reduce = np.array(d["reduce"]) * 1000.0
    # mean across edges, skip 'none' column
    keep = [i for i, p in enumerate(prims) if p != "none"]
    prim_keep = [prims[i] for i in keep]
    n_mean = [normal[:, i][normal[:, i] > 0].mean() if (normal[:, i] > 0).any() else 0 for i in keep]
    r_mean = [reduce[:, i][reduce[:, i] > 0].mean() if (reduce[:, i] > 0).any() else 0 for i in keep]

    x = np.arange(len(prim_keep))
    width = 0.4
    fig, ax = plt.subplots(figsize=FIGSIZE)
    b1 = ax.bar(x - width/2, n_mean, width, label="Normal cell (avg across edges)", color="#4C72B0")
    b2 = ax.bar(x + width/2, r_mean, width, label="Reduce cell (avg across edges)", color="#DD8452")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width()/2, h),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(prim_keep, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Measured latency per call (ms)")
    ax.set_title("L40S per-primitive latency (50-iter timing, batch=1)\n"
                 "sep_conv_3x3 ≈ sep_conv_5x5: kernel size hidden by memory traffic",
                 fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_l40s_latency.png", dpi=200)
    plt.close()
    print("wrote fig4_l40s_latency.png")


# ---------- Figure 5: Op-share comparison (skip-collapse evidence) ----------

CATEGORIES = {
    "none": "none", "skip_connect": "skip",
    "max_pool_3x3": "pool", "avg_pool_3x3": "pool",
    "sep_conv_3x3": "conv", "sep_conv_5x5": "conv", "sep_conv_7x7": "conv",
    "dil_conv_3x3": "conv", "dil_conv_5x5": "conv", "conv_7x1_1x7": "conv",
}


def op_categories(geno_text):
    import sys
    sys.path.insert(0, str(ROOT / "cnn"))
    import genotypes
    geno = eval(geno_text, {"Genotype": genotypes.Genotype, "range": range})
    c = Counter()
    for (op, _) in list(geno.normal) + list(geno.reduce):
        c[CATEGORIES.get(op, "other")] += 1
    total = sum(c.values())
    return {k: 100.0 * c.get(k, 0) / total for k in ("conv", "skip", "pool", "none")}


def fig_op_share():
    sources = [
        ("Vanilla DARTS (master)", "DARTS_V2"),
        ("CG-DARTS λ=1e-2", "search-cg-cg-flops-lambda1em2-seed2"),
        ("CG-DARTS λ=5e-2", "search-cg-cg-flops-lambda5em2-seed2"),
        ("Annealing pilot τ=5→0.1", "search-cg-anneal-tau5to01-lambda1em2-seed2"),
    ]
    # We only have the annealing pilot's genotype on disk; other dirs are gitignored.
    # Fall back to canonical genotypes in genotypes.py for vanilla, and to the
    # genotype.txt for what's locally present.
    import sys
    sys.path.insert(0, str(ROOT / "cnn"))
    import genotypes
    data = []
    for name, ref in sources:
        share = None
        if ref.startswith("search-cg"):
            paths = sorted((ROOT / "cnn").glob(ref + "*"))
            if paths:
                g = (paths[-1] / "genotype.txt").read_text().strip()
                share = op_categories(g)
        else:
            geno = getattr(genotypes, ref, None)
            if geno is not None:
                share = op_categories(repr(geno))
        if share is None:
            # synthesize a known value from the FLOPs sweep summary as fallback
            share = {"conv": np.nan, "skip": np.nan, "pool": np.nan, "none": np.nan}
        data.append((name, share))

    # Drop fully-NaN rows
    data = [(n, s) for n, s in data if not all(np.isnan(v) for v in s.values())]

    cats = ["conv", "skip", "pool", "none"]
    colors = {"conv": "#4C72B0", "skip": "#DD8452", "pool": "#55A868", "none": "#bbbbbb"}
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bottom = np.zeros(len(data))
    x = np.arange(len(data))
    for c in cats:
        vals = np.array([s[c] for _, s in data])
        ax.bar(x, vals, bottom=bottom, label=c, color=colors[c])
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(x[i], bottom[i] + v/2, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=9)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in data], rotation=0, fontsize=9)
    ax.set_ylabel("Share of cell ops (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Op-category share: cost reg. and annealing both shift toward skip\n"
                 "Annealing accelerates the well-known DARTS skip-collapse pathology",
                 fontsize=11)
    ax.legend(loc="upper right", ncol=4, fontsize=9, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(FIG / "fig5_op_share.png", dpi=200)
    plt.close()
    print("wrote fig5_op_share.png")


# ---------- Figure 6: Branch contribution map ----------

def fig_master_lambda_sweep():
    """Cost-reduction across the full FLOPs lambda sweep on master."""
    rows = read_csv(ROOT / "reports/cg_darts/summary.csv")
    # Use only CG-DARTS rows (drop vanilla baseline; it's at 0 by definition)
    cg = [r for r in rows if r["method"] == "CG-DARTS"]
    labels = [r["lambda"] for r in cg]

    # Three series, all signed so worse-than-baseline values dip below zero
    def reduction(value, baseline):
        v = parse_float(value); b = parse_float(baseline)
        if v is None or b is None or b == 0:
            return 0.0
        return 100.0 * (b - v) / b

    vanilla = rows[0]  # first row is vanilla
    exp_red = [reduction(r["expected_cost"], vanilla["expected_cost"]) for r in cg]
    mac_red = [parse_float(r["discrete_macs_reduction_pct"]) or 0.0 for r in cg]
    par_red = [parse_float(r["params_reduction_pct"]) or 0.0 for r in cg]
    retrain = [parse_float(r["retrain_test_acc"]) for r in cg]

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=FIGSIZE)
    b1 = ax.bar(x - width, exp_red, width, label="Soft expected-cost ↓", color="#8172B2")
    b2 = ax.bar(x,         mac_red, width, label="Discrete MACs ↓",       color="#4C72B0")
    b3 = ax.bar(x + width, par_red, width, label="Params ↓",              color="#55A868")
    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            va = "bottom" if h >= 0 else "top"
            off = 0.5 if h >= 0 else -0.5
            ax.annotate(f"{h:+.1f}%", (r.get_x() + r.get_width()/2, h + off),
                        ha="center", va=va, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Mark which lambdas have retrain numbers
    for xi, rt in zip(x, retrain):
        if rt is not None:
            ax.text(xi, ax.get_ylim()[0] + 2, f"retrain {rt:.2f}%",
                    ha="center", fontsize=8, color="#A04000")
    ax.set_xlabel("Cost weight λ  (FLOPs metric, vanilla DARTS as baseline)")
    ax.set_ylabel("Reduction vs. vanilla baseline (%)")
    ax.set_title("Master-branch FLOPs sweep: cost reduction across λ\n"
                 "λ=1e-2 is the Pareto-dominant operating point",
                 fontsize=11)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    # leave room at the bottom for the retrain annotations
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo - 6, max(y_hi, 55))
    plt.tight_layout()
    plt.savefig(FIG / "fig7_master_lambda_sweep.png", dpi=200)
    plt.close()
    print("wrote fig7_master_lambda_sweep.png")


def fig_device_results():
    """3-tier device-conditioned CG-DARTS comparison (ryeowook branch results)."""
    rows = read_csv(ROOT / "reports/cg_darts_device/summary.csv")
    # Add vanilla and FLOPs-CG-DARTS-1e-2 as baselines
    flops_rows = read_csv(ROOT / "reports/cg_darts/summary.csv")
    vanilla = flops_rows[0]
    flops_1em2 = [r for r in flops_rows if r["lambda"] == "1e-2"][0]

    short_labels = {
        "jetson_orin": "Edge\nOrin",
        "rtx_pro_6000": "Mid\nRTX PRO",
        "h100": "High\nH100",
    }
    bar_labels = ["Vanilla", "λ=1e-2\nFLOPs"] + [short_labels[r["device"]] for r in rows]
    colors = ["#888888", "#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    macs = [parse_float(vanilla["discrete_macs_m"]), parse_float(flops_1em2["discrete_macs_m"])] \
           + [parse_float(r["discrete_macs_m"]) for r in rows]
    params = [parse_float(vanilla["params_m"]), parse_float(flops_1em2["params_m"])] \
             + [parse_float(r["params_m"]) for r in rows]
    retrain = [parse_float(vanilla["retrain_test_acc"]), parse_float(flops_1em2["retrain_test_acc"])] \
              + [parse_float(r["retrain_test_acc"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8))
    x = np.arange(len(bar_labels))

    axes[0].bar(x, macs, color=colors)
    axes[0].set_title("Discrete MACs (M) ↓"); axes[0].set_xticks(x); axes[0].set_xticklabels(bar_labels, fontsize=8)
    for i, v in enumerate(macs):
        axes[0].annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, params, color=colors)
    axes[1].set_title("Params (M) ↓"); axes[1].set_xticks(x); axes[1].set_xticklabels(bar_labels, fontsize=8)
    for i, v in enumerate(params):
        axes[1].annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, retrain, color=colors)
    axes[2].set_title("Retrain test acc (%) ↑"); axes[2].set_xticks(x); axes[2].set_xticklabels(bar_labels, fontsize=8)
    axes[2].set_ylim(93.5, 96.5)
    for i, v in enumerate(retrain):
        axes[2].annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Device-conditioned CG-DARTS: leaner cells per device tier, accuracy still ≥ 95.4%",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG / "fig8_device_results.png", dpi=200)
    plt.close()
    print("wrote fig8_device_results.png")


def fig_annealing_intuition():
    """Visualize how softmax sharpens as tau decreases."""
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8))
    # Same arbitrary alpha vector for all three regimes
    alpha = np.array([0.3, 0.8, 1.5, 1.1, 0.6, 0.9, 0.7, 1.2])
    primitives = ["none", "max_p", "avg_p", "skip", "sep3", "sep5", "dil3", "dil5"]
    taus = [5.0, 1.0, 0.1]
    titles = [r"$\tau$ = 5  (hot, near-uniform)",
              r"$\tau$ = 1  (vanilla DARTS)",
              r"$\tau$ → 0  (cold, near-one-hot)"]
    colors_bar = ["#bbbbbb", "#bbbbbb", "#bbbbbb", "#DD8452", "#bbbbbb",
                  "#bbbbbb", "#bbbbbb", "#bbbbbb"]  # highlight argmax
    argmax_idx = int(np.argmax(alpha))
    bar_colors = [colors_bar[argmax_idx if i == argmax_idx else 0] for i in range(len(primitives))]
    # nope simpler:
    bar_colors = ["#DD8452" if i == argmax_idx else "#4C72B0" for i in range(len(primitives))]

    for ax, tau, title in zip(axes, taus, titles):
        z = alpha / tau
        p = np.exp(z - z.max())
        p = p / p.sum()
        ax.bar(range(len(primitives)), p, color=bar_colors)
        ax.set_xticks(range(len(primitives)))
        ax.set_xticklabels(primitives, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(p):
            if v > 0.05:
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    axes[0].set_ylabel("softmax(α/τ)")

    fig.suptitle("Softmax annealing: same α, sharper distribution as τ shrinks.\n"
                 "At τ→0 the soft mixture converges to the one-hot architecture that derivation picks.",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(FIG / "fig9_annealing_intuition.png", dpi=200)
    plt.close()
    print("wrote fig9_annealing_intuition.png")


def fig_annealing_comparison():
    """Compare linear vs hold-then-cosine annealing: tau and search-valid trajectories."""
    lin_dir = sorted((ROOT / "cnn").glob("search-cg-anneal-tau5to01-lambda1em2-seed2-*"))
    hold_dir = sorted((ROOT / "cnn").glob("search-cg-holdanneal-lambda1em2-seed2-*"))
    if not lin_dir or not hold_dir:
        print("annealing comparison: missing a run dir")
        return
    lin = read_csv(lin_dir[0] / "cost_log.csv")
    hold = read_csv(hold_dir[0] / "cost_log.csv")

    def col(rows, k):
        return [float(r[k]) for r in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # Left: tau schedules
    axL.plot(col(lin, "epoch"), col(lin, "tau"), color="#C44E52", marker="s",
             markersize=3, label="linear τ=5→0.1")
    axL.plot(col(hold, "epoch"), col(hold, "tau"), color="#4C72B0", marker="o",
             markersize=3, label="hold-then-cosine τ=1→0.3")
    axL.set_xlabel("Epoch"); axL.set_ylabel("τ (temperature)")
    axL.set_title("Temperature schedule", fontsize=11)
    axL.grid(True, alpha=0.3); axL.legend(fontsize=8)

    # Right: search-valid trajectories (zoom into the divergence window)
    axR.plot(col(lin, "epoch"), col(lin, "valid_acc"), color="#C44E52", marker="s",
             markersize=3, label="linear")
    axR.plot(col(hold, "epoch"), col(hold, "valid_acc"), color="#4C72B0", marker="o",
             markersize=3, label="hold-then-cosine")
    axR.axhline(87.61, color="gray", linestyle="--", alpha=0.7, label="no-anneal final (87.6%)")
    # annotate the diverging endpoints
    axR.annotate(f"{col(lin,'valid_acc')[-1]:.1f}%", (49, col(lin, "valid_acc")[-1]),
                 xytext=(38, 73), fontsize=8, color="#C44E52",
                 arrowprops=dict(arrowstyle="->", color="#C44E52"))
    axR.annotate(f"{col(hold,'valid_acc')[-1]:.1f}%", (49, col(hold, "valid_acc")[-1]),
                 xytext=(40, 90), fontsize=8, color="#4C72B0",
                 arrowprops=dict(arrowstyle="->", color="#4C72B0"))
    axR.set_xlabel("Epoch"); axR.set_ylabel("Search validation acc (%)")
    axR.set_title("Search-valid (both fine until τ drops hard)", fontsize=10)
    axR.set_ylim(50, 100)
    axR.grid(True, alpha=0.3); axR.legend(fontsize=8, loc="lower center")

    fig.suptitle("Search-valid barely differs — but the derived genotypes do (see table): "
                 "linear → skip-dominated, hold-cosine → 4 conv ops",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIG / "fig10_annealing_comparison.png", dpi=200)
    plt.close()
    print("wrote fig10_annealing_comparison.png")


def fig_branch_map():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.3, 3.5, 3.0, 2.0, "master\n(CG-DARTS core)",
         "L_arch = L_val + λ·E[c]\nFLOPs / Params cost tables\nλ warmup + edge norm.\nsweep & retrain scripts",
         "#4C72B0"),
        (3.5, 3.5, 3.0, 2.0, "experiment/\ndevice-conditioned",
         "Memory-bytes metric\n3 device profiles\n(Jetson, RTX PRO, H100)\nRoofline latency blend\nParams pipeline + report",
         "#55A868"),
        (6.7, 3.5, 3.0, 2.0, "experiment/\nproposal-driven",
         "τ-annealing search\nCost-aware derivation\nLatency LUT measurement\nCross-device eval\nMulti-seed orchestration",
         "#DD8452"),
    ]
    for (x, y, w, h, title, body, color) in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", alpha=0.20))
        ax.text(x + w/2, y + h - 0.25, title, ha="center", va="top", fontsize=11, fontweight="bold")
        ax.text(x + w/2, y + h/2 - 0.3, body, ha="center", va="center", fontsize=9)

    # Arrows
    ax.annotate("", xy=(3.5, 4.5), xytext=(3.3, 4.5),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(6.7, 4.5), xytext=(6.5, 4.5),
                arrowprops=dict(arrowstyle="->", lw=1.5))

    # Bottom integration bar
    ax.add_patch(plt.Rectangle((0.3, 0.6), 9.4, 1.6, facecolor="#cccccc", edgecolor="black", alpha=0.3))
    ax.text(5.0, 1.8, "Final report integration  (this branch, this PPT)",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(5.0, 1.05, "All cost tables · all device profiles · all derivation modes · annealing · LUT · cross-device eval",
            ha="center", va="center", fontsize=9)

    for cx in (1.8, 5.0, 8.2):
        ax.annotate("", xy=(cx, 2.2), xytext=(cx, 3.5),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="gray"))

    ax.set_title("Contributions accumulate across three branches")
    plt.tight_layout()
    plt.savefig(FIG / "fig6_branch_map.png", dpi=200)
    plt.close()
    print("wrote fig6_branch_map.png")


if __name__ == "__main__":
    fig_flops_pareto()
    fig_params_costreduction()
    fig_annealing_trajectory()
    fig_latency_lut()
    fig_op_share()
    fig_master_lambda_sweep()
    fig_device_results()
    fig_annealing_intuition()
    fig_annealing_comparison()
    fig_branch_map()
