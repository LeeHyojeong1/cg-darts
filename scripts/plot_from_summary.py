#!/usr/bin/env python
"""Generate report plots from summary.csv (no experiment dirs required)."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "cg_darts"
CSV_PATH = REPORT_DIR / "summary.csv"


def load_rows():
  with CSV_PATH.open() as f:
    return list(csv.DictReader(f))


def plot_results(rows):
  import matplotlib.pyplot as plt

  labels = [r["label"] for r in rows]
  search_acc = [float(r["search_valid_acc"]) for r in rows]
  expected = [float(r["expected_cost"]) for r in rows]

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

  retrain_rows = [r for r in rows if r.get("retrain_test_acc")]
  if retrain_rows:
    plt.figure(figsize=(7, 4.5))
    plt.scatter(
      [float(r["params_m"]) for r in retrain_rows],
      [float(r["retrain_test_acc"]) for r in retrain_rows],
      s=70)
    for r in retrain_rows:
      plt.annotate(r["label"], (float(r["params_m"]), float(r["retrain_test_acc"])),
                   textcoords="offset points", xytext=(6, 5), fontsize=9)
    plt.xlabel("Parameters (M)")
    plt.ylabel("Retrain Test Accuracy (%)")
    plt.title("Retrain Accuracy vs. Model Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "retrain_accuracy_vs_params.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.scatter(
      [float(r["discrete_macs_m"]) for r in retrain_rows],
      [float(r["retrain_test_acc"]) for r in retrain_rows],
      s=70)
    for r in retrain_rows:
      plt.annotate(r["label"], (float(r["discrete_macs_m"]), float(r["retrain_test_acc"])),
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
  reductions = [float(r["expected_cost_reduction_pct"] or 0) for r in rows]
  plt.bar(list(x), reductions)
  plt.xticks(list(x), labels, rotation=25, ha="right")
  plt.ylabel("Expected Cost Reduction vs. Vanilla (%)")
  plt.title("Cost Reduction by Lambda")
  plt.tight_layout()
  plt.savefig(REPORT_DIR / "expected_cost_reduction.png", dpi=200)
  plt.close()


def main():
  if not CSV_PATH.exists():
    raise SystemExit("Missing {}".format(CSV_PATH))
  rows = load_rows()
  REPORT_DIR.mkdir(parents=True, exist_ok=True)
  plot_results(rows)
  print("Wrote plots under {}".format(REPORT_DIR))


if __name__ == "__main__":
  main()
