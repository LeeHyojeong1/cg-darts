#!/usr/bin/env python
"""Pareto frontier plot for CG-DARTS sweeps.

Reads one or more summary CSVs (each produced by `cg_darts_report.py` or the
multi-seed runner) and renders an accuracy-vs-cost scatter with the
non-dominated frontier highlighted. Configurations that share a label across
seeds are averaged with error bars.

Usage:
  python scripts/plot_pareto.py \
      --csv reports/cg_darts/summary.csv:FLOPs \
      --csv reports/cg_darts_params/summary.csv:Params \
      --x-key discrete_macs_m --y-key retrain_test_acc \
      --out reports/pareto/macs_vs_acc.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path):
  with open(path) as f:
    return list(csv.DictReader(f))


def parse_float(value):
  if value is None or value == '' or value == 'None':
    return None
  try:
    return float(value)
  except ValueError:
    return None


def aggregate(rows, x_key, y_key, label_key):
  groups = defaultdict(list)
  for r in rows:
    x = parse_float(r.get(x_key))
    y = parse_float(r.get(y_key))
    if x is None or y is None:
      continue
    label = r.get(label_key, '')
    groups[label].append((x, y))
  out = []
  for label, pts in groups.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)
    out.append({
      'label': label,
      'n': n,
      'x_mean': sum(xs) / n,
      'y_mean': sum(ys) / n,
      'x_std': _std(xs),
      'y_std': _std(ys),
    })
  return out


def _std(xs):
  if len(xs) < 2:
    return 0.0
  m = sum(xs) / len(xs)
  return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def pareto_front(points, minimize_x=True, maximize_y=True):
  pts = sorted(points, key=lambda p: (p['x_mean'] if minimize_x else -p['x_mean']))
  front = []
  best_y = None
  for p in pts:
    y = p['y_mean']
    if best_y is None or (maximize_y and y > best_y) or (not maximize_y and y < best_y):
      front.append(p)
      best_y = y
  return front


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--csv', action='append', required=True,
                  help='path:label pair; may repeat (e.g. reports/x/summary.csv:FLOPs)')
  ap.add_argument('--x-key', default='discrete_macs_m')
  ap.add_argument('--y-key', default='retrain_test_acc')
  ap.add_argument('--label-key', default='label')
  ap.add_argument('--x-label', default='Discrete MACs (M)')
  ap.add_argument('--y-label', default='Retrain Test Accuracy (%)')
  ap.add_argument('--title', default='Accuracy-Cost Pareto Frontier')
  ap.add_argument('--out', default='reports/pareto/frontier.png')
  ap.add_argument('--no-frontier', action='store_true',
                  help='disable Pareto-line overlay')
  args = ap.parse_args()

  import matplotlib.pyplot as plt
  plt.figure(figsize=(8, 5))

  all_points = []
  for spec in args.csv:
    if ':' in spec:
      path, series_label = spec.split(':', 1)
    else:
      path, series_label = spec, Path(spec).stem
    rows = read_rows(path)
    pts = aggregate(rows, args.x_key, args.y_key, args.label_key)
    if not pts:
      print('no data in', path)
      continue
    xs = [p['x_mean'] for p in pts]
    ys = [p['y_mean'] for p in pts]
    xerr = [p['x_std'] for p in pts]
    yerr = [p['y_std'] for p in pts]
    plt.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt='o', label=series_label, capsize=3, alpha=0.8)
    for p in pts:
      plt.annotate(p['label'], (p['x_mean'], p['y_mean']),
                   textcoords='offset points', xytext=(5, 4), fontsize=8, alpha=0.7)
    all_points.extend((p, series_label) for p in pts)

  if not args.no_frontier and all_points:
    front = pareto_front([p for p, _ in all_points])
    xs = [p['x_mean'] for p in front]
    ys = [p['y_mean'] for p in front]
    plt.plot(xs, ys, 'k--', alpha=0.5, label='Pareto frontier')

  plt.xlabel(args.x_label)
  plt.ylabel(args.y_label)
  plt.title(args.title)
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(out_path, dpi=200)
  plt.close()
  print('wrote', out_path)


if __name__ == '__main__':
  main()
