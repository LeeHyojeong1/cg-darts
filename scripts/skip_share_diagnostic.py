#!/usr/bin/env python
"""Skip-connect / pool share diagnostic for CG-DARTS genotypes.

Hypothesis (slide-16 of the Team-9 review):
  Cost regularization may *increase* the share of parameter-free ops (skip,
  pool) because they have zero FLOPs / params / memory cost. This script
  reads search directories (each containing a genotype.txt), bins them by
  the lambda value parsed from the path, and reports the per-category op
  share so we can plot it against lambda.

Usage:
  python scripts/skip_share_diagnostic.py \
      --glob 'cnn/search-cg-*'   \
      --out reports/skip_share/summary.csv

The CSV plus a PNG plot are written to `--out`'s parent directory.
"""
import argparse
import csv
import glob as globmod
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))

import genotypes  # noqa: E402

CATEGORIES = {
  'none': 'none',
  'skip_connect': 'skip',
  'max_pool_3x3': 'pool',
  'avg_pool_3x3': 'pool',
  'sep_conv_3x3': 'conv',
  'sep_conv_5x5': 'conv',
  'sep_conv_7x7': 'conv',
  'dil_conv_3x3': 'conv',
  'dil_conv_5x5': 'conv',
  'conv_7x1_1x7': 'conv',
}


def parse_lambda(path):
  """Extract lambda from path tokens like 'lambda1em2', 'lambda5e-3', 'lambda1e-1'."""
  text = str(path)
  m = re.search(r'lambda(\d+)(?:em|e-)(\d+)', text)
  if m:
    base, exp = m.groups()
    return float('{}e-{}'.format(base, exp))
  m = re.search(r'lambda(\d+)p(\d+)e(\d+)', text)
  if m:
    return float('{}.{}e{}'.format(*m.groups()))
  m = re.search(r'lambda([0-9.eE+-]+)', text)
  if m:
    try:
      return float(m.group(1))
    except ValueError:
      return None
  return None


def parse_seed(path):
  m = re.search(r'seed(\d+)', str(path))
  return int(m.group(1)) if m else None


def parse_metric(path):
  for tag in ('flops', 'params', 'mem', 'device'):
    if tag in str(path):
      return tag
  return 'unknown'


def read_genotype(genotype_path):
  text = Path(genotype_path).read_text().strip()
  return eval(text, {'Genotype': genotypes.Genotype, 'range': range})


def op_counts(genotype):
  counter = Counter()
  for (op, _) in list(genotype.normal) + list(genotype.reduce):
    counter[CATEGORIES.get(op, 'other')] += 1
  total = sum(counter.values())
  return counter, total


def collect(patterns):
  rows = []
  for pat in patterns:
    for d in sorted(globmod.glob(pat)):
      gpath = os.path.join(d, 'genotype.txt')
      if not os.path.isfile(gpath):
        continue
      try:
        g = read_genotype(gpath)
      except Exception as e:
        print('skip {}: {}'.format(gpath, e), file=sys.stderr)
        continue
      counts, total = op_counts(g)
      rows.append({
        'search_dir': d,
        'lambda': parse_lambda(d),
        'seed': parse_seed(d),
        'metric': parse_metric(d),
        'total_ops': total,
        'skip': counts.get('skip', 0),
        'pool': counts.get('pool', 0),
        'conv': counts.get('conv', 0),
        'none': counts.get('none', 0),
        'skip_pct': 100.0 * counts.get('skip', 0) / total if total else 0.0,
        'pool_pct': 100.0 * counts.get('pool', 0) / total if total else 0.0,
        'conv_pct': 100.0 * counts.get('conv', 0) / total if total else 0.0,
      })
  return rows


def write_csv(rows, out_path):
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = ['search_dir', 'metric', 'lambda', 'seed', 'total_ops',
                'skip', 'pool', 'conv', 'none',
                'skip_pct', 'pool_pct', 'conv_pct']
  with out_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
      w.writerow(r)


def plot(rows, out_path):
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print('matplotlib not available; skipped plot')
    return
  # group by metric, then by lambda (averaged over seeds)
  by_metric = defaultdict(lambda: defaultdict(list))
  for r in rows:
    if r['lambda'] is None:
      continue
    by_metric[r['metric']][r['lambda']].append(r)
  out_path = Path(out_path).with_suffix('.png')
  plt.figure(figsize=(8, 5))
  for metric, by_lambda in by_metric.items():
    lambdas = sorted(by_lambda)
    skip_means = [
      sum(r['skip_pct'] for r in by_lambda[l]) / len(by_lambda[l])
      for l in lambdas
    ]
    plt.plot(lambdas, skip_means, marker='o', label='{} (skip%)'.format(metric))
  plt.xscale('log')
  plt.xlabel('cost lambda')
  plt.ylabel('% of edges using skip_connect')
  plt.title('Skip-connect share vs. lambda')
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  plt.close()
  print('wrote', out_path)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--glob', action='append', required=True,
                  help='glob pattern(s) for search-* directories; may repeat')
  ap.add_argument('--out', default=str(ROOT / 'reports' / 'skip_share' / 'summary.csv'))
  ns = ap.parse_args()
  rows = collect(ns.glob)
  if not rows:
    print('no genotype.txt found across {}'.format(ns.glob), file=sys.stderr)
    sys.exit(1)
  write_csv(rows, ns.out)
  plot(rows, ns.out)
  print('wrote', ns.out)
  for r in rows:
    print('{metric:>8s} lambda={lambda} seed={seed} skip={skip_pct:5.1f}% pool={pool_pct:5.1f}% conv={conv_pct:5.1f}%'.format(**r))


if __name__ == '__main__':
  main()
