#!/usr/bin/env python
"""Re-derive the genotype from a saved search run using a different mode.

After CG-DARTS finishes, the architecture parameters live in
`weights.pt`. The original `genotype.txt` was derived with whatever
--discretize_mode was passed at search time. This script loads the
saved alphas and produces a *new* genotype using a chosen mode, so we
can compare argmax / cost_sub / cost_div over the *same* search alphas.

Writes the new genotype next to the search dir as
`genotype.<mode>.<tag>.txt`.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))

from cost_utils import build_search_costs  # noqa: E402
from model_search import Network  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--search-dir', required=True)
  ap.add_argument('--mode', required=True, choices=['argmax', 'cost_sub', 'cost_div'])
  ap.add_argument('--cost-metric', default='flops', choices=['flops', 'params', 'lut'])
  ap.add_argument('--cost-weight', type=float, default=1.0,
                  help='mu for mode=cost_sub')
  ap.add_argument('--tau', type=float, default=1.0,
                  help='softmax temperature to apply at derivation time')
  ap.add_argument('--init-channels', type=int, default=16)
  ap.add_argument('--layers', type=int, default=8)
  ap.add_argument('--cost-normalize', default='edge')
  ap.add_argument('--lut-path', default='')
  ap.add_argument('--tag', default='')
  args = ap.parse_args()

  sd = Path(args.search_dir)
  weights = sd / 'weights.pt'
  if not weights.exists():
    sys.exit('no weights.pt in {}'.format(sd))

  criterion = nn.CrossEntropyLoss()
  model = Network(args.init_channels, 10, args.layers, criterion)
  state = torch.load(str(weights), map_location='cpu', weights_only=True)
  model.load_state_dict(state)

  costs = build_search_costs(
    args.init_channels, args.layers, steps=model._steps,
    input_size=32, metric=args.cost_metric,
    normalize=args.cost_normalize,
    lut_path=(args.lut_path or None),
  )
  geno = model.genotype(
    cost_normal=costs.normal,
    cost_reduce=costs.reduce,
    mode=args.mode,
    cost_weight=args.cost_weight,
    tau=args.tau,
  )
  suffix = '.{}{}.txt'.format(args.mode, '.' + args.tag if args.tag else '')
  out = sd / ('genotype' + suffix)
  out.write_text(repr(geno) + '\n')
  print('wrote', out)
  print(geno)


if __name__ == '__main__':
  main()
