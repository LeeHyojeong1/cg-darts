#!/usr/bin/env python
"""Measure per-primitive latency on the current CUDA device.

Walks the same (channel, H, W, stride) tuples that
`cnn/cost_utils.build_search_costs` enumerates during DARTS supernet
construction (8-cell, init_channels=16, input 32x32). For each tuple it
constructs the primitive, warms it up, then times N iterations with
torch.cuda.Event synchronisation.

Output: JSON with shape {"edge_idx": [lat_op0, lat_op1, ...], ...} for
both 'normal' and 'reduce' cells, plus metadata. Consumed by
`cost_utils.build_search_costs(metric='lut', lut_path=...)`.

Usage:
  python scripts/measure_latency_lut.py --device-name jetson_orin \
      --out reports/latency_lut/jetson_orin.json --iters 50
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))

from genotypes import PRIMITIVES  # noqa: E402
from operations import OPS  # noqa: E402


def edge_specs(steps):
  return [(i, j) for i in range(steps) for j in range(2 + i)]


def out_hw(hw, stride):
  return hw if stride == 1 else (hw + 1) // 2


def build_op(primitive, c, stride):
  op = OPS[primitive](c, stride, False)
  if 'pool' in primitive:
    op = torch.nn.Sequential(op, torch.nn.BatchNorm2d(c, affine=False))
  return op


def time_op(op, x, iters, warmup):
  device = x.device
  op = op.to(device).eval()
  with torch.no_grad():
    for _ in range(warmup):
      op(x)
  if device.type == 'cuda':
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    with torch.no_grad():
      for _ in range(iters):
        op(x)
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / 1000.0 / iters  # seconds per call
  t0 = time.perf_counter()
  with torch.no_grad():
    for _ in range(iters):
      op(x)
  return (time.perf_counter() - t0) / iters


def measure(init_channels, layers, input_size, steps, iters, warmup, device, batch):
  spec = edge_specs(steps)
  num_ops = len(PRIMITIVES)
  tables = {'normal': {}, 'reduce': {}}

  c_curr = init_channels
  hw = input_size
  reduction_layers = [layers // 3, 2 * layers // 3]

  # We measure per-layer, then *sum* into the alpha-shaped table because
  # alpha is shared across cells of the same type in the supernet.
  normal = [[0.0] * num_ops for _ in range(len(spec))]
  reduce = [[0.0] * num_ops for _ in range(len(spec))]

  for layer in range(layers):
    reduction = layer in reduction_layers
    if reduction:
      c_curr *= 2
    cell_output_hw = out_hw(hw, 2) if reduction else hw
    target = reduce if reduction else normal
    for edge_idx, (_, src) in enumerate(spec):
      stride = 2 if reduction and src < 2 else 1
      op_in_hw = hw if reduction and src < 2 else cell_output_hw
      x = torch.zeros(batch, c_curr, op_in_hw, op_in_hw, device=device)
      for op_idx, primitive in enumerate(PRIMITIVES):
        if primitive == 'none':
          continue  # zero op has no real cost
        op = build_op(primitive, c_curr, stride)
        try:
          lat = time_op(op, x, iters, warmup)
        except Exception as e:
          print('  skip {} c={} hw={} stride={} ({})'.format(
            primitive, c_curr, op_in_hw, stride, e), file=sys.stderr)
          lat = 0.0
        target[edge_idx][op_idx] += lat
        del op
        if device.type == 'cuda':
          torch.cuda.empty_cache()
      print('  layer={} edge={} c={} hw={} stride={} measured'.format(
        layer, edge_idx, c_curr, op_in_hw, stride))
    hw = cell_output_hw

  tables['normal'] = normal
  tables['reduce'] = reduce
  return tables


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--init-channels', type=int, default=16)
  ap.add_argument('--layers', type=int, default=8)
  ap.add_argument('--steps', type=int, default=4)
  ap.add_argument('--input-size', type=int, default=32)
  ap.add_argument('--iters', type=int, default=50)
  ap.add_argument('--warmup', type=int, default=5)
  ap.add_argument('--batch', type=int, default=1)
  ap.add_argument('--device-name', default='measured',
                  help='label written to JSON metadata (e.g. l40s, jetson_orin)')
  ap.add_argument('--gpu', type=int, default=0)
  ap.add_argument('--out', default=str(ROOT / 'reports' / 'latency_lut' / 'lut.json'))
  args = ap.parse_args()

  device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
  print('measuring on', device)
  tables = measure(
    args.init_channels, args.layers, args.input_size, args.steps,
    args.iters, args.warmup, device, args.batch)

  out = {
    'meta': {
      'device_name': args.device_name,
      'gpu': args.gpu,
      'init_channels': args.init_channels,
      'layers': args.layers,
      'steps': args.steps,
      'input_size': args.input_size,
      'iters': args.iters,
      'warmup': args.warmup,
      'batch': args.batch,
      'primitives': PRIMITIVES,
    },
    'normal': tables['normal'],
    'reduce': tables['reduce'],
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(out, indent=2))
  print('wrote', out_path)


if __name__ == '__main__':
  main()
