#!/usr/bin/env python
"""Cross-device evaluation matrix for discovered genotypes.

For each (search_dir, device_profile) pair, build the discrete eval network
(`cnn/model.NetworkCIFAR`) from the saved genotype and compute:

  * params (M)
  * MACs   (M)              # analytical FLOPs
  * activation memory (M B) # fp32 bytes proxy
  * roofline latency (ms)   # per device_profile: max(flops/tflops, bytes/bw)
  * measured latency (ms)   # actual fwd time on the current CUDA device

The result is a wide CSV: rows are search_dirs, columns are
{params, macs, mem_mb, measured_ms} and per-device roofline columns. This
exposes whether a cell searched on (say) the Jetson profile is actually
faster on a Jetson-like roofline than the H100-searched cell.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))

import genotypes  # noqa: E402
from device_profiles import DEVICE_PROFILES  # noqa: E402
from model import NetworkCIFAR  # noqa: E402


CIFAR_CLASSES = 10
BYTES_PER_ELEMENT = 4.0


def read_genotype(path):
  text = Path(path).read_text().strip()
  return eval(text, {'Genotype': genotypes.Genotype, 'range': range})


def count_macs_bytes(model, input_shape=(1, 3, 32, 32)):
  totals = {'macs': 0, 'act_bytes': 0}
  handles = []

  def conv_hook(module, inputs, output):
    out = output.shape
    in_shape = inputs[0].shape
    kH, kW = module.kernel_size
    macs = out[0] * out[1] * out[2] * out[3] * (module.in_channels // module.groups) * kH * kW
    totals['macs'] += macs
    # activation traffic: input read + output write
    totals['act_bytes'] += BYTES_PER_ELEMENT * (in_shape.numel() + output.numel())

  def linear_hook(module, inputs, output):
    totals['macs'] += inputs[0].shape[0] * module.in_features * module.out_features
    totals['act_bytes'] += BYTES_PER_ELEMENT * (inputs[0].numel() + output.numel())

  def pool_hook(module, inputs, output):
    k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
    totals['macs'] += output.numel() * k * k
    totals['act_bytes'] += BYTES_PER_ELEMENT * (inputs[0].numel() + output.numel())

  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      handles.append(m.register_forward_hook(conv_hook))
    elif isinstance(m, nn.Linear):
      handles.append(m.register_forward_hook(linear_hook))
    elif isinstance(m, (nn.AvgPool2d, nn.MaxPool2d)):
      handles.append(m.register_forward_hook(pool_hook))

  model.eval()
  model.drop_path_prob = 0.0
  with torch.no_grad():
    model(torch.zeros(*input_shape))
  for h in handles:
    h.remove()
  return totals['macs'], totals['act_bytes']


def measure_latency(model, device, iters=50, warmup=10):
  model = model.to(device).eval()
  model.drop_path_prob = 0.0
  x = torch.zeros(1, 3, 32, 32, device=device)
  with torch.no_grad():
    for _ in range(warmup):
      model(x)
  if device.type == 'cuda':
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    with torch.no_grad():
      for _ in range(iters):
        model(x)
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / iters  # ms
  t0 = time.perf_counter()
  with torch.no_grad():
    for _ in range(iters):
      model(x)
  return (time.perf_counter() - t0) * 1000.0 / iters


def roofline_ms(macs, act_bytes, profile):
  # macs == FLOPs/2 conventionally, but cost_utils uses MACs as FLOPs; mirror that.
  compute_s = macs / (profile.peak_tflops_fp32 * 1e12)
  mem_s = act_bytes / (profile.mem_bandwidth_gbps * 1e9)
  return 1000.0 * max(compute_s, mem_s)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--search-dir', action='append', required=True,
                  help='path to a search-* directory containing genotype.txt; may repeat')
  ap.add_argument('--init-channels', type=int, default=36)
  ap.add_argument('--layers', type=int, default=20)
  ap.add_argument('--auxiliary', action='store_true', default=True)
  ap.add_argument('--gpu', type=int, default=0)
  ap.add_argument('--iters', type=int, default=50)
  ap.add_argument('--out', default=str(ROOT / 'reports' / 'cross_device' / 'matrix.csv'))
  args = ap.parse_args()

  device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
  rows = []
  for sd in args.search_dir:
    sd = Path(sd)
    gpath = sd / 'genotype.txt'
    if not gpath.exists():
      print('skip {}: no genotype.txt'.format(sd), file=sys.stderr)
      continue
    geno = read_genotype(gpath)
    model = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, geno)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    macs, act_bytes = count_macs_bytes(model)
    measured_ms = measure_latency(model, device, args.iters)
    row = {
      'search_dir': str(sd),
      'params_m': params_m,
      'macs_m': macs / 1e6,
      'act_mb': act_bytes / 1e6,
      'measured_ms_cur_gpu': measured_ms,
    }
    for name, prof in DEVICE_PROFILES.items():
      row['roofline_ms_{}'.format(name)] = roofline_ms(macs, act_bytes, prof)
    rows.append(row)
    print(row)
    del model
    if device.type == 'cuda':
      torch.cuda.empty_cache()

  if not rows:
    print('no rows; nothing to write', file=sys.stderr)
    sys.exit(1)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys())
  with out_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
      w.writerow(r)
  print('wrote', out_path)


if __name__ == '__main__':
  main()
