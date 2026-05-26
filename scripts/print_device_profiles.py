#!/usr/bin/env python
"""Print device profiles used by device-conditioned CG-DARTS."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cnn'))
from device_profiles import DEVICE_PROFILES

if __name__ == '__main__':
  for name, profile in DEVICE_PROFILES.items():
    print('{} ({})'.format(profile.name, profile.label))
    print('  sku: {}'.format(profile.sku))
    print('  peak_fp32_tflops: {:.2f}'.format(profile.peak_tflops_fp32))
    print('  mem_bandwidth_gbps: {:.1f}'.format(profile.mem_bandwidth_gbps))
    print('  weights (F, M, L): ({:.2f}, {:.2f}, {:.2f})'.format(
      profile.w_flops, profile.w_mem, profile.w_lat))
    print('  source: {}'.format(profile.source))
    print('  notes: {}'.format(profile.notes))
    print('')
