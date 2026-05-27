"""Hardware profiles for device-conditioned CG-DARTS.

Each profile mixes three edge-normalised cost signals:
  cost = w_flops * flops + w_mem * mem + w_lat * latency_proxy

Latency uses a roofline model (seconds, relative across ops on the same device):
  lat = max(flops / peak_fp32_tflops, mem_bytes / mem_bandwidth)

Literature specs are used for peak FP32 TFLOPS and memory bandwidth.
Replace latency slice with measured micro-benchmark LUTs when hardware is available.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
  name: str
  label: str
  peak_tflops_fp32: float
  mem_bandwidth_gbps: float
  w_flops: float
  w_mem: float
  w_lat: float
  sku: str = ""
  source: str = ""
  notes: str = ""


# Mixing weights follow CG_DARTS Proposal v3 presets:
#   edge  -> (0.20, 0.60, 0.20)
#   server -> (0.50, 0.20, 0.30)
# Middle-end uses a balanced preset between the two.
DEVICE_PROFILES = {
  "jetson_orin": DeviceProfile(
    name="jetson_orin",
    label="Edge (Jetson AGX Orin 64GB)",
    peak_tflops_fp32=5.32,
    mem_bandwidth_gbps=204.8,
    w_flops=0.20,
    w_mem=0.60,
    w_lat=0.20,
    sku="Jetson AGX Orin 64GB module",
    source="NVIDIA DS-10662-001 / Jetson AGX Orin Technical Brief",
    notes="CUDA FP32 up to 5.32 TFLOPS; 256-bit LPDDR5 bandwidth up to 204.8 GB/s.",
  ),
  "rtx_pro_6000": DeviceProfile(
    name="rtx_pro_6000",
    label="Middle-end (RTX PRO 6000 Blackwell Workstation)",
    peak_tflops_fp32=125.0,
    mem_bandwidth_gbps=1792.0,
    w_flops=0.35,
    w_mem=0.35,
    w_lat=0.30,
    sku="RTX PRO 6000 Blackwell Workstation Edition",
    source="NVIDIA RTX PRO 6000 Blackwell Workstation Edition product specs",
    notes="FP32 125 TFLOPS; 96 GB GDDR7; memory bandwidth 1792 GB/s.",
  ),
  "h100": DeviceProfile(
    name="h100",
    label="High-end (H100 SXM5)",
    peak_tflops_fp32=67.0,
    mem_bandwidth_gbps=3350.0,
    w_flops=0.50,
    w_mem=0.20,
    w_lat=0.30,
    sku="NVIDIA H100 SXM5",
    source="NVIDIA H100 product specifications",
    notes="FP32 67 TFLOPS; 80 GB HBM3; GPU memory bandwidth 3.35 TB/s (3350 GB/s).",
  ),
}


def get_device_profile(device_name):
  key = device_name.lower()
  if key not in DEVICE_PROFILES:
    valid = ", ".join(sorted(DEVICE_PROFILES))
    raise ValueError("Unknown device '{}'. Choose from: {}".format(device_name, valid))
  return DEVICE_PROFILES[key]
