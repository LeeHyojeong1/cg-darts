# Device-Conditioned CG-DARTS Report

CIFAR-10, seed=2, $\lambda=10^{-2}$, search 50 epochs (warmup 10), retrain 100 epochs.

## Method (per proposal)

Each hardware tier uses a **device profile** with literature peak FP32 TFLOPS, memory
bandwidth, and tier-specific mixing weights $(w_F, w_M, w_L)$:

| Tier | Device | Peak FP32 | Mem BW | $(w_F, w_M, w_L)$ |
|------|--------|-----------|--------|-------------------|
| Edge (Orin) | Jetson AGX Orin 64GB module | 5.3 TFLOPS | 205 GB/s | (0.20, 0.60, 0.20) |
| Middle (RTX PRO 6000) | RTX PRO 6000 Blackwell Workstation Edition | 125.0 TFLOPS | 1792 GB/s | (0.35, 0.35, 0.30) |
| High-end (H100) | NVIDIA H100 SXM5 | 67.0 TFLOPS | 3350 GB/s | (0.50, 0.20, 0.30) |

Architecture loss: $L_{arch} = L_{val} + \lambda \cdot \mathbb{E}[cost(\alpha)]$

Per-op device cost combines edge-normalised FLOPs, activation-memory proxy, and
roofline latency $\max(F/\mathrm{peak}, M/\mathrm{BW})$ on the target device.

## Results summary

| Tier | Search val acc | E[cost] | MACs (M) | Params (M) | Retrain val acc |
|------|----------------|---------|----------|------------|-----------------|
| Edge (Jetson AGX Orin 64GB) | 87.51% | 7.304 | 313.0 | 1.936 | 95.40% |
| Middle-end (RTX PRO 6000 Blackwell Workstation) | 87.38% | 6.953 | 340.4 | 2.116 | 95.57% |
| High-end (H100 SXM5) | 87.58% | 7.048 | 431.3 | 2.712 | 95.52% |

### Baselines (existing FLOPs CG-DARTS)

- Vanilla DARTS retrain: 95.97%, 478.0M MACs, 3.021M params
- CG-DARTS FLOPs $\lambda=10^{-2}$: 96.05%, 420.1M MACs, 2.630M params

## Figures

- `device_cost_weights.png` — tier mixing weights from proposal
- `device_expected_cost_over_epochs.png` — cost trajectory during search
- `device_search_accuracy_vs_cost.png` — search Pareto view
- `device_tier_comparison.png` — accuracy / MACs / params by tier
- `device_retrain_accuracy_vs_macs.png` — retrain trade-off vs Vanilla & FLOPs CG-DARTS
- `device_retrain_accuracy_vs_params.png` — same vs parameter count
