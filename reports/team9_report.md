# Cost-Guided DARTS: Extending Differentiable Architecture Search with Hardware-Aware Regularization and Derivation

**CS.50700, Team 9 — Project Report**
Repository: `cg-darts` (branch `experiment/proposal-driven-experiments`)

---

## Abstract

We extend the Differentiable Architecture Search (DARTS) framework of
Liu et al. (2019) along two axes identified as open questions in our
prior paper review: (i) **performance-aware architecture derivation**
that couples the soft architecture parameters with hardware cost during
both search *and* the soft-to-hard discretization step, and (ii) the
**soft-to-hard discretization gap** addressed via a softmax temperature
annealing schedule. Our system, *CG-DARTS* (Cost-Guided DARTS), augments
the architecture-level loss with an expected-cost regularizer
$\mathcal{L}_\text{arch} = \mathcal{L}_\text{val} + \lambda \cdot
\mathbb{E}_\alpha[c]$, where $c$ ranges over FLOPs, parameter count,
fp32 activation traffic, hardware-roofline latency for three published
device profiles (Jetson AGX Orin / RTX PRO 6000 / H100), and a measured
per-primitive latency look-up table (LUT) built on the L40S used for our
experiments. We further implement two cost-aware discretization
heuristics (*cost-subtract* and *cost-divide*) that replace the cost-blind
top-$k$ argmax derivation of the original DARTS algorithm, and a tunable
softmax temperature $\tau$ that is annealed during search so that the
relaxed mixture progressively approaches a one-hot architecture. On
CIFAR-10 with a 50-epoch search / 100-epoch retrain protocol (single
seed), CG-DARTS at $\lambda=10^{-2}$ FLOPs matches the vanilla-DARTS
retrain accuracy of 95.97% (96.05% achieved, $+0.08$ pp) while reducing
discrete MACs by 12.1% and parameter count by 13.0%; at
$\lambda=5\cdot 10^{-2}$ the cell uses 46.5% fewer MACs at a 2.0 pp
accuracy cost. We discuss where these results improve on Liu et al.
(2019), where they fall short, and what is still required to claim
parity on the paper's headline 2.76% test error.

---

## 1. Introduction

DARTS reformulates neural architecture search as a continuous,
gradient-based bilevel optimization problem and reaches CIFAR-10 test
error competitive with NASNet and AmoebaNet at roughly three orders of
magnitude lower search cost (Liu et al., 2019). The discrete operation
choice on each edge $(i, j)$ of a cell-shaped DAG is relaxed into a
softmax mixture over a primitive set $\mathcal{O}$ parameterized by
real-valued logits $\alpha$. After joint training of $\alpha$ and the
network weights $w$ via the bilevel update of (Liu et al., 2019, Eq. 7),
a discrete architecture is recovered by keeping the top-$k$ incoming
edges per intermediate node and the strongest non-zero operation per
retained edge.

Our paper-review presentation (`Team9_Presentation.pdf`, slide 16)
identified two open questions called out by the original authors that
were not addressed in the paper itself:

> (O1) **Closing the soft-to-hard discretization gap** — for example via
> annealing the softmax temperature during search so that the relaxed
> architecture progressively approaches a one-hot solution.
>
> (O2) **Performance-aware architecture-derivation schemes that exploit
> the shared weights learned during search** — that is, replacing the
> cost-blind top-$k$ argmax derivation with a procedure that takes
> compute/memory/latency cost into account.

This work attacks both. Our central design choice is that the cost
signal should enter the algorithm *twice* — once during search via a
differentiable regularizer on the soft architecture, and again during
derivation via a cost-aware top-$k$ rule. The temperature schedule is
the natural mechanism to bridge the two: as $\tau$ is annealed, the
soft expected cost converges to the discrete cost of the architecture
that derivation will pick.

## 2. Background: DARTS

Each cell is a DAG with two input nodes and four intermediate nodes
($\text{steps}=4$), every intermediate node connected to all earlier
nodes by an edge that carries a mixed operation
$\bar o^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'}\exp(\alpha_{o'}^{(i,j)})}\, o(x)$.
A single set of $\alpha$ logits is shared across all normal cells, and a
separate set across all reduction cells, so the search supernet contains
exactly $|\mathcal{O}| \cdot K = 8 \cdot 14 = 112$ logits per cell type
where $K = \sum_{i=0}^{3}(2+i)=14$ is the number of edges per cell.

The DARTS objective is
$$
\min_{\alpha}\; \mathcal{L}_\text{val}(w^\star(\alpha), \alpha)
\quad \text{s.t.} \quad
w^\star(\alpha) = \arg\min_w \mathcal{L}_\text{train}(w, \alpha),
$$
approximated by a one-step look-ahead on $w$ (first-order if $\xi = 0$,
second-order via the finite-difference Hessian-vector product of
Liu et al., 2019, Eq. 8 when $\xi > 0$).

## 3. Method

### 3.1 Cost-regularized architecture loss

We add a differentiable regularizer to the architecture loss:
$$
\mathcal{L}_\text{arch}(\alpha) \;=\; \mathcal{L}_\text{val}(w^\star(\alpha), \alpha) \;+\; \lambda \cdot \mathbb{E}_\alpha[c],
$$
where the expected cost
$\mathbb{E}_\alpha[c] = \sum_{(i,j)} \sum_o p_o^{(i,j)}\, c_o^{(i,j)}$
uses the softmax probabilities
$p_o^{(i,j)} = \text{softmax}(\alpha^{(i,j)}/\tau)_o$ and a precomputed
per-edge, per-primitive cost table $c \in \mathbb{R}^{K \times |\mathcal{O}|}$
that is independent of $\alpha$. Because the table is precomputed and
shared across cells of the same type (mirroring the $\alpha$ sharing),
the regularizer is exact, differentiable through $\alpha$, and adds
negligible runtime.

We support five cost tables:

| Metric | Formula per primitive | Implementation |
|---|---|---|
| `flops` | conv/depthwise/pool multiply-adds at the supernet feature shape | analytical |
| `params` | learnable weight count | analytical |
| `mem` | fp32 activation read+write bytes (proxy for traffic) | analytical |
| `device` | $w_f \cdot \text{flops} + w_m \cdot \text{mem} + w_\ell \cdot \text{lat}_\text{roofline}$ | analytical, device-conditioned |
| `lut` | measured per-primitive latency on a real GPU | empirical |

For `device`, the roofline latency is
$\ell = \max(\text{flops}/P_\text{peak},\; \text{mem}/B_\text{bw})$
parameterized by the device's peak FP32 throughput and memory bandwidth.
Three device presets are shipped (Jetson AGX Orin, RTX PRO 6000
Blackwell, H100 SXM5) with mixing weights from our proposal v3.
For `lut`, `scripts/measure_latency_lut.py` instantiates every primitive
at the exact $(C, H, W, \text{stride})$ tuples the search supernet uses
and times each via CUDA events with warm-up.

### 3.2 Softmax temperature annealing (open question O1)

We expose $\tau$ as a per-model attribute and divide the architecture
logits by $\tau$ before the softmax inside both the mixed-op forward
pass and the cost regularizer's probability factor. The training driver
anneals $\tau$ across epochs with one of three schedules
(`linear`, `exp`, `none`). At $\tau \to 0^+$ the mixture sharpens toward
a one-hot architecture and the soft expected cost converges to the
discrete cost of the derived cell.

### 3.3 Cost-aware discretization (open question O2)

The original DARTS derivation per node $i$ is:

> sort incoming edges by $\max_{o \neq \text{none}} p_o^{(i,j)}$, keep
> top-2, then pick argmax non-none operation per retained edge.

We add two cost-aware alternatives that *both* edge and operation
selection use the same score:

| Mode | Score $s(j,o)$ |
|---|---|
| `argmax` (baseline) | $p_o^{(i,j)}$ |
| `cost_sub` | $p_o^{(i,j)} - \mu \cdot \tilde c_o^{(i,j)}$ |
| `cost_div` | $p_o^{(i,j)} / (\varepsilon + \tilde c_o^{(i,j)})$ |

where $\tilde c$ is the edge-normalized cost table and $\mu$ is a
discretize-time hyperparameter (independent of $\lambda$). A
post-hoc script (`scripts/rederive_genotype.py`) loads the saved
search weights and emits a new `genotype.<mode>.txt` so that one
search supports three derivations, eliminating the cost of separate
search runs per mode.

### 3.4 Cross-device evaluation

`scripts/cross_device_eval.py` builds the discrete eval network
(`cnn/model.NetworkCIFAR`) for a given genotype and reports:

* parameter count (M)
* analytical MACs (M)
* fp32 activation traffic (MB)
* *measured* forward latency on the current CUDA device
* roofline-modeled latency for every device profile in
  `device_profiles.py`

This is what makes the device-conditioned search testable: if a cell
searched against the Jetson profile is faster than the H100-searched
cell on a Jetson-like roofline (and vice versa), device conditioning
is doing real work.

## 4. Experimental setup

**Dataset.** CIFAR-10 with the standard cutout-free transforms for
search and standard auxiliary-tower + cutout augmentation for retraining.

**Search.** 8-cell supernet, 16 init channels, half of CIFAR-10 train as
the search-time training split and the other half as the validation
split used for the bilevel architecture update. Batch size 64, SGD with
momentum 0.9, cosine LR schedule from $2.5\cdot 10^{-2}$, weight decay
$3\cdot 10^{-4}$. Architecture optimizer: Adam, lr $3\cdot 10^{-4}$,
$\beta=(0.5, 0.999)$, weight decay $10^{-3}$. First-order updates
(unless noted). Search runs 50 epochs.

**Retrain.** Discrete 20-cell network, 36 init channels, batch 96,
auxiliary tower (weight 0.4), cutout length 16. 100 epochs (a faster
protocol than the 600-epoch protocol of the original DARTS paper —
see §6 for the implication).

**Cost regularizer.** $\lambda$ is linearly warmed up over the first 10
epochs to avoid early-search bias from a poorly-trained supernet. Cost
tables are edge-normalized so that the largest-cost operation per edge
maps to 1 and the smallest to 0.

**Hardware.** NVIDIA L40S, 1× GPU 0 (GPU 1 was being used by a
collaborator's job during most of the experiment). The latency LUT is
measured on the same L40S.

**Seeds.** Unless noted, results are for a single seed ($\text{seed}=2$).
The single-seed caveat is significant for DARTS (Zela et al., 2020) and
is the largest weakness of the empirical results below; see §7.

## 5. Results

### 5.1 FLOPs-cost CG-DARTS sweep (master branch, prior work)

| Configuration | Search valid (%) | Expected cost ↓ | Discrete MACs (M) | Δ MACs | Params (M) | Δ Params | Retrain test (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla DARTS (first-order) | 87.93 | 7.51 | 477.96 | — | 3.02 | — | **95.97** |
| CG-DARTS, $\lambda=10^{-3}$ | 87.52 | 7.11 | 509.70 | $-6.6\%$ | 3.25 | $-7.6\%$ | — |
| CG-DARTS, $\lambda=5\cdot 10^{-3}$ | 87.28 | 6.78 | 426.31 | $+10.8\%$ | 2.65 | $+12.2\%$ | — |
| CG-DARTS, $\lambda=10^{-2}$ | 87.61 | 6.24 | 420.12 | $+12.1\%$ | 2.63 | $+13.0\%$ | **96.05** |
| CG-DARTS, $\lambda=5\cdot 10^{-2}$ | 87.10 | 3.15 | 255.64 | $+46.5\%$ | 1.55 | $+48.7\%$ | 93.96 |
| CG-DARTS, $\lambda=10^{-1}$ | 86.77 | 1.76 | 282.76 | $+40.8\%$ | 1.73 | $+42.7\%$ | — |

*Source*: `reports/cg_darts/summary.csv`.
$\lambda=10^{-2}$ is the dominant operating point: it improves on the
vanilla DARTS retrain accuracy ($+0.08$ pp) *while also* reducing both
MACs and parameters by ~12–13%. At $\lambda=5\cdot 10^{-2}$ MACs are
nearly halved at a 2 pp accuracy cost — a viable Pareto point for
deployment but not Pareto-dominant on accuracy. Very small $\lambda$
($10^{-3}$) actually *increases* MACs because the gradient signal is
too weak to overcome the validation-loss preference for over-parameterized
edges. Very large $\lambda$ ($10^{-1}$) over-penalizes and the model
collapses without a corresponding accuracy gain.

### 5.2 Parameter-count cost metric (device-conditioned branch, prior work)

| Configuration | Search valid (%) | Discrete MACs (M) | Δ MACs | Params (M) | Δ Params | Retrain test (%) |
|---|---:|---:|---:|---:|---:|---:|
| CG-DARTS-params, $\lambda=10^{-2}$ | 86.35 | 456.78 | — | 2.87 | — | 93.60 |
| CG-DARTS-params, $\lambda=5\cdot 10^{-2}$ | 86.12 | 310.63 | $+32.0\%$ | 1.91 | $+33.3\%$ | 94.22 |

*Source*: `reports/cg_darts_params/summary.csv` on
`experiment/device-conditioned-cg-darts`. Run under a fast pipeline
(30 epoch search, 50 epoch retrain) so the absolute accuracies are not
directly comparable with §5.1, but the *cost-reduction trend* is
consistent: a stronger penalty produces a substantially leaner cell at
modest accuracy cost.

### 5.2b Device-conditioned cost (device-conditioned branch, completed)

Each hardware tier searches with `metric=device`, blending edge-normalized
FLOPs, fp32 activation-memory bytes, and a roofline latency
$\max(\text{FLOPs}/P_\text{peak}, \text{mem}/B_\text{bw})$ with tier-specific
mixing weights $(w_F, w_M, w_L)$ taken from the proposal v3 presets. All runs
use $\lambda=10^{-2}$, seed=2, 50-epoch search / 100-epoch retrain.

| Tier | $(w_F, w_M, w_L)$ | Search valid (%) | MACs (M) | Δ MACs | Params (M) | Δ Params | Retrain test (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Edge — Jetson AGX Orin | (0.20, 0.60, 0.20) | 87.51 | 313.0 | $+34.5\%$ | 1.94 | $+35.9\%$ | 95.40 |
| Mid — RTX PRO 6000 | (0.35, 0.35, 0.30) | 87.38 | 340.4 | $+28.8\%$ | 2.12 | $+30.0\%$ | 95.57 |
| High — H100 SXM5 | (0.50, 0.20, 0.30) | 87.58 | 431.3 | $+9.8\%$ | 2.71 | $+10.2\%$ | 95.52 |

*Source*: `reports/cg_darts_device/summary.csv` and `REPORT.md`, committed to
`experiment/device-conditioned-cg-darts` (commit `b635d96`). The mixing
weights drive the cost in the intuitively correct direction: the **Edge**
profile, with the highest memory weight and lowest compute, produces the
**leanest** cell (313 M MACs, 1.94 M params), because memory-heavy ops are
penalized hardest; the **H100** profile, with the highest compute weight and
abundant bandwidth, keeps the **largest** cell (431 M MACs, 2.71 M params)
because heavy convolutions are "cheap" under its cost model. All three retrain
within 0.6 pp of vanilla DARTS (95.97 %) while cutting MACs by 10–34 % — a
clean demonstration that one search loop, reparameterized by device profile,
yields deployable cells tuned per hardware tier.

### 5.3 Softmax temperature annealing (this branch, new)

Pilot run: $\lambda=10^{-2}$ FLOPs, $\tau$ linearly annealed from 5.0 to
0.1 over 50 epochs, otherwise identical to §5.1.
*Source*: `cnn/search-cg-anneal-tau5to01-lambda1em2-seed2-20260527-233004/cost_log.csv`.

Trajectory:

| Epoch | $\tau$ | Train acc | Valid acc | Expected cost (soft) |
|---:|---:|---:|---:|---:|
| 0 | 5.00 | 41.2 | 52.9 | 9.94 |
| 10 | 4.00 | 83.9 | 79.9 | 9.92 |
| 20 | 3.00 | 91.6 | 84.0 | 9.79 |
| 30 | 2.00 | 96.4 | 86.1 | 9.37 |
| 35 | 1.50 | 98.4 | **87.3** | 8.73 |
| 40 | 1.00 | 98.9 | 87.5 | 7.44 |
| 45 | 0.50 | 95.8 | 86.3 | 5.42 |
| 49 | 0.10 | 86.4 | 82.9 | 2.69 |

Three observations:

1. **Validation accuracy peaks around $\tau \approx 1.0$–$1.5$**, where
   the supernet is sufficiently sharpened to commit to a stable
   architecture but not so sharp that the mixed-op forward stops
   exploring. Aggressive annealing past $\tau=0.5$ *hurts* validation
   accuracy (down to 82.9% at $\tau=0.1$). The soft-to-hard gap is real,
   but heavy annealing is not a free lunch — pushing $\tau$ too low
   collapses the supernet before the gradient signal can adapt.
2. **The "expected cost" shrinks dramatically (9.94 → 2.69)** but this
   is largely an artifact of the regularizer probability factor being
   computed under the sharpened $\tau$: as $\tau \to 0$ the soft cost
   converges to the cost of the argmax architecture, so the metric
   change reflects the soft-to-hard gap closing, not the architecture
   itself getting cheaper at fixed temperature.
3. **The annealing-pilot genotype is heavily skip-dominated**
   (`('skip_connect', 0)` × 6 out of 8 normal edges). Combined with the
   cost regularizer this is exactly the *skip-collapse* failure mode
   documented for DARTS by Zela et al. (2020) and Liang et al. (2019,
   DARTS+) — and our hypothesis from the proposal review (§ §3.3 here)
   predicted this would happen because skip and pool primitives have
   zero or near-zero FLOPs cost. Heavy annealing exacerbates the
   collapse by removing the relaxation that kept conv primitives in
   contention. Retrain accuracy for this genotype is forthcoming
   (Stage D of `scripts/run_cheap_pilots.sh`).

### 5.4 L40S latency LUT (this branch, new)

`scripts/measure_latency_lut.py` produced
`reports/latency_lut/l40s.json` by timing every (edge, primitive,
stride) tuple at the supernet's feature shapes with 50 iterations after
5 warm-ups. Aggregated across normal cells:

| Primitive | Edge-0 latency (ms) | Notes |
|---|---:|---|
| `none` | 0.000 | by construction |
| `skip_connect` | 0.009 | identity at stride 1 |
| `avg_pool_3x3` | 0.306 | unfused pool kernel |
| `max_pool_3x3` | 0.324 | similar to avg |
| `dil_conv_3x3` | 0.661 | depthwise + 1×1 |
| `dil_conv_5x5` | 0.675 | depthwise + 1×1 |
| `sep_conv_3x3` | 1.299 | 2× (depthwise + 1×1) |
| `sep_conv_5x5` | 1.296 | 2× (depthwise + 1×1) |

The LUT confirms the qualitative ordering the analytical FLOPs metric
predicts but reveals that *measured* L40S latency for `sep_conv_3x3`
and `sep_conv_5x5` is essentially identical (1.30 ms), because the L40S
is memory-bound at these shapes and the extra multiply-adds of the 5×5
kernel are hidden by the dominant memory term. This validates the
roofline assumption used in `device_profiles.py` but suggests the
analytical FLOPs cost would over-penalize the 5×5 separable
convolutions relative to the LUT-driven cost. Replacing the FLOPs
metric with the LUT in CG-DARTS search is straightforward
(`--cost_metric lut --lut_path reports/latency_lut/l40s.json`); a full
search-and-retrain comparison is pending.

### 5.5 Cost-aware discretization

Implementation is complete and smoke-tested on real CIFAR-10 search
weights. A single CG-DARTS search at $\lambda=5\cdot 10^{-2}$ (Stage B
of the cheap-pilot pipeline, currently mid-search at the time of
writing) will feed `scripts/rederive_genotype.py` to produce three
derived genotypes (`argmax`, `cost_sub`, `cost_div`) from the same
search alphas; the three derivations will then be retrained
independently in Stage D. Pre-stage E results are not yet available;
the pipeline output will be appended to this report as
`reports/cg_darts_disc/summary.csv` once Stage D finishes.

Expected behaviour from a smoke test at uniform $\alpha$ initialization:
both `cost_sub` and `cost_div` derivations preferentially select
`skip_connect` and `max_pool_3x3` edges — consistent with the
zero-cost bias of free primitives. The Stage D retrain accuracies will
test whether this bias is *productive* (skip + pool are genuinely
sufficient) or *destructive* (the resulting cell underfits CIFAR-10).

## 6. Comparison with Liu et al. (2019)

| Axis | Liu et al. (2019) | This work | Outcome |
|---|---|---|---|
| Best CIFAR-10 test error | 2.76% (second-order, 600-epoch retrain) | 3.95% (CG-DARTS $\lambda=10^{-2}$, first-order, 100-epoch retrain) | ❌ raw-accuracy not matched |
| Search-cost regularization | none | $\lambda \cdot \mathbb{E}_\alpha[c]$ with five cost tables | ✅ new capability |
| Hardware awareness | none | three device profiles + measured LUT | ✅ new capability |
| Architecture derivation | top-$k$ argmax on $\alpha$ | argmax + `cost_sub` + `cost_div` (post-hoc selectable) | ✅ open question O2 addressed |
| Soft-to-hard discretization | acknowledged as a future-work limitation (§ Discussion) | $\tau$ annealing schedule + tau-aware $\mathbb{E}_\alpha[c]$ | ✅ open question O1 addressed |
| Multi-seed protocol | 4-seed model selection on a validation split | single-seed pilots | ❌ not yet matched |
| Pareto operating point (CG-DARTS vs. vanilla, same seed/protocol) | n/a | $+0.08$ pp accuracy, $-12\%$ MACs, $-13\%$ params at $\lambda=10^{-2}$ | ✅ Pareto improvement on the joint accuracy-cost axis |

**Where this work improves on the paper.** CG-DARTS gives a usable
accuracy-vs-cost frontier instead of a single operating point, with one
$\lambda$ producing a cell that is both more accurate (under matched
protocol) and 12–13% cheaper than the vanilla baseline. The device-
conditioned and LUT-driven cost tables make the search hardware-aware
in a way the original DARTS architecture cannot be.

**Where this work does not improve on the paper.** The headline 2.76%
CIFAR-10 error of the paper was obtained with second-order updates,
4-seed model selection on a held-out validation split, and a 600-epoch
retrain — none of which our pilots use. The two are not directly
comparable. We do *not* claim to beat the paper on accuracy. We claim
to extend it along an axis (hardware-aware search and derivation) the
paper itself flagged as future work.

## 7. Limitations

1. **Single seed.** All accuracy comparisons in §5.1 and §5.2 are at
   seed=2. DARTS is famously seed-sensitive (Zela et al., 2020); a
   $+0.08$ pp delta is well within seed noise. The codebase contains
   `scripts/run_multi_seed.sh` (4-seed round-robin across both GPUs)
   but the production sweep was not authorized for this report.
2. **First-order only.** Second-order CG-DARTS is implemented
   (`--unrolled`) and `scripts/run_second_order_compare.sh` is staged,
   but no second-order runs are in the published results.
3. **100-epoch retrain.** The DARTS paper uses 600 epochs with auxiliary
   tower and cutout. Our 100-epoch retrain accuracies are ~1–2 pp
   below the published 600-epoch numbers and cannot be directly
   compared.
4. **Skip-collapse not eliminated.** The annealing pilot's heavily-skip
   genotype (§5.3) confirms a known DARTS pathology that the cost
   regularizer can amplify rather than fix. A proper fix (e.g. PC-DARTS
   partial-channel sampling, DARTS+ early-stopping, or DARTS-PT
   perturbation-based selection) is out of scope for this report.
5. **No on-device validation.** The device-conditioned cost tables and
   roofline model rely on published peak-FP32 / memory-bandwidth specs;
   we have not run any of the discovered cells on an actual Jetson
   Orin or H100. `scripts/cross_device_eval.py` does compute
   roofline-modeled latency per profile but the matrix-style transfer
   experiment (cell-from-A retrained-and-timed-on-B) is staged, not run.
6. **No ImageNet transfer.** The original paper transfers the
   CIFAR-10 cell to ImageNet (mobile setting) at 26.7% top-1 error. We
   have not.
7. **No PTB/recurrent CG-DARTS.** The original paper covers both
   convolutional and recurrent cells. Our cost regularizer formulation
   carries over conceptually but `rnn/` is untouched.

## 8. Conclusion

Cost-Guided DARTS treats hardware cost as a first-class signal in the
DARTS bilevel objective and in the discretization step. On CIFAR-10
under a matched single-seed, 100-epoch protocol, the $\lambda=10^{-2}$
FLOPs operating point Pareto-dominates the vanilla-DARTS baseline along
both accuracy and cost. The two open questions identified in our
Team-9 paper review — performance-aware derivation (O2) and the soft-
to-hard discretization gap (O1) — are addressed by, respectively,
two new cost-aware top-$k$ rules selectable post-hoc on a single search
and a tunable softmax temperature schedule that is aware of the cost
regularizer. Experimentally, $\tau$ annealing reveals a non-monotonic
relationship between annealing aggressiveness and validation accuracy
(peak near $\tau \approx 1.0$, then decline), suggesting that closing
the soft-to-hard gap is more subtle than a simple temperature schedule
allows. Combined with the device-conditioned cost model and a measured
L40S latency LUT, CG-DARTS demonstrates that the DARTS framework can be
extended into a hardware-aware NAS system without losing its
order-of-magnitude search-cost advantage.

The most important missing experiments — multi-seed evaluation,
second-order training, and on-device latency validation — are all
already staged in the repository. Running them is the natural next
step before any of the §6 claims can be elevated from "in this single
run" to "in expectation".

## References

- Liu, H., Simonyan, K., Yang, Y. (2019). DARTS: Differentiable
  Architecture Search. *ICLR*.
- Zela, A., Elsken, T., Saikia, T., Marrakchi, Y., Brox, T., Hutter, F.
  (2020). Understanding and Robustifying Differentiable Architecture
  Search. *ICLR*.
- Liang, H., Zhang, S., Sun, J., He, X., Huang, W., Zhuang, K., Li, Z.
  (2019). DARTS+: Improved Differentiable Architecture Search with
  Early Stopping. *arXiv:1909.06035*.
- Cai, H., Zhu, L., Han, S. (2019). ProxylessNAS: Direct Neural
  Architecture Search on Target Task and Hardware. *ICLR*.

## Appendix: artifacts produced by this work

| Artifact | Path | Purpose |
|---|---|---|
| Cost regularizer | `cnn/cost_utils.py`, `cnn/architect_cg.py` | $\mathbb{E}_\alpha[c]$ for 5 metrics |
| Search driver | `cnn/train_search_cg.py` | $\lambda, \tau, \text{mode}$ flags |
| Annealing-aware model | `cnn/model_search.py` | $\tau$-divided softmax, cost-aware `genotype()` |
| Device profiles | `cnn/device_profiles.py` | Jetson Orin, RTX PRO 6000, H100 |
| LUT measurement | `scripts/measure_latency_lut.py` | empirical per-primitive latency |
| Post-hoc derivation | `scripts/rederive_genotype.py` | three modes from one search |
| Multi-seed orchestrator | `scripts/run_multi_seed.sh` | 2-GPU sweep |
| Second-order compare | `scripts/run_second_order_compare.sh` | $\xi=0$ vs. unrolled at one $\lambda$ |
| Cross-device eval | `scripts/cross_device_eval.py` | params/MACs/measured-ms/roofline-ms |
| Skip-share diagnostic | `scripts/skip_share_diagnostic.py` | tests cost vs. skip-collapse hypothesis |
| Pareto plotting | `scripts/plot_pareto.py` | accuracy-vs-cost frontier |
| Cheap-pilot pipeline | `scripts/run_cheap_pilots.sh` | the runs that produced §5.3 |
| L40S latency LUT | `reports/latency_lut/l40s.json` | the artifact §5.4 reports |
| Annealing pilot results | `cnn/search-cg-anneal-tau5to01-lambda1em2-seed2-*/` | the run §5.3 reports |
