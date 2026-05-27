#!/usr/bin/env bash
# Compare first-order vs second-order CG-DARTS at the headline lambda.
#
# Hypothesis: cost-aware search may interact with the unrolled gradient. The
# original DARTS paper shows second-order ~0.24pp better than first-order
# (2.76 vs 3.00 CIFAR-10 error). Whether that gap persists or widens under
# the cost regularizer is unknown.
#
# This script runs both first-order (UNROLLED=0) and second-order (UNROLLED=1)
# at LAMBDA=$LAMBDA across the requested seeds, in parallel across two GPUs.
#
# Env vars (in addition to the ones used by run_multi_seed.sh):
#   LAMBDA   - the cost weight to sweep, default 1e-2
#   METRIC   - flops | params | mem | device, default flops
#   SEEDS    - default "0 1 2"
#   EPOCHS   - default 50
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMBDA="${LAMBDA:-1e-2}"
METRIC="${METRIC:-flops}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-50}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-100}"
SKIP_RETRAIN="${SKIP_RETRAIN:-0}"

echo "[$(date)] First-order"
LAMBDAS="${LAMBDA}" SEEDS="${SEEDS}" METRIC="${METRIC}" EPOCHS="${EPOCHS}" \
  RETRAIN_EPOCHS="${RETRAIN_EPOCHS}" SKIP_RETRAIN="${SKIP_RETRAIN}" \
  UNROLLED=0 "${ROOT_DIR}/scripts/run_multi_seed.sh"

echo "[$(date)] Second-order"
LAMBDAS="${LAMBDA}" SEEDS="${SEEDS}" METRIC="${METRIC}" EPOCHS="${EPOCHS}" \
  RETRAIN_EPOCHS="${RETRAIN_EPOCHS}" SKIP_RETRAIN="${SKIP_RETRAIN}" \
  UNROLLED=1 "${ROOT_DIR}/scripts/run_multi_seed.sh"

echo "[$(date)] Done."
