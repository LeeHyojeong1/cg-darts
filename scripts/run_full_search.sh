#!/usr/bin/env bash
# Reproduce vanilla + CG-DARTS lambda sweep (paper-style 50 epochs, seed=2).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/members/ryeowook/miniconda3/bin/python}"
DATA="${DATA:-${ROOT_DIR}/data}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-2}"
DOWNLOAD="${DOWNLOAD:-0}"

cd "${ROOT_DIR}/cnn"
DOWNLOAD_ARG=()
if [[ "${DOWNLOAD}" == "1" ]]; then
  DOWNLOAD_ARG=(--download)
fi

RUN_CG_SWEEP="${RUN_CG_SWEEP:-1}"

echo "==> Vanilla DARTS search (${EPOCHS} epochs, seed=${SEED})"
"${PYTHON_BIN}" train_search.py \
  --data "${DATA}" \
  --gpu "${GPU}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --save "first-order-${EPOCHS}ep" \
  "${DOWNLOAD_ARG[@]}"

if [[ "${RUN_CG_SWEEP}" == "1" ]]; then
  echo "==> CG-DARTS lambda sweep"
  LAMBDAS="${LAMBDAS:-1e-3 5e-3 1e-2 5e-2 1e-1}"
  METRIC="${METRIC:-flops}"
  WARMUP="${WARMUP:-10}"

  for LAMBDA in ${LAMBDAS}; do
    SAFE_LAMBDA="${LAMBDA//./p}"
    SAFE_LAMBDA="${SAFE_LAMBDA//-/m}"
    SAVE_NAME="cg-${METRIC}-lambda${SAFE_LAMBDA}-seed${SEED}"
    echo "==> CG-DARTS: lambda=${LAMBDA}"
    "${PYTHON_BIN}" train_search_cg.py \
      --data "${DATA}" \
      --gpu "${GPU}" \
      --epochs "${EPOCHS}" \
      --seed "${SEED}" \
      --cost_metric "${METRIC}" \
      --cost_lambda "${LAMBDA}" \
      --cost_warmup_epochs "${WARMUP}" \
      --cost_normalize edge \
      --save "${SAVE_NAME}" \
      "${DOWNLOAD_ARG[@]}"
  done
fi

echo "Search complete. Run: ${PYTHON_BIN} ${ROOT_DIR}/scripts/cg_darts_report.py"
