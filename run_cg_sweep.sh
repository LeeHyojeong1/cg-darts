#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/members/ryeowook/miniconda3/bin/python}"
DATA="${DATA:-${ROOT_DIR}/data}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-0}"
METRIC="${METRIC:-flops}"
NORMALIZE="${NORMALIZE:-edge}"
WARMUP="${WARMUP:-10}"
LAMBDAS="${LAMBDAS:-0 1e-3 5e-3 1e-2 5e-2}"
DOWNLOAD="${DOWNLOAD:-1}"
UNROLLED="${UNROLLED:-0}"

DOWNLOAD_ARG=()
if [[ "${DOWNLOAD}" == "1" ]]; then
  DOWNLOAD_ARG=(--download)
fi

UNROLLED_ARG=()
if [[ "${UNROLLED}" == "1" ]]; then
  UNROLLED_ARG=(--unrolled)
fi

cd "${ROOT_DIR}/cnn"

for LAMBDA in ${LAMBDAS}; do
  SAFE_LAMBDA="${LAMBDA//./p}"
  SAFE_LAMBDA="${SAFE_LAMBDA//-/m}"
  SAFE_LAMBDA="${SAFE_LAMBDA//+/p}"
  SAVE_NAME="cg-${METRIC}-lambda${SAFE_LAMBDA}-seed${SEED}"

  echo "==> CG-DARTS search: metric=${METRIC}, lambda=${LAMBDA}, seed=${SEED}, epochs=${EPOCHS}"
  ${PYTHON_BIN} train_search_cg.py \
    --data "${DATA}" \
    --gpu "${GPU}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --cost_metric "${METRIC}" \
    --cost_lambda "${LAMBDA}" \
    --cost_warmup_epochs "${WARMUP}" \
    --cost_normalize "${NORMALIZE}" \
    --save "${SAVE_NAME}" \
    "${DOWNLOAD_ARG[@]}" \
    "${UNROLLED_ARG[@]}" \
    "$@"
done
