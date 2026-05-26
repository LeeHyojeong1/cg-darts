#!/usr/bin/env bash
# Run vanilla search on GPU 0 and CG lambda sweep on GPU 1, then batch retrain.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/members/ryeowook/miniconda3/bin/python}"
DATA="${DATA:-${ROOT_DIR}/data}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-2}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-100}"

echo "[$(date)] Starting parallel search (GPU0=vanilla, GPU1=CG sweep)"

CUDA_VISIBLE_DEVICES=0 EPOCHS="${EPOCHS}" SEED="${SEED}" GPU=0 DOWNLOAD=0 RUN_CG_SWEEP=0 \
  "${ROOT_DIR}/scripts/run_full_search.sh" \
  > "${LOG_DIR}/vanilla_search.log" 2>&1 &
PID_VANILLA=$!

CUDA_VISIBLE_DEVICES=1 EPOCHS="${EPOCHS}" SEED="${SEED}" GPU=0 DOWNLOAD=0 \
  LAMBDAS="${LAMBDAS:-1e-3 5e-3 1e-2 5e-2 1e-1}" \
  "${ROOT_DIR}/run_cg_sweep.sh" \
  > "${LOG_DIR}/cg_sweep.log" 2>&1 &
PID_CG=$!

wait "${PID_VANILLA}"
echo "[$(date)] Vanilla search finished"
wait "${PID_CG}"
echo "[$(date)] CG sweep finished"

echo "[$(date)] Starting batch retrain on GPU 0"
CUDA_VISIBLE_DEVICES=0 EPOCHS="${RETRAIN_EPOCHS}" SEED="${SEED}" GPU=0 \
  "${ROOT_DIR}/scripts/run_batch_retrain.sh" \
  > "${LOG_DIR}/batch_retrain.log" 2>&1

echo "[$(date)] Generating report"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/cg_darts_report.py" \
  > "${LOG_DIR}/report.log" 2>&1

echo "[$(date)] Pipeline complete. See ${LOG_DIR}/"
