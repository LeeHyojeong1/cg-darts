#!/usr/bin/env bash
# Fast params-cost CG-DARTS pipeline:
#   - search: 30 epochs, lambda in {1e-2, 5e-2}, 1 job per GPU
#   - retrain: 50 epochs, same lambdas only, 2 GPUs in parallel
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/members/ryeowook/miniconda3/bin/python}"
DATA="${DATA:-${ROOT_DIR}/data}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-2}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-50}"
METRIC="${METRIC:-params}"
WARMUP="${WARMUP:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LAMBDAS=(${LAMBDAS:-1e-2 5e-2})
RETRAIN_LAMBDAS=(${RETRAIN_LAMBDAS:-1e-2 5e-2})
SKIP_SEARCH="${SKIP_SEARCH:-0}"
SKIP_RETRAIN="${SKIP_RETRAIN:-0}"
REPORT_ONLY="${REPORT_ONLY:-0}"

if [[ "${REPORT_ONLY}" == "1" ]]; then
  SKIP_SEARCH=1
  SKIP_RETRAIN=1
fi

lambda_token() {
  local lmbda="$1"
  local token="${lmbda//./p}"
  token="${token//-/m}"
  echo "lambda${token}"
}

newest_search_dir() {
  local token
  token="$(lambda_token "$1")"
  ls -dt "${ROOT_DIR}/cnn"/search-cg-cg-"${METRIC}"-"${token}"* 2>/dev/null | head -1
}

run_search() {
  local gpu="$1"
  local lmbda="$2"
  local token
  token="$(lambda_token "${lmbda}")"
  cd "${ROOT_DIR}/cnn"
  echo "[$(date)] GPU${gpu} search metric=${METRIC} lambda=${lmbda} epochs=${EPOCHS}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" train_search_cg.py \
    --data "${DATA}" \
    --gpu 0 \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --cost_metric "${METRIC}" \
    --cost_lambda "${lmbda}" \
    --cost_warmup_epochs "${WARMUP}" \
    --cost_normalize edge \
    --save "cg-${METRIC}-${token}-seed${SEED}"
}

run_retrain() {
  local gpu="$1"
  local lmbda="$2"
  local search_dir
  search_dir="$(newest_search_dir "${lmbda}")"
  if [[ -z "${search_dir}" || ! -f "${search_dir}/genotype.txt" ]]; then
    echo "[$(date)] ERROR: missing search result for lambda=${lmbda}" >&2
    return 1
  fi
  local label
  label="$(basename "${search_dir}")"
  label="${label#search-}"
  local geno
  geno="$(cat "${search_dir}/genotype.txt")"
  cd "${ROOT_DIR}/cnn"
  echo "[$(date)] GPU${gpu} retrain lambda=${lmbda} from ${search_dir} epochs=${RETRAIN_EPOCHS}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" train.py \
    --data "${DATA}" \
    --gpu 0 \
    --epochs "${RETRAIN_EPOCHS}" \
    --seed "${SEED}" \
    --batch_size 96 \
    --num_workers "${NUM_WORKERS}" \
    --auxiliary \
    --cutout \
    --save "eval-${label}" \
    --genotype "${geno}"
}

echo "[$(date)] Fast params pipeline: search_epochs=${EPOCHS} retrain_epochs=${RETRAIN_EPOCHS} lambdas=${LAMBDAS[*]}"

if [[ "${SKIP_SEARCH}" != "1" ]]; then
if [[ ${#LAMBDAS[@]} -ge 2 ]]; then
  run_search 0 "${LAMBDAS[0]}" > "${LOG_DIR}/params_search_gpu0.log" 2>&1 &
  PID_SEARCH0=$!
  run_search 1 "${LAMBDAS[1]}" > "${LOG_DIR}/params_search_gpu1.log" 2>&1 &
  PID_SEARCH1=$!
  wait "${PID_SEARCH0}"
  wait "${PID_SEARCH1}"
else
  run_search 0 "${LAMBDAS[0]}" > "${LOG_DIR}/params_search_gpu0.log" 2>&1
fi
echo "[$(date)] Search done"
else
  echo "[$(date)] Skipping search (SKIP_SEARCH=1)"
fi

if [[ "${SKIP_RETRAIN}" != "1" ]]; then
if [[ ${#RETRAIN_LAMBDAS[@]} -ge 2 ]]; then
  run_retrain 0 "${RETRAIN_LAMBDAS[0]}" > "${LOG_DIR}/params_retrain_gpu0.log" 2>&1 &
  PID_RT0=$!
  run_retrain 1 "${RETRAIN_LAMBDAS[1]}" > "${LOG_DIR}/params_retrain_gpu1.log" 2>&1 &
  PID_RT1=$!
  wait "${PID_RT0}"
  wait "${PID_RT1}"
else
  run_retrain 0 "${RETRAIN_LAMBDAS[0]}" > "${LOG_DIR}/params_retrain_gpu0.log" 2>&1
fi
echo "[$(date)] Retrain done"
else
  echo "[$(date)] Skipping retrain (SKIP_RETRAIN=1)"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/cg_darts_report.py" \
  --cost-metric params \
  --lambdas 1e-2 5e-2 \
  --skip-vanilla \
  > "${LOG_DIR}/params_report.log" 2>&1

echo "[$(date)] Done. Report: ${ROOT_DIR}/reports/cg_darts_params/"
