#!/usr/bin/env bash
# Device-conditioned CG-DARTS: search per hardware profile (Edge / Middle / High-end).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/members/ryeowook/miniconda3/bin/python}"
DATA="${DATA:-${ROOT_DIR}/data}"
EPOCHS="${EPOCHS:-50}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-100}"
SEED="${SEED:-2}"
WARMUP="${WARMUP:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LAMBDAS=(${LAMBDAS:-1e-2})
DEVICES=(${DEVICES:-jetson_orin rtx_pro_6000 h100})
SKIP_SEARCH="${SKIP_SEARCH:-0}"
SKIP_RETRAIN="${SKIP_RETRAIN:-0}"

lambda_token() {
  local lmbda="$1"
  echo "lambda${lmbda//./p}"
}

newest_search_dir() {
  local device="$1"
  local lmbda="$2"
  local token dir
  token="$(lambda_token "${lmbda}")"
  for dir in $(ls -dt "${ROOT_DIR}/cnn"/search-cg-device-"${device}"-"${token}"* 2>/dev/null); do
    if [[ -f "${dir}/genotype.txt" && -f "${dir}/weights.pt" ]]; then
      echo "${dir}"
      return 0
    fi
  done
  ls -dt "${ROOT_DIR}/cnn"/search-cg-device-"${device}"-"${token}"* 2>/dev/null | head -1
}

run_search() {
  local gpu="$1"
  local device="$2"
  local lmbda="$3"
  local token
  token="$(lambda_token "${lmbda}")"
  cd "${ROOT_DIR}/cnn"
  echo "[$(date)] GPU${gpu} device=${device} lambda=${lmbda} epochs=${EPOCHS}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" train_search_cg.py \
    --data "${DATA}" \
    --gpu 0 \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --cost_metric device \
    --device "${device}" \
    --cost_lambda "${lmbda}" \
    --cost_warmup_epochs "${WARMUP}" \
    --cost_normalize edge \
    --save "device-${device}-${token}-seed${SEED}"
}

run_retrain() {
  local gpu="$1"
  local device="$2"
  local lmbda="$3"
  local search_dir
  search_dir="$(newest_search_dir "${device}" "${lmbda}")"
  if [[ -z "${search_dir}" || ! -f "${search_dir}/genotype.txt" ]]; then
    echo "missing search dir for device=${device} lambda=${lmbda}" >&2
    return 1
  fi
  local label geno
  label="$(basename "${search_dir}")"
  label="${label#search-}"
  geno="$(cat "${search_dir}/genotype.txt")"
  cd "${ROOT_DIR}/cnn"
  echo "[$(date)] GPU${gpu} retrain device=${device} lambda=${lmbda}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" train.py \
    --data "${DATA}" \
    --gpu 0 \
    --epochs "${RETRAIN_EPOCHS}" \
    --seed "${SEED}" \
    --batch_size 96 \
    --num_workers "${NUM_WORKERS}" \
    --cutout \
    --save "eval-${label}" \
    --genotype "${geno}"
}

echo "[$(date)] Device pipeline: devices=${DEVICES[*]} lambdas=${LAMBDAS[*]}"

if [[ "${SKIP_SEARCH}" != "1" ]]; then
  idx=0
  for device in "${DEVICES[@]}"; do
    for lmbda in "${LAMBDAS[@]}"; do
      gpu=$((idx % 2))
      log="${LOG_DIR}/device_search_${device}_lambda${lmbda//./p}.log"
      run_search "${gpu}" "${device}" "${lmbda}" > "${log}" 2>&1 &
      idx=$((idx + 1))
      if (( idx % 2 == 0 )); then
        wait
      fi
    done
  done
  wait
  echo "[$(date)] Search done"
fi

if [[ "${SKIP_RETRAIN}" != "1" ]]; then
  idx=0
  pids=()
  for device in "${DEVICES[@]}"; do
    for lmbda in "${LAMBDAS[@]}"; do
      gpu=$((idx % 2))
      log="${LOG_DIR}/device_retrain_${device}_lambda${lmbda//./p}.log"
      run_retrain "${gpu}" "${device}" "${lmbda}" > "${log}" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if (( idx % 2 == 0 )); then
        for pid in "${pids[@]}"; do
          wait "${pid}" || exit 1
        done
        pids=()
      fi
    done
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit 1
  done
  echo "[$(date)] Retrain done"
fi

echo "[$(date)] Device pipeline complete."
