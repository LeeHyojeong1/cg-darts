#!/usr/bin/env bash
# Multi-seed CG-DARTS sweep.
#
# DARTS is famously seed-sensitive (Zela et al. 2020). This script repeats a
# (metric, lambda) configuration across SEEDS so the headline mean+/-std can
# be reported rather than a single-seed point estimate.
#
# Searches are dispatched in round-robin across GPUs={0,1}. The same script
# also kicks off the retrains once the searches finish (set SKIP_RETRAIN=1
# to defer retraining to a separate pass).
#
# Env vars:
#   PYTHON_BIN     - python interpreter (default: ryeowook conda)
#   DATA           - CIFAR-10 directory
#   EPOCHS         - search epochs (default 50)
#   RETRAIN_EPOCHS - retrain epochs (default 100)
#   METRIC         - flops | params | mem | device (default flops)
#   LAMBDAS        - space-separated list (default "1e-2 5e-2")
#   SEEDS          - space-separated list (default "0 1 2 3")
#   UNROLLED       - 1 to enable second-order DARTS (default 0)
#   GPUS           - space-separated list of gpu IDs (default "0 1")
#   SKIP_SEARCH    - 1 to skip search phase
#   SKIP_RETRAIN   - 1 to skip retrain phase
#   WARMUP         - cost_lambda warmup epochs (default 10)
#   BATCH_SIZE     - search batch size (default 64)
#   TAU_START / TAU_END / TAU_ANNEAL - softmax temperature schedule (defaults: 1.0 / 1.0 / none)
#   DISCRETIZE_MODE / DISCRETIZE_COST_WEIGHT - derivation strategy
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/members/ryeowook/miniconda3/bin/python}"
DATA="${DATA:-${ROOT_DIR}/data}"
EPOCHS="${EPOCHS:-50}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-100}"
METRIC="${METRIC:-flops}"
LAMBDAS=(${LAMBDAS:-1e-2 5e-2})
SEEDS=(${SEEDS:-0 1 2 3})
UNROLLED="${UNROLLED:-0}"
GPUS=(${GPUS:-0 1})
SKIP_SEARCH="${SKIP_SEARCH:-0}"
SKIP_RETRAIN="${SKIP_RETRAIN:-0}"
WARMUP="${WARMUP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TAU_START="${TAU_START:-1.0}"
TAU_END="${TAU_END:-1.0}"
TAU_ANNEAL="${TAU_ANNEAL:-none}"
DISCRETIZE_MODE="${DISCRETIZE_MODE:-argmax}"
DISCRETIZE_COST_WEIGHT="${DISCRETIZE_COST_WEIGHT:-1.0}"
DEVICE="${DEVICE:-}"

NUM_GPUS=${#GPUS[@]}

lambda_token() {
  local lmbda="$1"
  local token="${lmbda//./p}"
  token="${token//-/m}"
  token="${token//+/p}"
  echo "lambda${token}"
}

run_search() {
  local gpu="$1"; local lmbda="$2"; local seed="$3"
  local token; token="$(lambda_token "${lmbda}")"
  local extra=()
  if [[ "${UNROLLED}" == "1" ]]; then extra+=(--unrolled); fi
  if [[ -n "${DEVICE}" ]]; then extra+=(--device "${DEVICE}"); fi
  cd "${ROOT_DIR}/cnn"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" train_search_cg.py \
    --data "${DATA}" \
    --gpu 0 \
    --epochs "${EPOCHS}" \
    --seed "${seed}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --cost_metric "${METRIC}" \
    --cost_lambda "${lmbda}" \
    --cost_warmup_epochs "${WARMUP}" \
    --cost_normalize edge \
    --tau_start "${TAU_START}" --tau_end "${TAU_END}" --tau_anneal "${TAU_ANNEAL}" \
    --discretize_mode "${DISCRETIZE_MODE}" \
    --discretize_cost_weight "${DISCRETIZE_COST_WEIGHT}" \
    --save "ms-${METRIC}-${token}-seed${seed}" \
    "${extra[@]}"
}

newest_search_dir() {
  local lmbda="$1"; local seed="$2"
  local token; token="$(lambda_token "${lmbda}")"
  ls -dt "${ROOT_DIR}/cnn"/search-cg-ms-"${METRIC}"-"${token}"-seed"${seed}"-* 2>/dev/null | head -1
}

run_retrain() {
  local gpu="$1"; local lmbda="$2"; local seed="$3"
  local search_dir; search_dir="$(newest_search_dir "${lmbda}" "${seed}")"
  if [[ -z "${search_dir}" || ! -f "${search_dir}/genotype.txt" ]]; then
    echo "[$(date)] ERROR: missing search for lambda=${lmbda} seed=${seed}" >&2
    return 1
  fi
  local label; label="$(basename "${search_dir}")"; label="${label#search-}"
  local geno;  geno="$(cat "${search_dir}/genotype.txt")"
  cd "${ROOT_DIR}/cnn"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" train.py \
    --data "${DATA}" --gpu 0 --epochs "${RETRAIN_EPOCHS}" --seed "${seed}" \
    --batch_size 96 --num_workers "${NUM_WORKERS}" --auxiliary --cutout \
    --save "eval-${label}" --genotype "${geno}"
}

dispatch() {
  local fn="$1"; shift
  local idx=0
  for lmbda in "${LAMBDAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local gpu="${GPUS[$(( idx % NUM_GPUS ))]}"
      local tag="${fn}_${METRIC}_$(lambda_token "${lmbda}")_seed${seed}"
      local log="${LOG_DIR}/${tag}.log"
      echo "[$(date)] dispatch ${fn} gpu=${gpu} lambda=${lmbda} seed=${seed} -> ${log}"
      "${fn}" "${gpu}" "${lmbda}" "${seed}" > "${log}" 2>&1 &
      idx=$((idx + 1))
      if (( idx % NUM_GPUS == 0 )); then
        wait
      fi
    done
  done
  wait
}

if [[ "${SKIP_SEARCH}" != "1" ]]; then
  echo "[$(date)] multi-seed SEARCH: metric=${METRIC} lambdas=${LAMBDAS[*]} seeds=${SEEDS[*]} unrolled=${UNROLLED}"
  dispatch run_search
fi

if [[ "${SKIP_RETRAIN}" != "1" ]]; then
  echo "[$(date)] multi-seed RETRAIN: metric=${METRIC} lambdas=${LAMBDAS[*]} seeds=${SEEDS[*]}"
  dispatch run_retrain
fi

echo "[$(date)] multi-seed pipeline complete."
