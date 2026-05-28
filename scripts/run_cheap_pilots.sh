#!/usr/bin/env bash
# Sequentially run the "Cheap" experiment pilots on a single GPU (GPU 0).
#
# Stage A: softmax-annealing pilot (tau 5 -> 0.1 linear) at lambda=1e-2 FLOPs.
# Stage B: standard CG-DARTS at lambda=5e-2 FLOPs (basis for the
#          cost-aware-discretization comparison via post-hoc rederive).
# Stage C: rederive Stage-B genotypes in argmax / cost_sub / cost_div modes.
# Stage D: retrain Stage A's genotype + the three Stage-B-derived genotypes
#          (4 retrains, 100 ep each).
# Stage E: skip-share diagnostic + cross-device eval + Pareto plotting over
#          everything that produced a genotype.
#
# This script is meant to be launched with `bash` in background. Progress
# goes to logs/cheap_pilots.log. Each stage writes its own per-run log too.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p logs reports

PYTHON_BIN="${PYTHON_BIN:-/home/members/ryeowook/miniconda3/bin/python}"
DATA="${DATA:-${ROOT_DIR}/data}"
GPU="${GPU:-0}"
SEED="${SEED:-2}"
EPOCHS="${EPOCHS:-50}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"

STAGE="${STAGE:-all}"  # all | A | B | C | D | E

LOG="${ROOT_DIR}/logs/cheap_pilots.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

newest_dir() {
  # arg: pattern (relative to cnn/)
  ls -dt "${ROOT_DIR}/cnn"/$1 2>/dev/null | head -1
}

stage_A_anneal() {
  say "stage A: search with tau anneal 5->0.1 linear, lambda=1e-2, FLOPs"
  cd "${ROOT_DIR}/cnn"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train_search_cg.py \
    --data "${DATA}" --download \
    --gpu 0 --epochs "${EPOCHS}" --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --cost_metric flops --cost_lambda 1e-2 --cost_warmup_epochs 10 \
    --cost_normalize edge \
    --tau_start 5.0 --tau_end 0.1 --tau_anneal linear \
    --save "anneal-tau5to01-lambda1em2-seed${SEED}" \
    >>"${ROOT_DIR}/logs/stage_A_anneal.log" 2>&1
  cd "${ROOT_DIR}"
}

stage_B_lambda5e2() {
  say "stage B: standard CG-DARTS search at lambda=5e-2 FLOPs"
  cd "${ROOT_DIR}/cnn"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train_search_cg.py \
    --data "${DATA}" --download \
    --gpu 0 --epochs "${EPOCHS}" --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --cost_metric flops --cost_lambda 5e-2 --cost_warmup_epochs 10 \
    --cost_normalize edge \
    --save "lambda5em2-seed${SEED}-disc-baseline" \
    >>"${ROOT_DIR}/logs/stage_B_lambda5e2.log" 2>&1
  cd "${ROOT_DIR}"
}

stage_C_rederive() {
  local sd; sd="$(newest_dir 'search-cg-lambda5em2-seed*-disc-baseline-*')"
  if [[ -z "${sd}" ]]; then
    say "stage C: ERROR — no stage-B search dir found"
    return 1
  fi
  say "stage C: rederive 3 modes from ${sd}"
  for mode in argmax cost_sub cost_div; do
    "${PYTHON_BIN}" scripts/rederive_genotype.py \
      --search-dir "${sd}" --mode "${mode}" --cost-metric flops \
      --cost-weight 0.5 \
      >>"${ROOT_DIR}/logs/stage_C_rederive.log" 2>&1
  done
  say "stage C: wrote genotype.<mode>.txt in ${sd}"
}

run_retrain() {
  local geno_path="$1"; local save_tag="$2"
  if [[ ! -f "${geno_path}" ]]; then
    say "retrain skip: ${geno_path} missing"
    return 1
  fi
  local geno; geno="$(cat "${geno_path}")"
  cd "${ROOT_DIR}/cnn"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train.py \
    --data "${DATA}" --download --gpu 0 \
    --epochs "${RETRAIN_EPOCHS}" --seed "${SEED}" \
    --batch_size 96 --num_workers "${NUM_WORKERS}" \
    --auxiliary --cutout \
    --save "${save_tag}" --genotype "${geno}" \
    >>"${ROOT_DIR}/logs/retrain_${save_tag}.log" 2>&1
  cd "${ROOT_DIR}"
}

stage_D_retrain() {
  say "stage D: retraining annealing genotype + 3 rederived genotypes"
  local anneal_sd; anneal_sd="$(newest_dir 'search-cg-anneal-tau5to01-lambda1em2-seed*')"
  local lambda_sd; lambda_sd="$(newest_dir 'search-cg-lambda5em2-seed*-disc-baseline-*')"
  if [[ -n "${anneal_sd}" ]]; then
    run_retrain "${anneal_sd}/genotype.txt" "anneal-tau5to01-lambda1em2-seed${SEED}" \
      || say "  anneal retrain failed"
  fi
  if [[ -n "${lambda_sd}" ]]; then
    for mode in argmax cost_sub cost_div; do
      run_retrain "${lambda_sd}/genotype.${mode}.txt" "disc-${mode}-lambda5em2-seed${SEED}" \
        || say "  ${mode} retrain failed"
    done
  fi
  say "stage D: retrains complete"
}

stage_E_analysis() {
  say "stage E: skip-share + cross-device + Pareto analyses"
  mkdir -p reports/skip_share reports/cross_device reports/pareto
  "${PYTHON_BIN}" scripts/skip_share_diagnostic.py \
    --glob 'cnn/search-cg-anneal-*' \
    --glob 'cnn/search-cg-lambda5em2-*' \
    --out reports/skip_share/cheap_pilots.csv \
    >>"${ROOT_DIR}/logs/stage_E_skip.log" 2>&1 || true

  # cross-device eval on every genotype.* in the two search dirs
  local args=()
  for sd in $(ls -d cnn/search-cg-anneal-* cnn/search-cg-lambda5em2-* 2>/dev/null); do
    if [[ -f "${sd}/genotype.txt" ]]; then
      args+=(--search-dir "${sd}")
    fi
  done
  if [[ ${#args[@]} -gt 0 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/cross_device_eval.py \
      "${args[@]}" --iters 30 --out reports/cross_device/cheap_pilots.csv \
      >>"${ROOT_DIR}/logs/stage_E_xdev.log" 2>&1 || true
  fi
  say "stage E: analyses complete"
}

case "${STAGE}" in
  all)
    stage_A_anneal
    stage_B_lambda5e2
    stage_C_rederive
    stage_D_retrain
    stage_E_analysis
    ;;
  A) stage_A_anneal ;;
  B) stage_B_lambda5e2 ;;
  C) stage_C_rederive ;;
  D) stage_D_retrain ;;
  E) stage_E_analysis ;;
  *) echo "unknown STAGE=${STAGE}"; exit 1 ;;
esac

say "cheap pilots done"
