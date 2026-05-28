#!/usr/bin/env bash
# Wait for the LUT-driven search to finish, then retrain the discovered
# genotype on GPU 1, then run cross-device eval to compare measured vs
# FLOPs cells on L40S latency. Designed to be backgrounded.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="/home/members/ryeowook/miniconda3/bin/python"
LOG="${ROOT_DIR}/logs/lut_experiment.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

newest_search() {
  ls -dt cnn/search-cg-lut-lambda1em2-seed2-* 2>/dev/null | head -1
}

# 1) Wait for search to finish — i.e. when no train_search_cg.py belonging
#    to us is still running for this save tag, and a genotype.txt is present.
say "stage 1: waiting for LUT search to complete"
while true; do
  sd="$(newest_search)"
  if [[ -n "${sd}" ]] && [[ -f "${sd}/genotype.txt" ]]; then
    last_epoch=$(awk -F, 'NR>1 {e=$1} END {print e}' "${sd}/cost_log.csv" 2>/dev/null || echo "")
    if ! pgrep -u jongjin -f "train_search_cg.py.*lut-lambda1em2-seed2" >/dev/null; then
      say "search dir ${sd} ready (last epoch=${last_epoch:-?}); proceeding"
      break
    fi
    say "  still searching... last epoch=${last_epoch:-?}"
  fi
  sleep 120
done

sd="$(newest_search)"
geno="$(cat "${sd}/genotype.txt")"
say "stage 2: retrain on GPU 1 with genotype:"
say "  ${geno}"

# 2) Retrain on GPU 1 (free), full 100-epoch protocol, same as master's runs.
cd "${ROOT_DIR}/cnn"
CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" train.py \
  --data "${ROOT_DIR}/data" --gpu 0 \
  --epochs 100 --seed 2 --batch_size 96 --num_workers 4 \
  --auxiliary --cutout \
  --save "eval-lut-lambda1em2-seed2" \
  --genotype "${geno}" \
  >>"${ROOT_DIR}/logs/lut_retrain.log" 2>&1
cd "${ROOT_DIR}"
say "stage 2: retrain finished"

# 3) Cross-device eval comparing this LUT cell vs the prior FLOPs cell.
#    Master's FLOPs lambda=1e-2 search dir is gone from this checkout, so
#    we only eval the LUT cell here; report.py will pull master's MACs from
#    summary.csv for the comparison.
say "stage 3: cross-device eval on LUT cell"
mkdir -p reports/cross_device
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/cross_device_eval.py \
  --search-dir "${sd}" --iters 50 \
  --out reports/cross_device/lut_lambda1em2.csv \
  >>"${ROOT_DIR}/logs/lut_xdev.log" 2>&1 || true

say "experiment complete"
