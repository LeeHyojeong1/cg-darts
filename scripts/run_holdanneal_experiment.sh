#!/usr/bin/env bash
# Hold-then-cosine annealing CG-DARTS experiment:
#   search at lambda=1e-2 FLOPs, tau held at 1.0 for 80% of epochs then
#   cosine-decay to 0.3.  Retrain on GPU 1 the moment search finishes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="/home/members/ryeowook/miniconda3/bin/python"
LOG="${ROOT_DIR}/logs/holdanneal_experiment.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

newest_search() {
  ls -dt cnn/search-cg-holdanneal-lambda1em2-seed2-* 2>/dev/null | head -1
}

say "stage 1: waiting for hold-anneal search to complete"
while true; do
  sd="$(newest_search)"
  if [[ -n "${sd}" ]] && [[ -f "${sd}/genotype.txt" ]]; then
    if ! pgrep -u jongjin -f "train_search_cg.py.*holdanneal-lambda1em2-seed2" >/dev/null; then
      last_epoch=$(awk -F, 'NR>1 {e=$1} END {print e}' "${sd}/cost_log.csv" 2>/dev/null || echo "")
      say "search done at ${sd} (last epoch=${last_epoch:-?})"
      break
    else
      last_epoch=$(awk -F, 'NR>1 {e=$1} END {print e}' "${sd}/cost_log.csv" 2>/dev/null || echo "")
      say "  still searching... last logged epoch=${last_epoch:-?}"
    fi
  fi
  sleep 180
done

sd="$(newest_search)"
geno="$(cat "${sd}/genotype.txt")"
say "stage 2: retrain on GPU 1"

cd "${ROOT_DIR}/cnn"
CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" train.py \
  --data "${ROOT_DIR}/data" --gpu 0 \
  --epochs 100 --seed 2 --batch_size 96 --num_workers 4 \
  --auxiliary --cutout \
  --save "eval-holdanneal-lambda1em2-seed2" \
  --genotype "${geno}" \
  >>"${ROOT_DIR}/logs/holdanneal_retrain.log" 2>&1
cd "${ROOT_DIR}"
say "stage 2: retrain finished"

say "experiment complete"
