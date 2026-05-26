#!/usr/bin/env bash
# Retrain every search-cg-* and search-first-order-* directory (100 epochs default).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/members/ryeowook/miniconda3/bin/python}"
DATA="${DATA:-${ROOT_DIR}/data}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-2}"

cd "${ROOT_DIR}/cnn"

for SEARCH_DIR in search-cg-* search-first-order-*; do
  [[ -d "${SEARCH_DIR}" ]] || continue
  GENO_FILE="${SEARCH_DIR}/genotype.txt"
  [[ -f "${GENO_FILE}" ]] || continue

  LABEL="${SEARCH_DIR#search-}"
  echo "==> Retrain from ${SEARCH_DIR} (${EPOCHS} epochs)"
  GENO="$(cat "${GENO_FILE}")"
  "${PYTHON_BIN}" train.py \
    --data "${DATA}" \
    --gpu "${GPU}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --auxiliary \
    --cutout \
    --save "eval-${LABEL}" \
    --genotype "${GENO}"
done

echo "Retrain complete."
