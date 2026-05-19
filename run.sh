#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=1
PYTHON_BIN="${PYTHON_BIN:-/home/members/hyojeong/anaconda3/bin/conda run -n hj python}"
TASK="${1:-}"

case "${TASK}" in
  cnn-search-select)
    shift
    ${PYTHON_BIN} scripts/reproduce_protocol.py cnn "$@"
    ;;
  rnn-search-select)
    shift
    ${PYTHON_BIN} scripts/reproduce_protocol.py rnn "$@"
    ;;
  cnn-search)
    shift
    cd cnn
    ${PYTHON_BIN} train_search.py "$@"
    ;;
  cnn-cg-search)
    shift
    cd cnn
    ${PYTHON_BIN} train_search_cg.py "$@"
    ;;
  cnn-cg-sweep)
    shift
    ./run_cg_sweep.sh "$@"
    ;;
  cnn-train)
    shift
    cd cnn
    ${PYTHON_BIN} train.py --auxiliary --cutout "$@"
    ;;
  rnn-search)
    shift
    cd rnn
    ${PYTHON_BIN} train_search.py "$@"
    ;;
  rnn-train)
    shift
    cd rnn
    ${PYTHON_BIN} train.py "$@"
    ;;
  *)
    cat <<'EOF'
Usage:
  ./run.sh cnn-search-select --data ./data --download --gpu 0 --unrolled
  ./run.sh cnn-cg-search --data ./data --download --gpu 0 --cost_lambda 1e-2
  LAMBDAS="0 1e-3 5e-3 1e-2" ./run.sh cnn-cg-sweep --batch_size 64
  ./run.sh rnn-search-select --data ./data/penn --gpu 0 --unrolled
  ./run.sh cnn-search --data ./data --download --gpu 0 --unrolled
  ./run.sh cnn-train --data ./data --download --gpu 0
  ./run.sh rnn-search --data ./data/penn --gpu 0 --unrolled
  ./run.sh rnn-train --data ./data/penn --gpu 0

The search-select modes implement the DARTS paper protocol:
four search runs, short retraining for selection, and a JSON summary with the
best validation-selected genotype.
EOF
    exit 1
    ;;
esac
