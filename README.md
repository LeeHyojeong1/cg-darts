# CG-DARTS

Official DARTS repo 기반으로, CNN search에 MAC cost regularization을 추가한 코드입니다.

L_arch = L_val + lambda * E[MAC(alpha)]

## Added files

- `cnn/cost_utils.py`
- `cnn/architect_cg.py`
- `cnn/train_search_cg.py`
- `run_cg_sweep.sh`
- `scripts/cg_darts_report.py`

## Run

환경에 맞게 Python 설정:

    export PYTHON_BIN=python

Vanilla DARTS search:

    ./run.sh cnn-search --data ./data --download --gpu 0 --epochs 50 --save vanilla-50ep

CG-DARTS single run:

    ./run.sh cnn-cg-search --data ./data --download --gpu 0 \
      --epochs 50 --cost_lambda 1e-2 --cost_warmup_epochs 10

Lambda sweep:

    LAMBDAS="1e-3 5e-3 1e-2 5e-2 1e-1" GPU=0 EPOCHS=50 SEED=2 \
      ./run.sh cnn-cg-sweep --batch_size 64

Retrain searched architecture:

    GENO="$(cat cnn/SEARCH_FOLDER/genotype.txt)"

    ./run.sh cnn-train --data ./data --download --gpu 0 \
      --epochs 100 --seed 2 --genotype "$GENO"

Generate result summary/plots:

    python scripts/cg_darts_report.py
    python scripts/plot_from_summary.py   # CSV-only plots

Results are saved under `reports/cg_darts/`.

Full reproduction (2 GPUs, search + retrain + report):

    ./scripts/run_parallel_pipeline.sh
    tail -f logs/pipeline.log

See `PROJECT_STATUS.md` for experiment progress.
