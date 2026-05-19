# Local run guide

This checkout is based on the official DARTS implementation and has been patched
to run on modern Python/PyTorch. The original CNN DARTS baseline is under
`cnn/`.

## Environment

Known working local environment:

```bash
/home/members/hyojeong/anaconda3/bin/conda run -n hj python
```

The code also parses under `toma_env`. Use `--gpu -1` for CPU smoke tests, or
`--gpu 0` when CUDA is visible.

The CNN search defaults follow the ICLR 2019 paper appendix: 8 cells, 16 initial
channels, 50 epochs, batch size 64, SGD weight learning rate 0.025 cosine-annealed
to 0, Adam architecture learning rate 3e-4, zero-initialized architecture logits,
and affine-free batch normalization with batch-specific statistics inside the
search network.

## Smoke tests

These do not require CIFAR-10 and only verify that the DARTS loops execute.

```bash
cd /home/members/hyojeong/darts/cnn
/home/members/hyojeong/anaconda3/bin/conda run -n hj python train_search.py \
  --epochs 1 --layers 2 --init_channels 2 --batch_size 2 \
  --debug_fake_data --debug_num_samples 8 --num_workers 0 --gpu -1
```

Second-order / unrolled search:

```bash
cd /home/members/hyojeong/darts/cnn
/home/members/hyojeong/anaconda3/bin/conda run -n hj python train_search.py \
  --epochs 1 --layers 2 --init_channels 2 --batch_size 2 \
  --debug_fake_data --debug_num_samples 8 --num_workers 0 --gpu -1 --unrolled
```

Evaluation training loop:

```bash
cd /home/members/hyojeong/darts/cnn
/home/members/hyojeong/anaconda3/bin/conda run -n hj python train.py \
  --epochs 1 --layers 2 --init_channels 2 --batch_size 2 \
  --debug_fake_data --debug_num_samples 8 --num_workers 0 --gpu -1
```

## CIFAR-10 baseline

Architecture search:

```bash
cd /home/members/hyojeong/darts/cnn
/home/members/hyojeong/anaconda3/bin/conda run -n hj python train_search.py \
  --data ../data --download --gpu 0
```

Second-order search:

```bash
cd /home/members/hyojeong/darts/cnn
/home/members/hyojeong/anaconda3/bin/conda run -n hj python train_search.py \
  --data ../data --download --gpu 0 --unrolled
```

Train the fixed DARTS genotype from scratch:

```bash
cd /home/members/hyojeong/darts/cnn
/home/members/hyojeong/anaconda3/bin/conda run -n hj python train.py \
  --data ../data --download --gpu 0 --auxiliary --cutout
```

## Paper selection protocol

The paper runs search multiple times, then retrains each discovered architecture
for a short budget and chooses the best validation result before final training.
Use `run.sh` for that protocol:

```bash
cd /home/members/hyojeong/darts
./run.sh cnn-search-select --data ./data --download --gpu 0 --unrolled
./run.sh rnn-search-select --data ./data/penn --gpu 0 --unrolled
```

Defaults are four seeds, 100 short-retrain epochs for CIFAR, and 300 short-retrain
epochs for PTB. The selected genotype is written to `paper_selection_summary.json`.

## PTB baseline

Recurrent search:

```bash
cd /home/members/hyojeong/darts/rnn
/home/members/hyojeong/anaconda3/bin/conda run -n hj python train_search.py \
  --data ../data/penn --gpu 0 --unrolled
```

Train the fixed DARTS recurrent genotype:

```bash
cd /home/members/hyojeong/darts/rnn
/home/members/hyojeong/anaconda3/bin/conda run -n hj python train.py \
  --data ../data/penn --gpu 0
```
