# CG-DARTS: Cost-Regularized DARTS

This repository is based on the official DARTS implementation:
https://github.com/quark0/darts

We extend Vanilla DARTS with differentiable MAC cost regularization for CIFAR-10 CNN architecture search.

Architecture loss:
L_arch = L_val + lambda * E[MAC(alpha)]

Main added files:
- cnn/cost_utils.py
- cnn/architect_cg.py
- cnn/train_search_cg.py
- run.sh
- run_cg_sweep.sh
- scripts/cg_darts_report.py
