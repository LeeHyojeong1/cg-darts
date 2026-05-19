#!/usr/bin/env python
"""Run the DARTS paper search-selection protocol.

This script automates the paper's model-selection step:

1. Run architecture search several times with different seeds.
2. Retrain each discovered genotype for a short budget.
3. Select the genotype with the best validation metric.

It intentionally does not run the final 600-epoch CIFAR or full PTB evaluation;
after selection, use the printed genotype with the normal train scripts.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def newest_dir(parent, prefix, before):
    after = {p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)}
    created = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not created:
        raise RuntimeError("could not find output directory for prefix {}".format(prefix))
    return created[-1]


def read_genotype(exp_dir):
    path = exp_dir / "genotype.txt"
    if not path.exists():
        raise RuntimeError("missing genotype file: {}".format(path))
    return path.read_text().strip()


def last_metric(log_path, pattern):
    regex = re.compile(pattern)
    value = None
    for line in log_path.read_text(errors="replace").splitlines():
        match = regex.search(line)
        if match:
            value = float(match.group(1))
    if value is None:
        raise RuntimeError("metric not found in {}".format(log_path))
    return value


def run_cnn(args):
    cwd = ROOT / "cnn"
    results = []
    for seed in args.seeds:
        before = {p for p in cwd.iterdir() if p.is_dir()}
        search_cmd = [
            sys.executable, "train_search.py",
            "--data", args.data,
            "--seed", str(seed),
            "--gpu", str(args.gpu),
            "--save", "paper-search-seed{}".format(seed),
        ]
        if args.download:
            search_cmd.append("--download")
        if args.unrolled:
            search_cmd.append("--unrolled")
        run(search_cmd, cwd)
        search_dir = newest_dir(cwd, "search-paper-search-seed{}".format(seed), before)
        genotype = read_genotype(search_dir)

        before = {p for p in cwd.iterdir() if p.is_dir()}
        select_cmd = [
            sys.executable, "train.py",
            "--data", args.data,
            "--epochs", str(args.short_epochs),
            "--seed", str(seed),
            "--gpu", str(args.gpu),
            "--save", "paper-select-seed{}".format(seed),
            "--genotype", genotype,
            "--auxiliary",
            "--cutout",
        ]
        if args.download:
            select_cmd.append("--download")
        run(select_cmd, cwd)
        select_dir = newest_dir(cwd, "eval-paper-select-seed{}".format(seed), before)
        valid_acc = last_metric(select_dir / "log.txt", r"valid_acc ([0-9.]+)")
        results.append({
            "seed": seed,
            "search_dir": str(search_dir),
            "selection_dir": str(select_dir),
            "genotype": genotype,
            "valid_acc": valid_acc,
        })

    best = max(results, key=lambda item: item["valid_acc"])
    write_summary(args.output, "cnn", results, best)


def run_rnn(args):
    cwd = ROOT / "rnn"
    results = []
    for seed in args.seeds:
        before = {p for p in cwd.iterdir() if p.is_dir()}
        search_cmd = [
            sys.executable, "train_search.py",
            "--data", args.data,
            "--seed", str(seed),
            "--gpu", str(args.gpu),
            "--save", "paper-search-seed{}".format(seed),
        ]
        if args.unrolled:
            search_cmd.append("--unrolled")
        run(search_cmd, cwd)
        search_dir = newest_dir(cwd, "search-paper-search-seed{}".format(seed), before)
        genotype = read_genotype(search_dir)

        before = {p for p in cwd.iterdir() if p.is_dir()}
        select_cmd = [
            sys.executable, "train.py",
            "--data", args.data,
            "--epochs", str(args.short_epochs),
            "--seed", str(seed),
            "--gpu", str(args.gpu),
            "--save", "paper-select-seed{}".format(seed),
            "--genotype", genotype,
        ]
        run(select_cmd, cwd)
        select_dir = newest_dir(cwd, "eval-paper-select-seed{}".format(seed), before)
        valid_ppl = last_metric(select_dir / "log.txt", r"valid ppl\s+([0-9.]+)")
        results.append({
            "seed": seed,
            "search_dir": str(search_dir),
            "selection_dir": str(select_dir),
            "genotype": genotype,
            "valid_ppl": valid_ppl,
        })

    best = min(results, key=lambda item: item["valid_ppl"])
    write_summary(args.output, "rnn", results, best)


def write_summary(output, task, results, best):
    payload = {"task": task, "results": results, "best": best}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print("Wrote", output)
    print("Best genotype:")
    print(best["genotype"])


def parse_args():
    parser = argparse.ArgumentParser(description="DARTS paper search-selection protocol")
    parser.add_argument("task", choices=["cnn", "rnn"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--short_epochs", type=int, default=None)
    parser.add_argument("--output", default=str(ROOT / "paper_selection_summary.json"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--unrolled", action="store_true")
    args = parser.parse_args()
    if args.short_epochs is None:
        args.short_epochs = 100 if args.task == "cnn" else 300
    return args


def main():
    args = parse_args()
    if args.task == "cnn":
        run_cnn(args)
    else:
        run_rnn(args)


if __name__ == "__main__":
    main()
