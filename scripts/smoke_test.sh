#!/usr/bin/env bash
# end to end check on a tiny subset, runs on cpu. use PYTHON=path/to/python to pick the interpreter
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-python}
CONFIG=configs/smoke.yaml

# start clean so the runs are not skipped
rm -rf results/smoke

# the real pipeline script: data, learning rate grids (one candidate each here), select_best, beta runs,
# test scoring, baselines and the summary with the verdict
RM_LRS=1e-4 DPO_LRS=5e-5 BETAS="0.05 0.1" bash scripts/run_all.sh smoke

# validation scores are needed for check b, the stats command is checked on the test split
RM=$(ls -d results/smoke/rm_lr*)
DPO=$(ls -d results/smoke/dpo_beta0.1_*)
$PYTHON src/score.py --config $CONFIG --run_dir "$RM" --splits val
$PYTHON src/score.py --config $CONFIG --run_dir "$DPO" --splits val
$PYTHON src/stats.py --split test --a "$RM" --b "$DPO" --extra results/smoke/baseline_longer results/smoke/baseline_random

# correctness checks: zero reward at initialization, our accuracy against the trainer's, our scorer against the trainer's forward pass, same data order
$PYTHON -m pytest -q -s tests
