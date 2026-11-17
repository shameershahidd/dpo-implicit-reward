#!/usr/bin/env bash
# end to end check on a tiny subset, runs on cpu. use PYTHON=path/to/python to pick the interpreter
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
CONFIG=configs/smoke.yaml

# start clean so the runs are not skipped
rm -rf results/smoke

$PYTHON src/data.py --config $CONFIG
$PYTHON src/train_rm.py --config $CONFIG --lr 1e-4
$PYTHON src/train_dpo.py --config $CONFIG --lr 5e-5 --beta 0.1
RM=$(ls -d results/smoke/rm_lr*)
DPO=$(ls -d results/smoke/dpo_beta*)

$PYTHON src/score.py --config $CONFIG --run_dir "$RM" --splits val test
$PYTHON src/score.py --config $CONFIG --run_dir "$DPO" --splits val test
$PYTHON src/score.py --config $CONFIG --baselines --splits val test
for split in val test; do
  $PYTHON src/stats.py --split $split --a "$RM" --b "$DPO" --extra results/smoke/baseline_longer results/smoke/baseline_random
done

# correctness checks: zero reward at initialization, our accuracy against the trainer's, our scorer against the trainer's forward pass, same data order
$PYTHON -m pytest -q -s tests
