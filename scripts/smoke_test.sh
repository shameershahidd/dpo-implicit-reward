#!/usr/bin/env bash
# end to end check on a tiny subset, runs on cpu. use PYTHON=path/to/python to pick the interpreter
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}

# start clean so the runs are not skipped
rm -rf results/smoke

$PYTHON src/data.py --config configs/smoke.yaml
$PYTHON src/train_rm.py --config configs/smoke.yaml --lr 1e-4
$PYTHON src/train_dpo.py --config configs/smoke.yaml --lr 5e-5 --beta 0.1
