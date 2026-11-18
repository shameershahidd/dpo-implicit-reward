#!/usr/bin/env bash
# usage: bash scripts/run_all.sh tierA
# every step skips work that is already on disk, so after a disconnect just run the same command again
set -euo pipefail
cd "$(dirname "$0")/.."
TIER=${1:?usage: run_all.sh <tier>}
CONFIG=configs/$TIER.yaml
PYTHON=${PYTHON:-python}

# three learning rates for each method, so the baseline is tuned as much as dpo
# the DPO learning rate is tuned only at beta 0.1 and reused for the other betas, a full grid per beta would triple the cost
RM_LRS=${RM_LRS:-"5e-5 1e-4 2e-4"}
DPO_LRS=${DPO_LRS:-"2e-5 5e-5 1e-4"}
BETAS=${BETAS:-"0.05 0.1 0.5"}
BETA_FOR_LR=0.1

mkdir -p results/$TIER
# a run that crashes is logged and the script goes on, select_best.py lists the crashed run as dropped
try() { "$@" || echo "FAILED: $*" | tee -a results/$TIER/failures.log; }

echo "== 1. data prep"
[ -f data/processed/$TIER/data_report.json ] || $PYTHON src/data.py --config $CONFIG

echo "== 2. reward model learning rate grid"
for lr in $RM_LRS; do
  try $PYTHON src/train_rm.py --config $CONFIG --lr $lr
done

echo "== 3. dpo learning rate grid at beta $BETA_FOR_LR"
for lr in $DPO_LRS; do
  try $PYTHON src/train_dpo.py --config $CONFIG --lr $lr --beta $BETA_FOR_LR
done

echo "== 4. pick the learning rates on validation accuracy"
$PYTHON src/select_best.py --config $CONFIG --beta_for_lr $BETA_FOR_LR
RM_RUN=$($PYTHON -c "import json; print(json.load(open('results/$TIER/selected.json'))['rm']['run_dir'])")
DPO_LR=$($PYTHON -c "import json; print(json.load(open('results/$TIER/selected.json'))['dpo']['lr'])")

echo "== 5. beta ablation at dpo learning rate $DPO_LR (the beta $BETA_FOR_LR run already exists and is reused)"
for beta in $BETAS; do
  try $PYTHON src/train_dpo.py --config $CONFIG --lr $DPO_LR --beta $beta
done

echo "== 6. score on the test split, and on val for check b"
# check b: our strict val accuracy must match the accuracy the trainer recorded, or the scorer is wrong
SCORED="$RM_RUN"
try $PYTHON src/score.py --config $CONFIG --run_dir "$RM_RUN" --splits test val
for beta in $BETAS; do
  for run in results/$TIER/dpo_beta${beta}_lr${DPO_LR}_seed*; do
    try $PYTHON src/score.py --config $CONFIG --run_dir "$run" --splits test val
    SCORED="$SCORED $run"
  done
done
$PYTHON src/score.py --config $CONFIG --baselines --splits test
$PYTHON scripts/check_b.py --config $CONFIG --run_dirs $SCORED

echo "== 7. summary"
$PYTHON scripts/summary.py --config $CONFIG
