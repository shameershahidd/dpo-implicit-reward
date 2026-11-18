#!/usr/bin/env bash
# the first thing to run on colab, a few minutes to find problems before a long run
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
export PYTHON

echo "== gpu"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PYTHON - <<'PY'
import torch
assert torch.cuda.is_available(), "no gpu visible to torch: choose Runtime > Change runtime type > T4 GPU, and check the torch cuda build"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY

# start clean so nothing is skipped
rm -rf results/smoke results/throwaway

echo "== smoke tier on the gpu, fp16 autocast"
$PYTHON src/data.py --config configs/smoke.yaml
$PYTHON src/train_rm.py --config configs/smoke.yaml --lr 1e-4
$PYTHON src/train_dpo.py --config configs/smoke.yaml --lr 5e-5 --beta 0.1
RM=$(ls -d results/smoke/rm_lr*)
DPO=$(ls -d results/smoke/dpo_beta*)
$PYTHON src/score.py --config configs/smoke.yaml --run_dir "$RM" --splits val
$PYTHON src/score.py --config configs/smoke.yaml --run_dir "$DPO" --splits val

echo "== check b on the gpu: our strict accuracy against the accuracy the trainer reported"
if ! $PYTHON scripts/check_b.py --config configs/smoke.yaml --run_dirs "$RM" "$DPO"; then
  echo
  echo "CHECK B FAILED ON THE GPU. Our scorer and the trainer disagree by more than 0.01."
  echo "The pairs printed above are the closest to a tie, the ones a small numeric difference can flip."
  echo "The same check passed on the cpu in fp32, so look at what is different here:"
  echo "  1. fp16 autocast: the trainer sums log probs in fp16, our scorer casts to fp32 first, near ties can flip"
  echo "  2. a tokenization or padding difference that only shows up on the gpu"
  echo "Do not start run_all.sh. Send the output above."
  exit 1
fi

echo "== time 10 optimizer steps with the tierA settings (per device batch 2, accumulation 8)"
[ -f data/processed/tierA/data_report.json ] || $PYTHON src/data.py --config configs/tierA.yaml
$PYTHON src/train_rm.py --config configs/tierA.yaml --lr 1e-4 --max_steps 10
$PYTHON src/train_dpo.py --config configs/tierA.yaml --lr 5e-5 --beta 0.1 --max_steps 10
$PYTHON - <<'PY'
import glob, json, torch
total = torch.cuda.get_device_properties(0).total_memory / 1e9
for path in sorted(glob.glob("results/throwaway/*/metrics.json")):
    m = json.load(open(path))
    print(f"{m['method']:4s} {m['seconds_per_step']:6.2f} seconds per step, peak gpu memory {m['peak_gpu_memory_gb']:.2f} GB of {total:.1f} GB")
print("a larger per device batch (with less accumulation) fits only if the peak is well under half of the total")
PY
echo "== gpu_check finished, look at the lines above before starting run_all.sh"
