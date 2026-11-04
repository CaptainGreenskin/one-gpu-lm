#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
root=${1:?provide a new run directory}
mkdir -p "$(dirname "$root")"
mkdir "$root"
exec > >(tee "$root/console.log") 2>&1
python scripts/wiki_zh.py verify
python scripts/check_gpu.py | tee "$root/environment.json"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,power.limit --format=csv > "$root/hardware.csv"
for seed in 42 43 44; do
  run="$root/seed-$seed"
  echo "Training seed $seed at 100 and 1000 steps"
  python experiments/002-first-train/train.py --device cuda --seed "$seed" --steps 1000 --snapshot-step 100 \
    --train data/wiki-zh-first/train.txt --val data/wiki-zh-first/val.txt --run-dir "$run"
  for step in 100 1000; do
    ckpt="$run/model.pt"
    if [[ "$step" == 100 ]]; then ckpt="$run/step-100.pt"; fi
    python experiments/003-training-budget/evaluate.py --checkpoint "$ckpt" --out "$run/eval-$step.json"
  done
done
echo "Completed all three seeds."
