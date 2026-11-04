#!/usr/bin/env bash
# Run from any directory on the Linux workstation, with the experiment's venv active.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
data_dir=data/wiki-zh-first
experiment=experiments/002-first-train
python scripts/wiki_zh.py verify
mkdir -p runs
run_dir=$(mktemp -d "runs/pro6000-wiki-XXXXXXXX")
echo "Run directory: $run_dir"
exec > >(tee "$run_dir/console.log") 2>&1

echo "[1/4] Checking driver, visible GPU and BF16 attention backward"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv | tee "$run_dir/hardware.csv"
python scripts/check_gpu.py | tee "$run_dir/environment.json"
python -m pip freeze > "$run_dir/pip-freeze.txt"
cp "$data_dir/manifest.json" "$run_dir/data-manifest.json"

echo "[2/4] Running a 12-step pipeline check on the GPU"
python "$experiment/train.py" --device cuda --tiny --steps 12 --seed 42 \
  --train "$experiment/toy-train.txt" --val "$experiment/toy-val.txt" --run-dir "$run_dir/smoke"

echo "[3/4] Training from random initialization for 100 steps on Chinese Wikipedia"
python "$experiment/train.py" --device cuda --steps 100 --seed 42 \
  --train "$data_dir/train.txt" --val "$data_dir/val.txt" --run-dir "$run_dir/train"

echo "[4/4] Checking both runs, the corpus hashes and the final checkpoint"
python scripts/check_first_run.py "$run_dir"
