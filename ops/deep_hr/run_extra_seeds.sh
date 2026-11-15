#!/usr/bin/env bash
set -euo pipefail
ROOT=/data_pool/khangle/facevitals
export TMPDIR="$ROOT/tmp" XDG_CACHE_HOME="$ROOT/cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1
cd "$ROOT/code/deep_hr"
exec 9>"$ROOT/logs/efficientphys36_v1.lock"
flock -n 9 || { echo 'Training already running'; exit 1; }
for seed in 2028 2029; do
  "$ROOT/envs/hr/bin/python" -u train_efficientphys.py \
    --root "$ROOT/datasets/UBFC-rPPG/efficientphys36_trainval_v1" \
    --out "$ROOT/results/efficientphys36_ubfc_seed$seed" --config "config_seed$seed.json"
done
