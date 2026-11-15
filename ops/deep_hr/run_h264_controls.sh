#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>/data_pool/khangle/facevitals/tmp/h264_ablation.lock
flock -n 9 || exit 1
for seed in 2027 2028 2029; do
 /data_pool/khangle/facevitals/envs/hr/bin/python -u train_h264_ablation.py --mode control --seed "$seed"
done
