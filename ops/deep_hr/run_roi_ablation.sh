#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>/data_pool/khangle/facevitals/tmp/roi_ablation.lock
flock -n 9
PY=/data_pool/khangle/facevitals/envs/hr/bin/python
"$PY" -u verify_roi_caches.py
for seed in 2027 2028 2029; do
 "$PY" -u train_roi_ablation.py --mode augmentation --seed "$seed"
done
"$PY" -u evaluate_roi_ablation.py
tar -czf /data_pool/khangle/facevitals/results/roi_ema010_completed.tar.gz -C /data_pool/khangle/facevitals results/roi_ema010_ablation_v1 code/deep_hr/train_roi_ablation.py code/deep_hr/evaluate_roi_ablation.py code/deep_hr/verify_roi_caches.py code/deep_hr/run_roi_ablation.sh
