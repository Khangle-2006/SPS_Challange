#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>/data_pool/khangle/facevitals/tmp/h264_ablation.lock
flock -n 9 || exit 1
PY=/data_pool/khangle/facevitals/envs/hr/bin/python
"$PY" -u diagnose_h264_waveforms.py
"$PY" -u verify_spectral_objective.py
for mode in control augmentation; do
 for seed in 2027 2028 2029; do
  "$PY" -u train_spectral_ablation.py --mode "$mode" --seed "$seed"
 done
done
