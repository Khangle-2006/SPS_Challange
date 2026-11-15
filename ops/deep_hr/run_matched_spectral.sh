#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>/data_pool/khangle/facevitals/tmp/matched_spectral.lock
flock -n 9
PY=/data_pool/khangle/facevitals/envs/hr/bin/python
for seed in 2027 2028 2029; do
 "$PY" -u train_matched_spectral.py --mode augmentation --seed "$seed"
done
"$PY" -u evaluate_matched_spectral.py
"$PY" -u diagnose_subject32_matched.py
tar -czf /data_pool/khangle/facevitals/results/matched_spectral_completed_20261019.tar.gz -C /data_pool/khangle/facevitals results/matched_spectral_20261019 code/deep_hr/train_matched_spectral.py code/deep_hr/evaluate_matched_spectral.py code/deep_hr/diagnose_subject32_matched.py code/deep_hr/MATCHED_SPECTRAL_PROTOCOL.json code/deep_hr/run_matched_spectral.sh code/deep_hr/spectral_objective.py code/deep_hr/train_matched_detector_roi.py
