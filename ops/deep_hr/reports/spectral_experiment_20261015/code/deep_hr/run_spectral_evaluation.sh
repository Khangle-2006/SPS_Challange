#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>/data_pool/khangle/facevitals/tmp/h264_ablation.lock
flock 9
/data_pool/khangle/facevitals/envs/hr/bin/python -u evaluate_spectral_frozen.py
/data_pool/khangle/facevitals/envs/hr/bin/python -u package_spectral_results.py
