#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
gpu="$1"; shift
export CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>"/data_pool/khangle/facevitals/tmp/cv_folds_gpu${gpu}.lock"
flock -n 9
for fold in "$@"; do
 for loss in mse spectral; do
  /data_pool/khangle/facevitals/envs/hr/bin/python -u train_cv_loss.py --fold "$fold" --loss "$loss"
 done
done
printf 'QUEUE_COMPLETE GPU%s\n' "$gpu"
