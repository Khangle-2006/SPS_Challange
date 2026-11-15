#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>/data_pool/khangle/facevitals/tmp/matched_detector_roi.lock
flock -n 9
PY=/data_pool/khangle/facevitals/envs/hr/bin/python
"$PY" -u prepare_matched_roi_training.py
"$PY" -u verify_video_roi_frontend.py
"$PY" -u verify_matched_detector_roi.py
for seed in 2027 2028 2029; do
 for arm in native detected; do
  "$PY" -u train_matched_detector_roi.py --arm "$arm" --seed "$seed"
 done
done
"$PY" -u evaluate_matched_detector_roi.py
tar -czf /data_pool/khangle/facevitals/results/matched_detector_roi_completed_20261019.tar.gz -C /data_pool/khangle/facevitals results/matched_detector_roi_20261019 code/deep_hr/train_matched_detector_roi.py code/deep_hr/prepare_matched_roi_training.py code/deep_hr/evaluate_matched_detector_roi.py code/deep_hr/run_matched_detector_roi.sh code/deep_hr/verify_matched_detector_roi.py code/deep_hr/video_roi_frontend.py datasets/UBFC-rPPG/detected_on_h264_train500_20261019/coverage.json datasets/UBFC-rPPG/detected_on_h264_train500_20261019/frontend_reports.json
