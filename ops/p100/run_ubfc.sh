#!/usr/bin/env bash
set -euo pipefail
source /data_pool/khangle/facevitals/code/activate.sh
python -u "$FACEVITALS_ROOT/code/download_ubfc2.py"
python -u "$FACEVITALS_ROOT/code/hr_baseline.py" --self-test
python -u "$FACEVITALS_ROOT/code/hr_baseline.py" --root "$FACEVITALS_ROOT/datasets/UBFC-rPPG/DATASET_2" --out "$FACEVITALS_ROOT/results/ubfc_pos_green_v1"

