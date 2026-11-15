#!/usr/bin/env bash
set -euo pipefail
source /data_pool/khangle/facevitals/code/activate.sh
python -u "$FACEVITALS_ROOT/code/download_aipex.py"
python -u "$FACEVITALS_ROOT/code/aipex_baseline.py" --root "$FACEVITALS_ROOT/datasets/AiPEX" --out "$FACEVITALS_ROOT/results/aipex_pos_green_v1"

