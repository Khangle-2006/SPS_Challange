#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/data_pool/khangle/facevitals/envs/hr/bin/python
until "$PY" -c 'import json;from pathlib import Path;p=Path("/data_pool/khangle/facevitals/results/cv_5fold_20261020");assert (p/"SECONDARY_COMPLETE.json").exists();assert all((p/f"fold{k}"/s/"status.json").exists() and json.loads((p/f"fold{k}"/s/"status.json").read_text())["state"]=="complete" for k in range(5) for s in ["mse","spectral"])' 2>/dev/null; do
 sleep 20
done
"$PY" -u evaluate_cv_loss.py
"$PY" -u package_cv_loss.py
