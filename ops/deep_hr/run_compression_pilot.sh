#!/usr/bin/env bash
set -euo pipefail
cd /data_pool/khangle/facevitals/code/deep_hr
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>/data_pool/khangle/facevitals/tmp/compression_ablation.lock
flock -n 9 || exit 1
PY=/data_pool/khangle/facevitals/envs/hr/bin/python
for mode in control augmentation; do
  "$PY" -u train_compression_ablation.py --seed 2027 --mode "$mode"
done
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('/data_pool/khangle/facevitals/results/compression_ablation_v1')
c=json.loads((root/'control_2027/best_validation.json').read_text())
a=json.loads((root/'augmentation_2027/best_validation.json').read_text())
improvement=1-a['selection_score']/c['selection_score']
clean_delta=a['conditions']['clean']['subject_macro_MAE']-c['conditions']['clean']['subject_macro_MAE']
result={'relative_score_improvement':improvement,'clean_mae_delta':clean_delta,'expand_to_more_seeds':improvement>=.1 and clean_delta<=.25}
(root/'pilot_decision.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
PY
