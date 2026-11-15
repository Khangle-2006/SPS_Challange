"""CPU-only real-cache check; never calls a training entry point."""
import os,sys,json,hashlib
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
code=Path('/data_pool/khangle/facevitals/code/t8_polarity_v2_20261102')
sys.path.insert(0,str(code))
import numpy as np
import t8_run as run
import polarity_guard as guard
from t8_plan import code_fingerprint,VARIANTS
from t8_t1_loader import load_clip
os.environ['CUDA_VISIBLE_DEVICES']=''
frozen=json.loads((code/'FROZEN_SHA256.json').read_text())
assert code_fingerprint(code)==frozen
root=Path('/mnt/samsung_t5/khangle/facevitals/t8_oval_20261031')
index=json.loads((root/'training_index.json').read_text())
checks=[]
for ds in ['UBFC2','MCD']:
    record=next(r for r in index['records'] if r['dataset']==ds and r['eligible_starts'])
    rid=record['recording_id'];start=record['eligible_starts'][0]
    for variant in VARIANTS:
        folder=Path(index['roots'][ds])/rid/variant/'crop'
        original_x,original_y,_=load_clip(folder,start)
        x,y=run.batch_arrays(index,[(ds,rid,start,variant)])
        np.testing.assert_array_equal(x[0],original_x)
        np.testing.assert_array_equal(y[0],original_y*(-1 if ds=='MCD' else 1))
        checks.append(dict(dataset=ds,record=rid,start=start,variant=variant,input_identical=True,target_sign_correct=True))
try:
    handle,_=guard.acquire_completed_baseline(root/'runs')
except RuntimeError as e:
    assert 'BASELINE_WAIT:' in str(e)
    refusal=str(e)
else:
    handle.close();raise AssertionError('Expected currently active original queue to refuse')
assert 'torch' not in sys.modules
out=Path('/mnt/samsung_t5/khangle/facevitals/t8_label_diagnostic_20261102/STAGED_V2_VALIDATION.json')
result=dict(state='pass_cpu_only',checks=checks,baseline_barrier=refusal,torch_imported=False,corrected_GPU_started=False,frozen_manifest_sha256=hashlib.sha256((code/'FROZEN_SHA256.json').read_bytes()).hexdigest(),correction_sha256=hashlib.sha256((code/'CORRECTION.json').read_bytes()).hexdigest())
out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'}));print('ACTUAL_CACHE_CHECKS',len(checks))
