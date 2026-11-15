"""Reload the selected checkpoint and reproduce validation metrics; no test access."""
import json, math, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from EfficientPhys_fixed import EfficientPhys
from train_efficientphys import Clips, evaluate
root=Path('/data_pool/khangle/facevitals')
out=root/'results/efficientphys36_ubfc_v1'
torch.set_num_threads(2)
status=json.loads((out/'status.json').read_text())
assert status['state']=='complete'
expected=json.loads((out/'best_validation.json').read_text())
ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
model=EfficientPhys(frame_depth=10,img_size=36).cuda()
model.load_state_dict(ck['model'],strict=True)
assert ck['epoch']+1==expected['epoch']
data=Clips(root/'datasets/UBFC-rPPG/efficientphys36_trainval_v1','val')
result=evaluate(model,DataLoader(data,batch_size=2,num_workers=2), 'cuda')
for k in ('subject_macro_MAE','window_MAE','window_RMSE','mse'):
    assert abs(result[k]-expected[k])<1e-5, (k,result[k],expected[k])
rows=result['records']
pos_errors=np.array([abs(r['pos_hr']-r['gt_hr']) for r in rows])
ep_errors=np.array([r['abs_error'] for r in rows])
comparison=[]
for subject in sorted({r['subject'] for r in rows}):
    rr=[r for r in rows if r['subject']==subject]
    comparison.append({'subject':subject,'windows':len(rr),
                       'pos_MAE':float(np.mean([abs(r['pos_hr']-r['gt_hr']) for r in rr])),
                       'efficientphys_MAE':float(np.mean([r['abs_error'] for r in rr]))})
report={'checkpoint_reload_reproduced':True,'best_epoch':expected['epoch'],
        'epochs_completed':status['completed_epochs'],'test_evaluated':False,
        'subjects':7,'windows':len(rows),'invalid_predictions':result['invalid'],
        'efficientphys_MAE':float(ep_errors.mean()),'efficientphys_RMSE':float(np.sqrt(np.mean(ep_errors**2))),
        'pos_MAE':float(pos_errors.mean()),'pos_RMSE':float(np.sqrt(np.mean(pos_errors**2))),
        'efficientphys_median_abs_error':float(np.median(ep_errors)),
        'pos_median_abs_error':float(np.median(pos_errors)),
        'windows_improved':int((ep_errors<pos_errors).sum()),'windows_worse':int((ep_errors>pos_errors).sum()),
        'windows_tied':int((ep_errors==pos_errors).sum()),'per_subject':comparison,
        'best_checkpoint_sha256':hashlib.file_digest((out/'best.pt').open('rb'),'sha256').hexdigest(),
        'original_source_hash_verification':'Google API blocked; cache transfer SHA256 and inventory passed'}
(out/'validation_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
