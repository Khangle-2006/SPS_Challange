import json
import numpy as np
import torch
from stress_ubfc import ROOT,DATA
from train_efficientphys import Clips
from EfficientPhys_fixed import EfficientPhys
from inference_hr import predict_hr
from hr_decoder import LEGACY_BAND_HZ

torch.set_num_threads(2)
model=EfficientPhys(frame_depth=10,img_size=36).cuda()
model.load_state_dict(torch.load(ROOT/'results/efficientphys36_ubfc_v1/best.pt',map_location='cuda',weights_only=False)['model'])
rows=json.loads((ROOT/'results/efficientphys36_ubfc_v1/best_validation.json').read_text())['records']
ds=Clips(DATA,'val');tested=0
for i,r in enumerate(rows):
    ds[i];raw,t,ppg=ds.cache[r['subject']]
    q=r['start_s']+np.arange(301)/30;j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1)
    w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)[:,None,None,None]
    x=raw[j-1].astype(np.float32)*(1-w)+raw[j].astype(np.float32)*w
    result=predict_hr(model,x,band=LEGACY_BAND_HZ)
    assert result['hr_bpm'] is not None and abs(result['hr_bpm']-r['pred_hr'])<1e-8
    x[:]=x[0];result=predict_hr(model,x)
    assert result['hr_bpm'] is None and result['quality']['reason']=='NO_TEMPORAL_VARIATION'
    tested+=1
result={'clean_validation_predictions_unchanged':tested,'frozen_validation_clips_rejected':tested,
        'scope':'Inference wrapper validated on seed2027 and 42 validation clips; not a general low-SNR confidence estimator.'}
(ROOT/'results/stress_ubfc_v1/guarded_inference_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
