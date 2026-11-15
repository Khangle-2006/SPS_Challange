"""Explain seed-2027 validation regressions without changing any estimator."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import signal
from train_efficientphys import Clips
from EfficientPhys_fixed import EfficientPhys
from hr_baseline import detrend,estimate_hr
ROOT=Path('/data_pool/khangle/facevitals')
out=ROOT/'results/efficientphys36_three_seeds';out.mkdir(exist_ok=True,parents=True)
rows=json.loads((ROOT/'results/efficientphys36_ubfc_v1/best_validation.json').read_text())['records']
data=Clips(ROOT/'datasets/UBFC-rPPG/efficientphys36_trainval_v1','val')
assert len(rows)==len(data)==42
torch.set_num_threads(2)
model=EfficientPhys(frame_depth=10,img_size=36).cuda().eval()
ck=torch.load(ROOT/'results/efficientphys36_ubfc_v1/best.pt',map_location='cuda',weights_only=False)
model.load_state_dict(ck['model'])
def peaks(wave):
    sos=signal.butter(1,[.75,3.],btype='bandpass',fs=30,output='sos')
    f,p=signal.periodogram(signal.sosfiltfilt(sos,detrend(wave)),fs=30,window='hann',nfft=4096)
    ix=[i for i in signal.find_peaks(p)[0] if .75<=f[i]<=3]
    ix=sorted(ix,key=lambda i:p[i],reverse=True)[:3]
    return [{'bpm':float(f[i]*60),'relative_to_main_peak':float(p[i]/p[ix[0]])} for i in ix]
result=[];grid_check=[]
with torch.no_grad():
    for i,r in enumerate(rows):
        x,y,_=data[i]
        assert data.rows[i]['subject']==r['subject'] and data.rows[i]['start_s']==r['start_s']
        target=np.concatenate([[0.],np.cumsum(y.numpy()[:-1])])
        grid_check.append(estimate_hr(target,30)-r['gt_hr'])
        pos_error=abs(r['pos_hr']-r['gt_hr'])
        if r['abs_error']<=pos_error:continue
        pred=model(x[None].cuda())[0].cpu().numpy()
        wave=np.concatenate([[0.],np.cumsum(pred[:-1])])
        hr=estimate_hr(wave,30.)
        assert abs(hr-r['pred_hr'])<1e-8
        result.append({**r,'pos_abs_error':pos_error,'extra_error_bpm':r['abs_error']-pos_error,
                       'extra_error_fft_bins':(r['abs_error']-pos_error)/(30*60/4096),
                       'predicted_peaks':peaks(wave),'training_target_peaks':peaks(target)})
summary={'seed':2027,'regressions':len(result),'fft_grid_bpm':30*60/4096,
         'all_extra_errors_below_1bpm':all(r['extra_error_bpm']<1 for r in result),
         'max_extra_error_bpm':max(r['extra_error_bpm'] for r in result),
         'max_total_prediction_error_in_regressions':max(r['abs_error'] for r in result),
         'max_training_target_vs_evaluation_reference_hr_difference':float(max(map(abs,grid_check))),
         'target_reference_differences_bpm':grid_check,'records':result,
         'note':'FFT zero-padding grid spacing is not physiological resolution or proof the prediction is exact.'}
(out/'validation_regression_diagnostics.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('records','target_reference_differences_bpm')},indent=2))
