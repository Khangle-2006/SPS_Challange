"""Frozen-model waveform diagnostics; no estimator changes or label-based corrections."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy import signal
from train_h264_ablation import CLEAN,COMP_TRAIN,COMP_VAL,ROOT,SEEDS,Clips,EfficientPhys
from hr_baseline import detrend,estimate_hr
torch.set_num_threads(2)
out=ROOT/'results/spectral_ablation_v1/diagnostics_before';out.mkdir(parents=True,exist_ok=True)
chosen={'train':[('subject11',5),('subject11',2),('subject11',3),('subject10',4)],'val':[('subject32',0),('subject9',3),('subject18',1)]}
def spectrum(w):
    filt=signal.sosfiltfilt(signal.butter(1,[.75,3],btype='bandpass',fs=30,output='sos'),detrend(w))
    f,p=signal.periodogram(filt,fs=30,window='hann',nfft=4096,detrend=False);b=(f>=.75)&(f<=3)
    return filt,f[b]*60,p[b]/max(p[b].max(),1e-30)
records=[];target_deltas=[]
for split in ['train','val']:
    clean=Clips(CLEAN,split);comp=Clips(COMP_TRAIN if split=='train' else COMP_VAL,split)
    for i,r in enumerate(clean.rows):
        if (r['subject'],r['window']) not in chosen[split]:continue
        x,y,_=clean[i];xc,yc,_=comp[i];torch.testing.assert_close(y,yc)
        target=np.r_[0.,np.cumsum(y.numpy()[:-1])];gt=estimate_hr(target,30);target_deltas.append(gt-r['gt_hr'])
        ref,bpm,ps=spectrum(target);traces={'target':target,'target_filtered':ref,'target_psd':ps,'bpm':bpm}
        meta=np.load(CLEAN/(r['subject']+'_meta.npz'));q=r['start_s']+np.arange(301)/30
        ti=meta['original_t'];mask=(ti>=q[0])&(ti<=q[-1]);boxes=meta['boxes'][mask]
        rr={**r,'target_reconstructed_hr':gt,'models':{},'roi_min_width':int(np.min(boxes[:,2]-boxes[:,0])),
            'roi_min_height':int(np.min(boxes[:,3]-boxes[:,1])),
            'roi_max_step_pixels':float(np.max(np.abs(np.diff(boxes,axis=0))))}
        for mode in ['control','augmentation']:
            for seed in SEEDS:
                name=f'{mode}_{seed}';model=EfficientPhys(frame_depth=10,img_size=36).cuda().eval()
                model.load_state_dict(torch.load(ROOT/'results/h264_ablation_v1'/name/'best.pt',map_location='cuda',weights_only=False)['model'])
                for condition,inp in [('clean',x),('h264_500k',xc)]:
                    with torch.no_grad():pred=model(inp[None].cuda())[0].cpu().numpy()
                    wave=np.r_[0.,np.cumsum(pred[:-1])];filt,bpm,power=spectrum(wave)
                    corr=[];lags=np.arange(-15,16)
                    for lag in lags:
                        aa,bb=(filt[lag:],ref[:-lag]) if lag>0 else ((filt[:lag],ref[-lag:]) if lag<0 else (filt,ref))
                        corr.append(float(np.corrcoef(aa,bb)[0,1]))
                    peaks=signal.find_peaks(power)[0];peaks=sorted(peaks,key=lambda j:power[j],reverse=True)[:3]
                    hr=estimate_hr(wave,30);label=name+'_'+condition
                    rr['models'][label]={'hr':hr,'error':abs(hr-r['gt_hr']),'correlation_zero_lag':corr[15],
                        'max_abs_correlation_within_0_5s':max(map(abs,corr)),
                        'best_abs_correlation_lag_s':float(lags[np.argmax(np.abs(corr))]/30),
                        'peaks':[{'bpm':float(bpm[j]),'relative_power':float(power[j])} for j in peaks]}
                    traces[label+'_wave']=wave;traces[label+'_psd']=power;traces[label+'_filtered']=filt
                del model
        name=f"{r['subject']}_w{r['window']}";np.savez_compressed(out/(name+'.npz'),**traces);records.append(rr)
        print('DIAGNOSED',split,name,flush=True)
result={'records':records,'max_target_HR_vs_saved_reference_delta':max(map(abs,target_deltas)),
        'test_opened':False,'note':'Lags are diagnostics only, never used to correct inputs, labels or HR estimates. Frozen original ROI geometry; diagnostic sampling is selected from previously observed errors.'}
(out/'results.json').write_text(json.dumps(result,indent=2));print('DIAGNOSTICS COMPLETE',flush=True)
