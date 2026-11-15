"""Frozen six-model, all-train-window signal diagnosis; never validation/test."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader
from train_h264_ablation import ROOT,CLEAN,COMP_TRAIN,SEEDS,Clips,EfficientPhys,verify,same_pair
from train_efficientphys import sha
from hr_baseline import detrend,estimate_hr,pos
OUT=ROOT/'results/train_compression_diagnosis_20261019'
def spectral(w,gt):
    w=np.asarray(w,dtype=float)
    if w.std()<1e-10:return {'valid':False,'hr':None,'true_band_fraction':0.,'top3':[]},np.zeros_like(w)
    filt=signal.sosfiltfilt(signal.butter(1,[.75,3],fs=30,btype='bandpass',output='sos'),detrend(w))
    f,p=signal.periodogram(filt,fs=30,window='hann',nfft=4096,detrend=False);band=(f>=.75)&(f<=3);bp=f[band]*60;power=p[band]
    peaks=signal.find_peaks(power)[0].tolist();peaks=sorted(peaks,key=lambda j:power[j],reverse=True)[:3]
    return {'valid':True,'hr':estimate_hr(w,30),'true_band_fraction':float(power[np.abs(bp-gt)<=6].sum()/max(power.sum(),1e-30)),
       'top3':[{'bpm':float(bp[j]),'relative_to_max':float(power[j]/max(power.max(),1e-30))} for j in peaks]},filt
def corr(a,b):
    if np.std(a)<1e-10 or np.std(b)<1e-10:return None
    return float(np.corrcoef(a,b)[0,1])
def rawclip(ds,i):
    row=ds.rows[i];s=row['subject'];meta=np.load(ds.root/(s+'_meta.npz'));raw=np.load(ds.root/(s+'_rgb.npy'),mmap_mode='r');t=meta['t']
    q=row['start_s']+np.arange(301)/30;j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1);w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)[:,None,None,None]
    return raw[j-1].astype(np.float32)*(1-w)+raw[j].astype(np.float32)*w
def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    datasets={'clean':Clips(CLEAN,'train'),'h264_500k':Clips(COMP_TRAIN,'train')}
    verify(CLEAN);verify(COMP_TRAIN);same_pair(*datasets.values());assert len(datasets['clean'])==161
    checkpoints={}
    for family,folder in [('mse','h264_ablation_v1'),('spectral','spectral_ablation_v1')]:
        for seed in SEEDS:
            p=ROOT/'results'/folder/f'augmentation_{seed}'/'best.pt';st=json.loads((p.parent/'status.json').read_text())
            assert st['state']=='complete' and st['reload_verified'];h=sha(p);assert h==st['checkpoint_sha256']
            checkpoints[f'{family}_{seed}']={'path':str(p),'sha256':h}
    protocol={'checkpoints':checkpoints,'train_windows':161,'train_subjects':28,'conditions':['clean','h264_500k'],
       'true_band_bpm_halfwidth':6,'spatial_grid':[3,3],'maximum_correlation_lag_s':.5,'test_opened':False,'validation_opened':False,
       'note':'Descriptive diagnostics on all train windows. Reference-conditioned band power and best patch are oracle diagnostics, never an inference selector or evidence that absent pulse information is irrecoverable. No new weights, decoder or estimator selection.',
       'code_sha256':sha(Path(__file__))}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));records=[];traces={}
    for i,row in enumerate(datasets['clean'].rows):
        _,y,_=datasets['clean'][i];target=np.r_[0.,np.cumsum(y.numpy()[:-1])];targetstats,ref=spectral(target,row['gt_hr'])
        r={**row,'reference':targetstats,'input':{},'models':{}}
        for c,ds in datasets.items():
            raw=rawclip(ds,i);rgb=raw.mean(axis=(1,2));pr=pos(rgb,30.)[:300]
            pst,pf=spectral(pr,row['gt_hr']);gst,gf=spectral(rgb[:300,1],row['gt_hr'])
            patches=[]
            for yy in range(3):
                for xx in range(3):
                    small=raw[:,yy*12:(yy+1)*12,xx*12:(xx+1)*12].mean(axis=(1,2));ps,_=spectral(pos(small,30.)[:300],row['gt_hr'])
                    patches.append({'row':yy,'col':xx,**ps})
            meta=np.load(ds.root/(row['subject']+'_meta.npz'));mask=(meta['original_t']>=row['start_s'])&(meta['original_t']<=row['start_s']+10)
            boxes=meta['boxes'][mask]
            r['input'][c]={'pos':pst,'green':gst,'patch_pos':patches,'max_coordinate_step_px':int(np.abs(np.diff(boxes,axis=0)).max()),
                'pos_abs_corr_zero':abs(corr(pf,ref)) if corr(pf,ref) is not None else None,
                'oracle_best_patch_true_band_fraction':max(p['true_band_fraction'] for p in patches)}
        records.append(r)
    (OUT/'input_diagnostics.json').write_text(json.dumps(records,indent=2));print('INPUT SIGNALS COMPLETE',flush=True)
    loaders={c:DataLoader(ds,batch_size=2,num_workers=2) for c,ds in datasets.items()}
    for name,m in checkpoints.items():
        p=Path(m['path']);assert sha(p)==m['sha256'];model=EfficientPhys(frame_depth=10,img_size=36).cuda().eval()
        model.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model'])
        prior=json.loads((ROOT/'results/spectral_ablation_v1/frozen_evaluation'/f'{name.split("_")[0]}_augmentation_{name.split("_")[1]}.json').read_text())
        for c,dl in loaders.items():
            before={(r['subject'],r['window']):r for r in prior['train_conditions'][c]['records']}
            with torch.no_grad():
                for x,y,ids in dl:
                    pred=model(x.cuda()).cpu().numpy()
                    for j,i in enumerate(ids.tolist()):
                        row=records[i];wave=np.r_[0.,np.cumsum(pred[j,:-1])];s,f=spectral(wave,row['gt_hr']);old=before[(row['subject'],row['window'])]
                        assert abs(s['hr']-old['pred_hr'])<1e-8
                        s['error']=abs(s['hr']-row['gt_hr']);ref=np.r_[0.,np.cumsum(y[j].numpy()[:-1])];_,rf=spectral(ref,row['gt_hr'])
                        lags=[]
                        for lag in range(-15,16):
                            a,b=(f[lag:],rf[:-lag]) if lag>0 else ((f[:lag],rf[-lag:]) if lag<0 else (f,rf))
                            co=corr(a,b)
                            if co is not None:lags.append(abs(co))
                        s['max_abs_corr_within_0_5s']=max(lags) if lags else None
                        row['models'][name+'_'+c]=s
                        traces[f'{i}_{name}_{c}']=wave
        del model;print('FROZEN MODEL VERIFIED',name,flush=True)
    np.savez_compressed(OUT/'model_waveforms.npz',**traces)
    (OUT/'results.json').write_text(json.dumps({'records':records,'protocol_sha256':sha(OUT/'protocol.json'),'all_saved_HR_predictions_reproduced':True,'complete':True,'test_opened':False,'validation_opened':False},indent=2))
    print('ALL TRAIN DIAGNOSTICS COMPLETE',flush=True)
if __name__=='__main__':main()
