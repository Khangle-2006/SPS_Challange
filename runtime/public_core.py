"""Fixed public P4m spectral/fusion path; no fit or parameter search."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy import signal
from common import ROOT,save,sha
from btc_spectral_math import supported,trend,pos_team,chrom_team
from reader_v4 import MODELS

ASSETS=ROOT/'checkpoints/r1/facevitals-inference-assets-r1-20261124'
sys.path.insert(0,str(ASSETS/'code'))
from public_fixed_math import BPM,ma3,qscale,marg,point
F=np.fft.rfftfreq(4096,1/30.);BAND=(F>=.7)&(F<=3.)
SOS=signal.butter(1,[.7,3.],btype='bandpass',fs=30,output='sos')


def spectrum(w):
    w=np.asarray(w,float)
    if w.shape!=(300,) or not np.isfinite(w).all() or w.std()<1e-10:return None
    y=signal.sosfiltfilt(SOS,w-trend(300)(w))
    p=signal.periodogram(y,fs=30,window='hann',nfft=4096,detrend=False)[1][BAND]
    return p.astype(np.float32) if np.isfinite(p).all() and p.sum()>1e-30 else None


def merge(t,*arrays):
    u,inv,cnt=np.unique(t,return_inverse=True,return_counts=True);out=[]
    for a in arrays:
        if a.dtype==bool:
            m=np.ones(len(u),bool);np.logical_and.at(m,inv,a)
        else:
            m=np.zeros((len(u),)+a.shape[1:],np.float32);np.add.at(m,inv,a)
            m/=cnt.reshape((-1,)+(1,)*(a.ndim-1))
        out.append(m)
    return [u]+out


def main(out):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    from EfficientPhys_fixed import EfficientPhys
    from replicate_tsm import use_replicate_padding
    sys.path.insert(0,str(ROOT/'tools'));from check_inference_assets import verify
    verify(ASSETS)
    req=json.loads((out/'request.json').read_text());front=json.loads((out/'frontend.json').read_text())
    modelpath=ROOT/'models/public_p4m_r1.json';mf=json.loads((ROOT/'models/PUBLIC_MANIFEST.json').read_text())
    if sha(modelpath)!=mf['sha256'][modelpath.name]:raise ValueError('Reader hash mismatch')
    fits=json.loads(modelpath.read_text());name='deploy' if req['fold'] is None else 'fold'+str(req['fold'])
    w=np.asarray(fits['w_'+name]);g,sigma,tau,est,ref,_=fits['ro_'+name+'_'+req['dataset']]
    cp=ASSETS/'checkpoints'/f"padding_fixed_fold{req['cnn_fold']}_seed2027.pt"
    model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36))
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
    with np.load(out/'trace.npz') as z:
        t=z['t'];v=z['valid'];sv=z['source_valid'];starts=z['window_start'];eligible=z['window_eligible']
    crop=np.load(out/'oval36.npy',mmap_mode='r')
    blk=crop.reshape(len(crop),6,6,6,6,3).mean((2,4),dtype=np.float32)
    tu,crop,v,blk=merge(t,crop,v,blk);v &= sv
    filtered,k=ma3(crop,tu,v)
    # Original x13/x02 path stores float16 filtered crops, then merges to float32.
    filtered=filtered.astype(np.float16).astype(np.float32)
    rgb=blk.astype(float).mean((1,2));rows=[];spectra=[];waves={};indices=[]
    ew={} if front['static_input'] else dict(np.load(out/'expert_waves.npz',allow_pickle=False))
    for i,start in enumerate(starts):
        row=dict(window_start_s=float(start),window_end_s=float(start+10),hr_bpm=None,quality=0.,abstain=True,reason='unsupported_window')
        rows.append(row)
        if front['static_input']:row['reason']='static_input';continue
        if not eligible[i] or not supported(tu,v,start,start+10):continue
        q=start+np.arange(301)/30.;ix=np.searchsorted(tu,q,side='right').clip(1,len(tu)-1)
        a=((q-tu[ix-1])/(tu[ix]-tu[ix-1])).astype(np.float32)
        raw=filtered[ix-1]*(1-a[:,None,None,None])+filtered[ix]*a[:,None,None,None]
        if not np.isfinite(raw).all() or raw.std()<1e-6:row['reason']='flat_input';continue
        x=((raw-raw.mean())/raw.std()).transpose(0,3,1,2).copy()
        with torch.inference_mode():d=model(torch.from_numpy(x[None]))[0].double().numpy()
        rgbw=np.stack([np.interp(q[:-1],tu,rgb[:,c]) for c in range(3)],1)
        if rgbw.mean(0).min()<=0:continue
        try:ww=[np.cumsum(d),pos_team(rgbw),chrom_team(rgbw)]
        except (ValueError,IndexError):continue
        for nm in MODELS:
            ts=ew[nm+'_t']
            if len(ts)<300 or start<ts[0]-.2 or start+10>ts[-1]+.2:break
            ww.append(np.interp(q[:-1],ts,ew[nm]))
        if len(ww)!=6:continue
        sp=[spectrum(y) for y in ww]
        if any(x is None for x in sp):row['reason']='flat_or_invalid_spectrum';continue
        ll=np.log(np.asarray(sp,dtype=np.float64)+1e-30)
        zz=(ll-ll.mean(1,keepdims=True))/ll.std(1,keepdims=True)
        spectra.append(np.concatenate([zz[:3],zz[3:].mean(0,keepdims=True)]));indices.append(i)
        waves['window_'+str(i)]=np.stack([d]+ww)
        row['reason']=None
    if indices:
        indices=np.asarray(indices);Z=np.asarray(spectra).transpose(1,0,2);E=np.tensordot(w,Z,1)
        cuts=np.flatnonzero(np.abs(np.diff(starts[indices])-10)>=.6)+1
        for run in np.split(np.arange(len(indices)),cuts):
            pred=point(marg(qscale(E[run],g,ref),sigma,tau,np.zeros(len(BPM))),'median' if est==1 else 'mean')
            for j,h in zip(run,pred):rows[indices[j]].update(hr_bpm=float(h),quality=float(E[j].max()/ref),abstain=False)
        np.savez_compressed(out/'spectra.npz',start=starts[indices],Z=Z)
    np.savez_compressed(out/'pulse_waves.npz',**waves)
    with (out/'processed_output.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['window_start_s','window_end_s','hr_bpm','quality','abstain','reason']);writer.writeheader();writer.writerows(rows)
    save(out/'result.json',dict(pipeline='P4m R1',dataset=req['dataset'],fold=req['fold'],cnn_fold=req['cnn_fold'],checkpoint_sha256=sha(cp),reader_sha256=sha(modelpath),prefilter_k=k,window_seconds=10,stride_seconds=10,windows=rows,holdout_reads=0,raw_video=True,validation_status='See S1 A1 receipt; CLI availability is not a parity PASS'))


if __name__=='__main__':main(Path(sys.argv[1]))
