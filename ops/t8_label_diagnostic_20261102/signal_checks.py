"""Read-only native-cache polarity and exact training-target diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,hashlib,csv,argparse
from pathlib import Path
import numpy as np
from scipy import signal
CODE=Path('/data_pool/khangle/facevitals/code/t8_oval_training_20261031')
sys.path.insert(0,str(CODE))
from t8_t1_loader import load_clip
from t8_evaluate import decode_difference,reference_row
from score_round1 import reference_hr

def pos(rgb):
    out=np.zeros(len(rgb)); length=48
    for end in range(length,len(rgb)+1):
        x=rgb[end-length:end]; x=x/np.maximum(x.mean(axis=0),1e-8)
        s=np.array([[0,1,-1],[-2,1,1]])@x.T
        h=s[0]+np.std(s[0])/max(np.std(s[1]),1e-8)*s[1]
        out[end-length:end]+=h-h.mean()
    return out

def filt(x):
    return signal.sosfiltfilt(signal.butter(2,[.7,3],fs=30,btype='bandpass',output='sos'),signal.detrend(x))

def corr(a,b):
    if min(np.std(a),np.std(b))<1e-10:return float('nan')
    return float(np.corrcoef(a,b)[0,1])

def savecsv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def hr(w):
    return reference_hr(dict(window_start_s=0.,ppg_timestamps_json=json.dumps((np.arange(301)/30).tolist()),ppg_values_json=json.dumps(np.asarray(w).tolist())))

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    indexpath=Path(a.index);ix=json.loads(indexpath.read_text())
    assert hashlib.sha256(indexpath.read_bytes()).hexdigest()=='12a87b829ab5e158b430c67701fa0dd978c9a3d592e0060da3e876069108d505'
    config=dict(index_sha256=hashlib.sha256(indexpath.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),lags=list(range(-5,6)),lag_definition='positive k: POS[t] vs PPG[t+k]; choose maximum absolute r, retain sign',windows='all frozen training-eligible native 10s windows; each window equal weight within recording',POS='Wang 48-sample overlap-add, RGB spatial mean of frozen36 masked crop, interpolated to30Hz; trim1.6s each edge; both signals same zero-phase .7-3Hz order2 filter',target_test='first/middle/last eligible clip per recording; exact load_clip output; no labels or clocks changed',cycle='raw PPG resampled30Hz, linear detrend, zero-phase8Hz lowpass; trough-to-trough cycles .333-1.429s; per-cycle zscore then per-record mean; no sign flips')
    (out/'signal_config.json').write_text(json.dumps(config,indent=2))
    records=[];windows=[];targets=[];cycles={'UBFC2':[],'MCD':[]};errors=[]
    for r in ix['records']:
        ds=r['dataset'];rid=r['recording_id'];folder=Path(ix['roots'][ds])/rid/'native'/'crop'
        try:
            with np.load(folder/'meta.npz') as z:m={k:z[k] for k in z.files}
            rgb=np.load(folder/'rgb.npy',mmap_mode='r')
            means=np.empty((len(rgb),3))
            for lo in range(0,len(rgb),128):means[lo:lo+128]=np.asarray(rgb[lo:lo+128],dtype=np.float32).mean(axis=(1,2))
            rr=[];cc=[]
            starts=r['eligible_starts'];teststarts=set(starts[i] for i in sorted(set([0,len(starts)//2,len(starts)-1]))) if starts else set()
            for start in starts:
                q=start+np.arange(301)/30
                rgb30=np.stack([np.interp(q,m['t'],means[:,j]) for j in range(3)],axis=1)
                y=np.interp(q,m['ppg_t'],m['ppg'])
                p=filt(pos(rgb30));b=filt(y)
                rs=[corr(p[48:253],b[48+k:253+k]) for k in range(-5,6)]
                if not np.isfinite(rs).all():raise ValueError('nonfinite correlation')
                best=int(np.argmax(np.abs(rs))); rr.append(rs)
                windows.append(dict(dataset=ds,recording=rid,subject=r['subject_id'],start=start,r_zero=rs[5],r_best_abs=rs[best],lag_best_abs=best-5,**{'r_lag_'+str(k):rs[k+5] for k in range(-5,6)}))
                smooth=signal.sosfiltfilt(signal.butter(2,8,fs=30,output='sos'),signal.detrend(y))
                troughs=signal.find_peaks(-smooth,distance=10,prominence=.25*np.std(smooth))[0]
                for l,h in zip(troughs[:-1],troughs[1:]):
                    if 10<=h-l<=43:
                        cycle=np.interp(np.linspace(l,h,101),np.arange(len(y)),smooth)
                        if cycle.std()>1e-10:cc.append((cycle-cycle.mean())/cycle.std())
                if start in teststarts:
                    _,target,ref=load_clip(folder,start)
                    expected=np.diff(y.astype(np.float32));expected/=expected.std()
                    assert np.array_equal(expected,target)
                    row=reference_row(r,{'start_s':start},m,'native')
                    rawhr=reference_hr(row);inv=np.r_[0.,np.cumsum(target.astype(float))]
                    actual=decode_difference(target);exact=hr(inv)
                    reconstructed=inv*float(np.diff(y.astype(np.float32)).std())+float(y.astype(np.float32)[0])
                    targets.append(dict(dataset=ds,recording=rid,start=start,raw_hr=rawhr,actual_decoder_hr=actual,exact_inverse_hr=exact,actual_delta_bpm=actual-rawhr,exact_delta_bpm=exact-rawhr,target_bitwise_match=True,inverse_max_error_relative_sd=float(np.max(abs(reconstructed-y.astype(np.float32)))/y.std())))
            meanr=np.mean(rr,axis=0);best=int(np.argmax(abs(meanr)))
            records.append(dict(dataset=ds,recording=rid,subject=r['subject_id'],windows=len(rr),r_zero=float(meanr[5]),r_best_abs=float(meanr[best]),lag_best_abs=best-5,positive_window_fraction=float(np.mean(np.array(rr)[np.arange(len(rr)),np.argmax(abs(np.array(rr)),axis=1)]>0)),**{'r_lag_'+str(k):float(meanr[k+5]) for k in range(-5,6)}))
            if cc:cycles[ds].append(np.mean(cc,axis=0))
            print(ds,rid,len(rr),'target',len(teststarts),flush=True)
        except Exception as e:errors.append(dict(dataset=ds,recording=rid,error=repr(e)));print('ERROR',errors[-1],flush=True)
    for name,rows in [('record_correlations',records),('window_correlations',windows),('target_roundtrip',targets)]:
        if rows:savecsv(out/(name+'.csv'),rows)
    summary={'errors':errors,'datasets':{}}
    for ds in cycles:
        rs=[r for r in records if r['dataset']==ds];ts=[t for t in targets if t['dataset']==ds]
        summary['datasets'][ds]=dict(recordings=len(rs),windows=sum(r['windows'] for r in rs),positive_records=sum(r['r_best_abs']>0 for r in rs),negative_records=sum(r['r_best_abs']<0 for r in rs),median_signed_best_r=float(np.median([r['r_best_abs'] for r in rs])),median_zero_r=float(np.median([r['r_zero'] for r in rs])),strong_positive=sum(r['r_best_abs']>=.3 for r in rs),strong_negative=sum(r['r_best_abs']<=-.3 for r in rs),target_clips=len(ts),actual_error_gt1=sum(abs(t['actual_delta_bpm'])>=1 for t in ts),exact_error_gt1=sum(abs(t['exact_delta_bpm'])>=1 for t in ts),max_actual_abs_delta=float(max(abs(t['actual_delta_bpm']) for t in ts)),max_exact_abs_delta=float(max(abs(t['exact_delta_bpm']) for t in ts)))
    (out/'signal_summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'cycle_templates.npz',**{k:np.array(v) for k,v in cycles.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ds,color in [('UBFC2','tab:blue'),('MCD','tab:orange')]:
        arr=np.asarray(cycles[ds]);xx=np.linspace(0,1,101)
        axs[0].plot(xx,np.mean(arr,axis=0),label=f'{ds} ({len(arr)} recordings)',color=color)
        axs[0].fill_between(xx,*np.quantile(arr,[.25,.75],axis=0),alpha=.15,color=color)
        rs=[r['r_best_abs'] for r in records if r['dataset']==ds]
        axs[1].hist(rs,bins=np.linspace(-1,1,21),alpha=.5,label=ds,color=color)
    axs[0].set(xlabel='Trough-to-trough phase',ylabel='PPG z-score',title='Raw sign retained; mean and recording IQR')
    axs[1].set(xlabel='Signed Pearson r at max |r| within +/-5 frames',ylabel='Recordings',title='Native oval POS versus contact PPG')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'polarity_cycles.png',dpi=160)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--output',required=True);main(p.parse_args())
