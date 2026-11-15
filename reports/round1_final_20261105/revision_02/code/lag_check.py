"""Corrected global polarity, signed Pearson +/-20frames; no label/time edits."""
import argparse,json,csv,ast,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import signal
from metrics import csv_write,write_json
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pos-source',type=Path,required=True);a=p.parse_args();root=a.root
ns=dict(np=np,math=math);tree=ast.parse(a.pos_source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pos');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.pos_source),'exec'),ns)
w=json.loads((root/'frozen_data/FROZEN_WINDOWS.json').read_text());groups=defaultdict(list)
for r in w:
    if r['branch']=='ubfc_only' and r['seed']==2027 and r['condition'] in ('UBFC clean','MCD native') and r['accepted']:groups[r['dataset'],r['recording_id']].append(r)
rows=[];curves=[]
for (ds,rid),rr in groups.items():
    with np.load(root/'frozen_data/spatial_rgb'/ds/rid/'native.npz') as z:
        t,rgb,pt,ppg=(z[k] for k in ('t','rgb','ppg_t','ppg'))
        pp=ppg*(-1 if ds=='MCD' else 1)
        cc=[]
        for r in rr:
            q=r['window_start_s']+np.arange(300)/30
            x=np.stack([np.interp(q,t,rgb[:,k]) for k in range(3)],axis=1)
            pos=-ns['pos'](x,30);target=np.interp(q,pt,pp)
            filt=lambda x:signal.sosfiltfilt(signal.butter(2,[.7,3],fs=30,btype='bandpass',output='sos'),signal.detrend(x))[48:-48]
            u,v=filt(pos),filt(target);cs=[]
            for lag in range(-20,21):
                aa,bb=(u[:-lag],v[lag:]) if lag>0 else (u[-lag:],v[:lag]) if lag<0 else (u,v)
                cs.append(float(np.corrcoef(aa,bb)[0,1]) if min(aa.std(),bb.std())>=1e-10 else None)
            if all(v is not None and np.isfinite(v) for v in cs):cc.append(cs)
        assert cc,'No usable lag window for '+rid
        curve=np.mean(cc,axis=0);lags=list(range(-20,21));best=max(range(41),key=lambda k:(curve[k],-abs(lags[k]),-lags[k]))
        rows.append(dict(dataset=ds,camera='FullHDwebcam' if ds=='MCD' else 'UBFC2 webcam',person=rr[0]['subject_id'],recording=rid,windows=len(cc),lag_frames=lags[best],lag_seconds=lags[best]/30,peak_r=float(curve[best])))
        curves.append(dict(dataset=ds,recording=rid,lags=lags,mean_signed_r=curve.tolist()))
summary={}
for ds in ('MCD','UBFC2'):
    group=[r for r in rows if r['dataset']==ds];lag=np.array([r['lag_frames'] for r in group]);peak=[r['peak_r'] for r in group]
    summary[ds]=dict(recordings=len(group),people=len({r['person'] for r in group}),median_lag=float(np.median(lag)),IQR=np.quantile(lag,[.25,.75]).tolist(),median_abs_lag=float(np.median(abs(lag))),abs_median_lag=float(abs(np.median(lag))),fraction_abs_gt3=float(np.mean(abs(lag)>3)),median_peak_r=float(np.median(peak)))
summary['LABEL_PHASE_ISSUE']=summary['MCD']['median_abs_lag']>3
csv_write(root/'lag_after_polarity.csv',rows);write_json(root/'LAG_CURVES.json',curves);write_json(root/'LAG_SUMMARY.json',summary)
text=('# LABEL PHASE ISSUE\n\n' if summary['LABEL_PHASE_ISSUE'] else '')+'# Lag after polarity correction\n\n'+json.dumps(summary,indent=2)+'\n\nCanonicalPOS uses a fixed global minus sign for both datasets, declared before this check. CorrectedPPG uses globalMCD-1/UBFC+1. Search maximizes signedPearson in[-20,+20] at30Hz after common0.7–3Hz filtering. Curves are averaged over accepted10s windows per recording after1.6s edge trimming. Positive lag means POS[t] versusPPG[t+lag].\n\nThe flag uses median(abs(recordinglag)), with abs(medianlag) also printed. Periodic ambiguity and signal morphology limit interpretation; this is not a hardware synchronization measurement. No label,offset,decoder or training change is made.\n'
(root/'lag_after_polarity.md').write_text(text)
print(json.dumps(summary),flush=True)
