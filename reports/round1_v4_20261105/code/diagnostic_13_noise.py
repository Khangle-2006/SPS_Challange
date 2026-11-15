"""Existing dev traces only: matched-frame codec error, fixed10recordings, no GPU."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy import signal
BASE=Path('/data/khangle/facevitals');OUT=BASE/'reports/round1_v4_20261105/diagnostic_13'
CODE=BASE/'code/t9_fps_codec_v2_20261031';sys.path.insert(0,str(CODE))
from hr_baseline import pos
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=json.loads((Path(__file__).parent/'folds_frozen.json').read_text())
subjects=sorted({s for f in spec['folds'] for s in f['ubfc_val_subjects']},key=lambda x:int(x.replace('subject','')))[:10]
assert set(subjects).isdisjoint(spec['original_ubfc_test_subjects'])
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'SUMMARY.json').exists()
(OUT/'PREDECLARED.json').write_text(json.dumps(dict(subjects=subjects,selection='first10 numeric-sorted frozen dev35; no outcome selection',PSD='RGB and POS resampled to30Hz as frozen decoder frontend; Welch density Hann300 overlap150 nfft4096 linear detrend; trim48 bothends',reading='median paired recording density ratio10/30>=2 supports the specified prediction; not causal proof',ROI_limitation='Existing traces independently track each stream, so error includes ROI tracking differences; this is not pure encoder quantization noise.'),indent=2))
rows=[];receipts=[];failures=[]
for rid in subjects:
 try:
  d=BASE/'t9_fps_codec_v2_20261031'/rid;qa=BASE/'t6_qalite_20261031/t5_UBFC2_ovalv2'/rid/'trace.npz'
  with np.load(d/'native_trace.npz') as n,np.load(d/'h26430_trace.npz') as c,np.load(d/'raw10_trace.npz') as r,np.load(qa) as q:
   assert np.array_equal(n['source_indices'],c['source_indices'])
   np.testing.assert_allclose(n['t'],c['t'],rtol=0,atol=1e-9)
   np.testing.assert_allclose(r['t'],q['timestamps_s'],rtol=0,atol=1e-9)
   np.testing.assert_allclose(n['t'][r['source_indices']],r['t'],rtol=0,atol=1e-9)
   streams={30:(n['t'],n['rgb'],c['rgb'],n['oval_points'],c['oval_points']),10:(r['t'],r['rgb'],q['rgb'][:,0,:],r['oval_points'],q['oval_points'])}
   lo=max(streams[f][0][0] for f in streams);hi=min(streams[f][0][-1] for f in streams);grid=np.arange(math.ceil(lo*30),math.floor(hi*30)+1)/30
   densities={};geometries={}
   for fps,(t,clean,comp,np_,cp_) in streams.items():
    assert np.isfinite(clean).all() and np.isfinite(comp).all(),'Invalid trace frame; preserve failure'
    # Collapse any exact timestamp duplicates in both aligned traces identically.
    ts,inv=np.unique(t,return_inverse=True);counts=np.bincount(inv)
    def sample(x):
     x=np.stack([np.bincount(inv,weights=x[:,j])/counts for j in range(3)],axis=1)
     assert np.diff(ts).max()<=.2
     return np.stack([np.interp(grid,ts,x[:,j]) for j in range(3)],axis=1)
    a,b=sample(clean),sample(comp);errs=dict(R=b[:,0]-a[:,0],G=b[:,1]-a[:,1],B=b[:,2]-a[:,2],Y=(b-a)@np.array([.299,.587,.114]),POS=pos(b,30)-pos(a,30))
    geometries[fps]=float(np.sqrt(np.nanmean((np_-cp_)**2)))
    for channel,e in errs.items():
     e=e[48:-48];assert len(e)>=300
     hz,psd=signal.welch(e,fs=30,window='hann',nperseg=300,noverlap=150,nfft=4096,detrend='linear',scaling='density')
     mask=(hz>=.7)&(hz<=3);densities[fps,channel]=float(np.trapz(psd[mask],hz[mask])/(hz[mask][-1]-hz[mask][0]))
   for channel in ('R','G','B','Y','POS'):
    numerator=densities[10,channel];denominator=densities[30,channel];assert denominator>0
    rows.append(dict(recording=rid,channel=channel,density_30=denominator,density_10=numerator,ratio_10_over_30=numerator/denominator,roi_RMSE30_pixels=geometries[30],roi_RMSE10_pixels=geometries[10],same_source_frames_verified=True))
  receipts.append(dict(recording=rid,hashes={str(p):sha(p) for p in (d/'native_trace.npz',d/'h26430_trace.npz',d/'raw10_trace.npz',qa)}))
 except Exception as e:failures.append(dict(recording=rid,error=repr(e)))
 print(rid,'complete' if not failures or failures[-1]['recording']!=rid else 'failed',flush=True)
summary={}
for channel in ('R','G','B','Y','POS'):
 ratios=[r['ratio_10_over_30'] for r in rows if r['channel']==channel]
 summary[channel]=dict(recordings=len(ratios),median_ratio=float(np.median(ratios)) if ratios else None,range=[min(ratios),max(ratios)] if ratios else None,fraction_ge2=float(np.mean(np.array(ratios)>=2)) if ratios else None,fixed_reading_supported=bool(len(ratios)==10 and np.median(ratios)>=2))
with (OUT/'noise_density.csv').open('w',newline='') as f:
 if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'SOURCE_HASHES.json').write_text(json.dumps(receipts,indent=2))
(OUT/'SUMMARY.json').write_text(json.dumps(dict(subjects=subjects,channels=summary,failures=failures,limitation='Frame identity is verified; ROI tracking is NOT held fixed. Ratio includes tracking effects and cannot distinguish aliasing, quantization, motion or rate control.',fixed_30_label='all native frames; actual source/mux FPS separate, not forced exactly30',no_new_encoding=True,no_label_or_offset_changes=True),indent=2))
print('COMPLETE',json.dumps(summary),flush=True)
