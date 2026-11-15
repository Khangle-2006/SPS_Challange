"""CPU-only V4 lag association and GREEN/luma diagnostics on frozen dev traces."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import ast,csv,hashlib,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import signal,stats
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/reports/round1_final_20261105/revision_02')
OUT=Path('/mnt/samsung_t5/khangle/facevitals/reports/round1_v4_20261105/diagnostics_11_12')
CODE=Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102');sys.path.insert(0,str(CODE))
from score_round1 import reference_hr
from t8_t1_loader import supported
from t8_plan import sha
def save(name,obj):
 p=OUT/name;assert not p.exists();p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def csvsave(name,rows):
 p=OUT/name;assert not p.exists()
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bootdelta(a):
 a=np.asarray(a);assert len(a);rng=np.random.default_rng(0);b=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
 return dict(n_persons=len(a),delta=float(a.mean()),CI95=np.quantile(b,[.025,.975]).tolist())
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'SUMMARY.json').exists()
windows=json.loads((ROOT/'frozen_data/FROZEN_WINDOWS.json').read_text());lagrows=list(csv.DictReader((ROOT/'lag_after_polarity.csv').open()))
lags={r['recording']:int(r['lag_frames']) for r in lagrows if r['dataset']=='MCD'}
v3=[r for r in windows if r['branch']=='v3_joint' and r['condition']=='MCD native' and r['accepted']]
groups=defaultdict(list);joined=[];window_errors=defaultdict(list)
for r in v3:
 lag=lags[r['recording_id']];err=abs(r['prediction_hr_bpm']-r['reference_hr_bpm']);stratum='gt3' if abs(lag)>3 else 'le3'
 groups[r['subject_id'],r['seed'],stratum].append(err)
 joined.append(dict(person=r['subject_id'],recording=r['recording_id'],start=r['window_start_s'],seed=r['seed'],lag_frames=lag,abs_error=err,stratum=stratum))
 window_errors[r['subject_id'],r['recording_id'],r['window_start_s'],abs(lag)].append(err)
person_strata=defaultdict(list)
for (sid,seed,s),errs in groups.items():person_strata[sid,s].append((float(np.mean(errs)),float(np.quantile(errs,.95))))
person={k:np.mean(v,axis=0) for k,v in person_strata.items()};ids=sorted({k[0] for k in person});paired=[sid for sid in ids if (sid,'le3') in person and (sid,'gt3') in person]
lag_result=dict(descriptive={},paired_people=paired,unpaired_people=sorted(set(ids)-set(paired)))
for s in ('le3','gt3'):
 a=np.asarray([v for (sid,st),v in person.items() if st==s]);lag_result['descriptive'][s]=dict(persons=len(a),MAE=float(a[:,0].mean()),P95=float(a[:,1].mean()),seed_averaged_windows=sum(1 for k in window_errors if ('gt3' if k[3]>3 else 'le3')==s))
for j,metric in enumerate(('MAE','P95')):
 lag_result[metric+'_paired_gt3_minus_le3']=bootdelta([person[s,'gt3'][j]-person[s,'le3'][j] for s in paired]) if paired else None
xx=np.array([k[3] for k in window_errors]);yy=np.array([np.mean(v) for v in window_errors.values()]);sidarr=np.array([k[0] for k in window_errors]);rho=float(stats.spearmanr(xx,yy).statistic)
rng=np.random.default_rng(0);indices={sid:np.flatnonzero(sidarr==sid) for sid in ids};rb=[]
for _ in range(10000):
 idx=np.concatenate([indices[s] for s in rng.choice(ids,size=len(ids),replace=True)]);rb.append(float(stats.spearmanr(xx[idx],yy[idx]).statistic))
lag_result['spearman']=dict(rho=rho,person_cluster_CI95=np.quantile(rb,[.025,.975]).tolist(),n_unique_windows=len(xx),seeds_averaged_errors=True)
ci=lag_result['MAE_paired_gt3_minus_le3'];lag_result['fixed_lower_CI_gt1']=bool(ci and ci['CI95'][0]>1)
lag_result['interpretation']='Observational lag-error association; even crossing the fixed threshold does not establish causality or exclude confounding.'
csvsave('lag_oof_join.csv',joined);save('LAG_ERROR.json',lag_result);print('LAG_ERROR_COMPLETE',json.dumps(lag_result),flush=True)
# 30s non-overlapping native segments, no fit or edits to any label offset.
ns=dict(np=np,math=math);source=Path('/data_pool/khangle/facevitals/code/hr_baseline.py');tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pos');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
segments=[];segment_summary=[];filt=signal.butter(2,[.7,3],fs=30,btype='bandpass',output='sos')
for rid in sorted(lags):
 with np.load(ROOT/'frozen_data/spatial_rgb/MCD'/rid/'native.npz') as z:
  t,rgb,pt,ppg=(z[k] for k in ('t','rgb','ppg_t','ppg'));good=[]
  for start in range(0,int(min(t[-1],pt[-1]))-29,30):
   q=start+np.arange(900)/30
   if not supported(t,z['valid'],q[0],q[-1]) or not supported(pt,z['ppg_valid'],q[0],q[-1]):
    segments.append(dict(recording=rid,start=start,lag_frames=None,peak_r=None,status='unsupported'));continue
   x=np.stack([np.interp(q,t,rgb[:,k]) for k in range(3)],axis=1);p=-ns['pos'](x,30);y=np.interp(q,pt,-ppg)
   p,y=[signal.sosfiltfilt(filt,signal.detrend(a))[48:-48] for a in (p,y)];cc=[]
   for lag in range(-20,21):
    aa,bb=(p[:-lag],y[lag:]) if lag>0 else (p[-lag:],y[:lag]) if lag<0 else (p,y)
    cc.append(float(np.corrcoef(aa,bb)[0,1]) if min(aa.std(),bb.std())>1e-10 else float('nan'))
   if not np.isfinite(cc).all():segments.append(dict(recording=rid,start=start,lag_frames=None,peak_r=None,status='constant_or_nonfinite'));continue
   best=max(range(41),key=lambda k:(cc[k],-abs(k-20),-(k-20)));good.append(best-20)
   segments.append(dict(recording=rid,start=start,lag_frames=best-20,peak_r=cc[best],status='ok'))
  segment_summary.append(dict(recording=rid,segments=len(good),median_lag=float(np.median(good)) if good else None,IQR_width=float(np.subtract(*np.quantile(good,[.75,.25]))) if good else None))
csvsave('lag_30s_segments.csv',segments);csvsave('lag_30s_recordings.csv',segment_summary)
print('SEGMENT_LAG_COMPLETE',len(segments),flush=True)
# Classical POS/CHROM predictions are already frozen on exactly these windows.
base=[r for r in windows if r['branch']=='v3_joint' and r['seed']==2027 and r['condition'] in ('UBFC C5','MCD-C5')]
classical={(r['branch'],r['condition'],r['recording_id'],r['window_start_s']):r for r in windows if r['branch'] in ('POS','CHROM') and r['seed']==2027}
rgroups=defaultdict(list)
for r in base:rgroups[r['dataset'],r['recording_id']].append(r)
channel_rows=[]
for (ds,rid),rr in rgroups.items():
 with np.load(ROOT/'frozen_data/spatial_rgb'/ds/rid/'f10_b500.npz') as z:
  t,rgb=z['t'],z['rgb']
  for r in rr:
   predictions={};failure={};q=r['window_start_s']+np.arange(300)/30
   if r['accepted']:
    x=np.stack([np.interp(q,t,rgb[:,k]) for k in range(3)],axis=1)
    for method,wave in [('GREEN',x[:,1]),('Y',x@np.array([.299,.587,.114]))]:
     try:predictions[method]=reference_hr(dict(window_start_s=0.,ppg_timestamps_json=json.dumps((np.arange(301)/30).tolist()),ppg_values_json=json.dumps(np.r_[wave,wave[-1]].tolist())))
     except ValueError as e:predictions[method]=None;failure[method]=str(e)
   for method in ('GREEN','Y','POS','CHROM'):
    other=classical.get((method,r['condition'],rid,r['window_start_s']));hr=other['prediction_hr_bpm'] if other and other['accepted'] else predictions.get(method)
    channel_rows.append(dict(dataset=ds,person=r['subject_id'],recording=rid,start=r['window_start_s'],condition=r['condition'],method=method,reference_eligible=r['reference_eligible'],accepted=bool(r['accepted'] and hr is not None),hr_bpm=hr,reference_hr_bpm=r['reference_hr_bpm'],failure=failure.get(method,'')))
csvsave('channels_C5_windows.csv',channel_rows);channel_summary={}
for condition in ('UBFC C5','MCD-C5'):
 rr=[r for r in channel_rows if r['condition']==condition];lookup={(r['method'],r['person'],r['recording'],r['start']):r for r in rr};channel_summary[condition]={}
 for method in ('GREEN','Y','CHROM','POS'):
  pm=defaultdict(list);matched=defaultdict(list);deltas=defaultdict(list)
  for r in rr:
   if r['method']!=method or not r['accepted']:continue
   err=abs(r['hr_bpm']-r['reference_hr_bpm']);pm[r['person']].append(err);pos=lookup['POS',r['person'],r['recording'],r['start']]
   if pos['accepted']:deltas[r['person']].append(err-abs(pos['hr_bpm']-pos['reference_hr_bpm']));matched[r['person']].append(err)
  comparison=bootdelta([np.mean(v) for v in deltas.values()])
  channel_summary[condition][method]=dict(MAE=float(np.mean([np.mean(v) for v in pm.values()])),P95=float(np.mean([np.quantile(v,.95) for v in pm.values()])),people=len(pm),accepted=sum(map(len,pm.values())),eligible=sum(r['reference_eligible'] for r in rr if r['method']==method),paired_minus_POS=comparison,helps=comparison['delta']<=-2 and comparison['CI95'][1]<0)
save('CHANNELS.json',channel_summary)
summary=dict(lag_error=lag_result,channels=channel_summary,source_windows_sha256=sha(ROOT/'frozen_data/FROZEN_WINDOWS.json'),lag_source_sha256=sha(ROOT/'lag_after_polarity.csv'),scorer_sha256=sha(CODE/'score_round1.py'),pos_source_sha256=sha(source),bootstrap_persons=10000,bootstrap_seed=0,new_label_offsets=False,D3_opened=False,test_opened=False)
save('SUMMARY.json',summary)
(OUT/'diagnostics_report.md').write_text('# V4 CPU diagnostics 1.1 and 1.2\n\n'+json.dumps(summary,indent=2)+'\n\n30-second segment lags are descriptive, with support failures retained. P95 is person-first; seeds are averaged before comparisons. Lag is not a direct measurement of hardware delay. Tasks1.3/1.4 remain pending; no causal verdict from these associations alone.\n')
print('COMPLETE_11_12',flush=True)
