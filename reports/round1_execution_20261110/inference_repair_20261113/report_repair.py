"""Complete-stage only, person-first paired evaluation; never promotes automatically."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
import numpy as np,csv,argparse,traceback
from collections import defaultdict
from scipy.stats import spearmanr

def csvout(path,rows):
 with Path(path).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)

def stats(rows,other=None):
 if other is not None:
  by={r['recording']:r for r in other};rows=[r for r in rows if r['recording'] in by]
 people=defaultdict(list)
 for r in rows:people[r['person']].append(r)
 if not people:return dict(people=0,videos=0)
 values=[];truths=[];preds=[]
 for person in sorted(people):
  rr=people[person];e=np.array([r['prediction']-r['truth'] for r in rr]);v=np.array([abs(e).mean(),e.mean(),np.quantile(abs(e),.95)])
  if other is not None:
   d=np.array([by[r['recording']]['prediction']-r['truth'] for r in rr]);v-=np.array([abs(d).mean(),d.mean(),np.quantile(abs(d),.95)])
  values.append(v);truths.append(np.mean([r['truth'] for r in rr]));preds.append(np.mean([r['prediction'] for r in rr]))
 values=np.asarray(values);ix=np.random.default_rng(0).integers(0,len(values),(10000,len(values)));ci=np.quantile(values[ix].mean(1),[.025,.975],axis=0).T;mean=values.mean(0)
 return dict(people=len(people),videos=len({r['recording'] for r in rows}),observations=len(rows),MAE=float(mean[0]),bias=float(mean[1]),mean_person_P95=float(mean[2]),CI95_MAE=ci[0].tolist(),CI95_bias=ci[1].tolist(),CI95_mean_person_P95=ci[2].tolist(),Spearman_person_means=float(spearmanr(truths,preds).statistic) if other is None and len(people)>2 and np.ptp(truths)>0 and np.ptp(preds)>0 else None)

def window_rows(stage):
 variants=['original'] if stage=='original' else ['original','rotate','mask','rotate_mask'];result=[];coverage=[];jobs=manifest()
 for j in jobs:
  for condition in ['native','c5','c5prime_default','c5prime_bf0']:
   for variant in variants:
    root=P/'original'/j['key']/condition;classic=P/'classical_original_v2'/j['key']/condition
    if variant!='original':
     parent=P/'spatial'/j['key']/condition
     if (parent/'EXCLUDED.json').exists():coverage.append(dict(key=j['key'],condition=condition,input=variant,reason=read(parent/'EXCLUDED.json')['reason']));continue
     receipt=read(parent/'COMPLETE.json');root=Path(receipt['aliases'].get(variant,str(parent/variant)));classic=P/'classical_original_v2'/j['key']/condition if root==P/'original'/j['key']/condition else root
    if (root/'EXCLUDED.json').exists():coverage.append(dict(key=j['key'],condition=condition,input=variant,reason=read(root/'EXCLUDED.json')['reason']));continue
    cp=read(root/'COMPLETE.json');assert sha(root/'PREDICTIONS.json')==cp['rows_sha256'];rows=read(root/'PREDICTIONS.json');buckets=defaultdict(list)
    for r in rows:buckets[('padding' if r['model'].startswith('padding') else 'V3',r['start'])].append(r)
    for (method,start),rs in buckets.items():
     expected=15 if method=='V3' and j['dataset']=='BTC' else 1;assert len(rs)==expected
     for readout in rs[0]['readouts']:
      vals=[r['readouts'][readout] for r in rs];valid=all(v is not None for v in vals);base=dict(dataset=j['dataset'],recording=j['recording'],person=j['person'],fold=j['fold'],condition=condition,input=variant,method=method,readout=readout,start=start,truth=rs[0]['truth'],prediction=float(np.median(vals)) if valid else None,valid=valid)
      result.append(base)
    for r in read(classic/'CLASSICAL.json'):
     for low,v in r['readouts'].items():result.append(dict(dataset=j['dataset'],recording=j['recording'],person=j['person'],fold=j['fold'],condition=condition,input=variant,method=r['model'],readout='power_'+low,start=r['start'],truth=r['truth'],prediction=v,valid=v is not None))
 return result,coverage

def video_rows(windows):
 groups=defaultdict(list)
 for r in windows:
  if r['dataset']=='BTC':groups[tuple(r[k] for k in ['recording','condition','input','method','readout'])].append(r)
 out=[]
 for key,rs in groups.items():
  valid=[r for r in rs if r['prediction'] is not None]
  if valid:out.append(dict(valid[0],prediction=float(np.median([r['prediction'] for r in valid])),valid_windows=len(valid),candidate_windows=len(rs),near175=float(np.mean([r['prediction']>=175 for r in valid]))))
 return out

def calibrate(videos):
 lookup={(r['recording'],r['condition'],r['input'],r['method'],r['readout']):r for r in videos};out=[];fits=[]
 for condition,variant,readout in sorted({(r['condition'],r['input'],r['readout']) for r in videos if r['method']=='V3'}):
  low=readout.rsplit('_',1)[1];med=[]
  for r in videos:
   if (r['condition'],r['input'],r['method'],r['readout'])!=(condition,variant,'V3',readout):continue
   parts=[lookup.get((r['recording'],condition,variant,m,'power_'+low)) for m in ['POS','CHROM']]
   if all(p is not None for p in parts):med.append(dict(r,prediction=float(np.median([r['prediction']]+[p['prediction'] for p in parts]))))
  for fold in range(5):
   tr=sorted([r for r in med if r['fold']!=fold],key=lambda r:r['recording']);va=[r for r in med if r['fold']==fold]
   if not va:continue
   assert not {r['person'] for r in tr}&{r['person'] for r in va};counts={p:sum(r['person']==p for r in tr) for p in {r['person'] for r in tr}};w=np.sqrt([1/counts[r['person']] for r in tr]);x=np.array([[r['prediction'],1] for r in tr]);y=np.array([r['truth'] for r in tr]);coef=np.linalg.lstsq(x*w[:,None],y*w,rcond=None)[0]
   fits.append(dict(condition=condition,input=variant,readout=readout,fold=fold,a=float(coef[0]),b=float(coef[1]),training_people=sorted(counts),validation_people=sorted({r['person'] for r in va})))
   out.extend(dict(r,method='calibrated_refit',prediction=float(coef[0]*r['prediction']+coef[1])) for r in va)
 return out,fits

def main(stage):
 assert (P/'ORIGINAL_COMPLETE.json').exists() and (P/'CLASSICAL_ORIGINAL_V2_COMPLETE.json').exists() and (P/'REFERENCE_COMPLETE.json').exists()
 if stage!='original':assert (P/'SPATIAL_COMPLETE.json').exists()
 out=P/('report_'+stage);assert not out.exists();out.mkdir();windows,coverage=window_rows(stage);videos=video_rows(windows);cal,fits=calibrate(videos);videos+=cal;save(out/'CALIBRATION_FITS.json',fits);csvout(out/'WINDOWS.csv',windows);csvout(out/'BTC_VIDEOS.csv',videos);save(out/'EXCLUSIONS.json',coverage)
 buckets=defaultdict(list)
 for r in windows:buckets[tuple(r[k] for k in ['dataset','condition','input','method','readout'])].append(r)
 cover=[]
 for k,rr in buckets.items():
  valid=[r for r in rr if r['prediction'] is not None];cover.append(dict(zip(['dataset','condition','input','method','readout'],k),candidate_windows=len(rr),accepted_windows=len(valid),candidate_recordings=len({r['recording'] for r in rr}),accepted_recordings=len({r['recording'] for r in valid}),candidate_people=len({r['person'] for r in rr}),accepted_people=len({r['person'] for r in valid})))
 csvout(out/'COVERAGE.csv',cover)
 baseline=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()));base={}
 for method,label in [('constant','constant_trainfold'),('calibrated_median_POS_CHROM_V3','calibrated_native_saved')]:
  base[label]=[dict(r,prediction=float(r['prediction']),truth=float(r['truth'])) for r in baseline if r['method']==method and r['condition']=='native']
 base['padding_native_saved']=[r for r in read(E/'final_priorities_20261111/padding_evaluation/BTC_VIDEO.json') if r['condition']=='native'];jobs={j['recording']:j for j in manifest() if j['dataset']=='BTC'};subsets=defaultdict(set);all_orientation=all((P/'orientation'/j['key']/'COMPLETE.json').exists() for j in jobs.values())
 for rid,j in jobs.items():
  md=j['metadata'];activity=md['group']+'-Resting' if md['task'].lower()=='resting' else 'Walking' if md['task'].lower()=='walking' else 'Task'
  for key in ['all',activity,md['group']]:subsets[key].add(rid)
  path=P/'orientation'/j['key']/'COMPLETE.json'
  if path.exists() and all_orientation:
   a=read(path);subsets['rotated' if a['resolved'] and a['quarter_turn_ccw'] else 'upright' if a['resolved'] else 'unresolved'].add(rid)
   full=P/'spatial'/j['key']/'native/prepare/PREPARED.json';fraction=read(full)['mask_fraction_mean'] if full.exists() else a['non_skin_fraction_mean']
   if fraction is not None and fraction>.1:subsets['color_proxy_over10pct_full_frames' if full.exists() else 'color_proxy_over10pct_9sample_estimate'].add(rid)
 metrics=[];tables=[]
 for condition,variant,method,readout in sorted({tuple(r[k] for k in ['condition','input','method','readout']) for r in videos}):
  rr=[r for r in videos if tuple(r[k] for k in ['condition','input','method','readout'])==(condition,variant,method,readout)]
  for group,ids in subsets.items():
   rows=[r for r in rr if r['recording'] in ids];keys={r['recording'] for r in rows}
   for b in base.values():keys&={r['recording'] for r in b}
   rows=[r for r in rows if r['recording'] in keys]
   if not rows:continue
   a=dict(condition=condition,input=variant,method=method,readout=readout,group=group,**stats(rows));a['delta_saved_calibrated']=stats(rows,base['calibrated_native_saved']);a['delta_saved_padding']=stats(rows,base['padding_native_saved']);a['delta_constant']=stats(rows,base['constant_trainfold']);metrics.append(a)
   # Each comparison block explicitly includes both required saved baselines.
   for name,b in [('candidate',rows)]+[(k,[r for r in v if r['recording'] in keys]) for k,v in base.items()]:tables.append(dict(condition=condition,input=variant,candidate=method,readout=readout,group=group,row=name,**stats(b)))
 save(out/'BTC_METRICS.json',metrics);csvout(out/'BTC_MATCHED_TABLES.csv',tables)
 refs={(r['dataset'],r['recording'],r['start']):r for r in read(P/'REFERENCE_06.json')};pub=[]
 for ds,c,var,method,readout in sorted({tuple(r[k] for k in ['dataset','condition','input','method','readout']) for r in windows if r['dataset']!='BTC'}):
  rr=[r for r in windows if tuple(r[k] for k in ['dataset','condition','input','method','readout'])==(ds,c,var,method,readout) and r['prediction'] is not None and r['truth'] is not None];old=[r for r in windows if r['dataset']==ds and r['condition']==c and r['input']=='original' and r['method']==method and r['readout']=='power_0.7' and r['prediction'] is not None and r['truth'] is not None];keys={(r['recording'],r['start']) for r in old};rr=[r for r in rr if (r['recording'],r['start']) in keys]
  # Pair by window, then summarize within person (not mean fold means).
  rr=[dict(r,recording=r['recording']+'@'+str(r['start'])) for r in rr];old=[dict(r,recording=r['recording']+'@'+str(r['start'])) for r in old]
  if rr:pub.append(dict(dataset=ds,condition=c,input=var,method=method,readout=readout,reference='frozen_0.7',**stats(rr),delta_same_method_original=stats(rr,old)))
  new=[]
  for r in rr:
   rid=r['recording'].rsplit('@',1)[0];y=refs[ds,rid,r['start']]['new_06']
   if y is not None:new.append(dict(r,truth=y))
  if new:pub.append(dict(dataset=ds,condition=c,input=var,method=method,readout=readout,reference='versioned_0.6_same_raw_clock',**stats(new)))
 for r in pub:r['matched_windows']=r.pop('videos');r['unit_note']='person-first window errors; matched_windows is not a count of videos'
 save(out/'PUBLIC_METRICS.json',pub);csvout(out/'PUBLIC_METRICS.csv',pub)
 scores=[]
 for variant,method,readout in sorted({(r['input'],r['method'],r['readout']) for r in windows if r['dataset']!='BTC'}):
  per={};pair={};complete=True
  for ds,c in [('UBFC2','native'),('UBFC2','c5'),('MCD','native')]:
   rr=[r for r in windows if (r['dataset'],r['condition'],r['input'],r['method'],r['readout'])==(ds,c,variant,method,readout)];old=[r for r in windows if (r['dataset'],r['condition'],r['input'],r['method'],r['readout'])==(ds,c,'original',method,'power_0.7')];by={(r['recording'],r['start']):r for r in old};pp=defaultdict(list);dd=defaultdict(list)
   for r in rr:
    q=by[r['recording'],r['start']]
    if r['truth'] is None:continue
    if (r['prediction'] is None)!=(q['prediction'] is None):complete=False
    if r['prediction'] is None or q['prediction'] is None:continue
    err=abs(r['prediction']-r['truth']);pp[r['person']].append(err);dd[r['person']].append(err-abs(q['prediction']-q['truth']))
   per[ds,c]={p:float(np.mean(a)) for p,a in pp.items()};pair[ds,c]={p:float(np.mean(a)) for p,a in dd.items()}
  u=sorted(set(per['UBFC2','native'])&set(per['UBFC2','c5']));m=sorted(per['MCD','native'])
  if not u or not m:continue
  rng=np.random.default_rng(0);ui=rng.integers(0,len(u),(10000,len(u)));mi=rng.integers(0,len(m),(10000,len(m)))
  def score(d):
   a=np.array([d['UBFC2','native'][p]+d['UBFC2','c5'][p] for p in u]);b=np.array([d['MCD','native'][p] for p in m]);value=float((a.mean()+b.mean())/3);bs=(a[ui].mean(1)+b[mi].mean(1))/3;return dict(value=value,CI95=np.quantile(bs,[.025,.975]).tolist())
  scores.append(dict(input=variant,method=method,readout=readout,reference='frozen_0.7',UBFC_people=len(u),MCD_people=len(m),failure_parity=complete,S=score(per),delta_original_same_method=score(pair)))
 save(out/'PUBLIC_S.json',scores)
 text=['# Inference-only repair: '+stage,'','Complete-stage matched results; no automatic promotion. Holdout evaluations:0.','', 'Every BTC comparison block in BTC_MATCHED_TABLES.csv contains candidate, saved train-fold constant, saved native calibrated estimator and saved native padding estimator on identical videos. Saved native baselines remain native comparators even in compressed columns; they are not called compressed-condition predictions.','', 'MAE/bias/P95 are computed within person then averaged. P95 means mean of person P95s. Bootstrap10,000 paired people, seed0. New inference variants are exploratory multiple comparisons.','', '## BTC native, full matched group','', '|Input|Method|Readout|Videos/people|MAE|Bias|Delta saved calibration (95% CI)|','|---|---|---|---:|---:|---:|---|']
 for a in metrics:
  if a['condition']=='native' and a['group']=='all':
   d=a['delta_saved_calibrated'];text.append(f"|{a['input']}|{a['method']}|{a['readout']}|{a['videos']}/{a['people']}|{a['MAE']:.4f}|{a['bias']:.4f}|{d['MAE']:+.4f} [{d['CI95_MAE'][0]:+.4f}, {d['CI95_MAE'][1]:+.4f}]|")
 text+=['','This is not a new FREEZE. Full spatial/visual/codec/static review and a human-readable interpretation remain required. V3 retains its known static failure; no TSM change was made.','', 'Public frozen-reference and .6Hz-reference columns are separate. Do not credit a change in reference labels as a model improvement.']
 (out/'REPORT.md').write_text('\n'.join(text),encoding='utf8');save(out/'COMPLETE.json',dict(utc=now(),stage=stage,window_rows=len(windows),BTC_video_rows=len(videos),holdout_evaluations=0,automatic_promotion=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['original','spatial'],required=True);args=ap.parse_args()
 try:main(args.stage)
 except Exception as e:save(P/('REPORT_'+args.stage+'_ERROR.json'),dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
