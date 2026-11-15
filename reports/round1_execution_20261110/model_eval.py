"""OOF inference against frozen candidate clocks, including explicit failures."""
import numpy as np
from common import *

def c5_rows(model,fold,device,method,btc_only=False):
 import torch
 from t8_t1_loader import supported
 from t8_evaluate import decode_difference
 data=BTCData(fold);meta={v['recording']:v for v in data.rows}
 refs=read(B/'public_comparison/REFERENCE_COMPACT.json')
 original=R/'reports/round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json'
 assert sha(original)==refs['source_sha256']
 model.eval();rows=[]
 for cp in sorted((P/'c5prime').glob('*/*/*/COMPLETE.json')):
  a=read(cp);ds=a['dataset'];rid=a['recording']
  if a['fold']!=fold or (btc_only and ds!='BTC'):continue
  if a['state']!='complete':continue # retained in C5_COMPLETE exclusion manifest
  d=cp.parent/'crop';c=read(d/'COMPLETE.json')
  assert sha(d/'crop.mkv')==c['crop_sha256'] and sha(d/'meta.npz')==c['meta_sha256']
  with np.load(d/'meta.npz') as z:t=z['t'].copy();valid=z['valid'].copy()
  raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(d/'crop.mkv'),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
  assert len(raw)==len(t)*3888
  rgb=np.frombuffer(raw,np.uint8).reshape(-1,36,36,3)
  if ds=='BTC':
   v=meta[rid];windows=[dict(start=s,truth=v['pr_bpm'],eligible=True) for s in sorted({w['start'] for w in data.candidates(v,'native') if w['scheme']=='10s'})];extra={k:v[k] for k in ['group','task','skin']}
  else:
   r=next(r for r in refs['records'] if r['dataset']==ds and r['recording_id']==rid and r['variant']=='native')
   windows=[dict(start=w['window_start_s'],truth=w['reference_hr_bpm'],eligible=w['reference_eligible']) for w in r['windows']];extra={}
  for w in windows:
   pred=None;reason=None
   try:
    if not w['eligible']:raise ValueError('reference_ineligible')
    if not a['bitrate_pass']:raise ValueError('bitrate_above_500_no_retuning')
    q=w['start']+np.arange(301)/30
    if not supported(t,valid,q[0],q[-1]):raise ValueError('image_support_gap_or_tracking_failure')
    j=np.searchsorted(t,q,side='right').clip(1,len(t)-1);weight=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
    x=rgb[j-1].astype(np.float32)*(1-weight[:,None,None,None])+rgb[j].astype(np.float32)*weight[:,None,None,None]
    if not np.isfinite(x).all() or x.std()<1e-6:raise ValueError('constant_or_nonfinite_input')
    x=((x-x.mean())/x.std()).transpose(0,3,1,2).copy()
    with torch.inference_mode():wave=model(torch.from_numpy(x[None]).to(device))[0].cpu().numpy()
    pred=decode_difference(wave)
   except ValueError as e:reason=str(e)
   rows.append(dict(dataset=ds,recording=rid,person=a['person'],fold=fold,condition='c5prime_'+a['variant'],method=method,**w,**extra,prediction=pred,reason=reason,receipt_sha256=sha(cp)))
 return rows

def weak_gate(before,after):
 from scipy.stats import spearmanr
 def videos(rr):
  return {rid:dict(person=a[0]['person'],truth=a[0]['truth'],prediction=float(np.median([r['prediction'] for r in a]))) for rid in {r['recording'] for r in rr} if (a:=[r for r in rr if r['recording']==rid and r['prediction'] is not None])}
 a,b=videos(before),videos(after);keys=sorted(set(a)&set(b));people=sorted({a[k]['person'] for k in keys})
 def rho(x):
  pred=[np.mean([x[k]['prediction'] for k in keys if x[k]['person']==p]) for p in people];truth=[np.mean([a[k]['truth'] for k in keys if a[k]['person']==p]) for p in people]
  r=float(spearmanr(pred,truth).statistic) if len(people)>2 and np.std(pred)>0 and np.std(truth)>0 else float('nan')
  return r if np.isfinite(r) else None
 rb,ra=rho(a),rho(b);sd=float(np.std([b[k]['prediction'] for k in keys],ddof=1)) if len(keys)>1 else None
 return dict(videos=len(keys),people=len(people),SD_video_bpm=sd,Spearman_before=rb,Spearman_after=ra,passed=bool(sd is not None and sd>=8 and rb is not None and ra is not None and ra>rb),policy='Reject weak-label run on failure, retain measured results; no threshold tuning')
