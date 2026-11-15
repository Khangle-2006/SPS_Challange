"""Native BTC CPU experiment: train-fold PBV, G, nested person-fold linear stack."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,fcntl,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import signal
P=Path(__file__).resolve().parent;B=P.parent;OLD=B.parent/'round1_v4_20261105/btc_zeroshot'
sys.path.insert(0,str(OLD));from infer_dev import classical_functions,decode_wave,supported
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,a):p.write_text(json.dumps(a,indent=2,allow_nan=False))
def main():
 lock=(P/'pbv_stack.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (P/'PBV_STACK_COMPLETE.json').exists(),'Do not rerun completed experiment'
 assert sha(OLD/'btc_split.json')=='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
 videos=read(OLD/'dev_manifest.json')['videos'];split=read(OLD/'btc_split.json');assert len(videos)==221 and all(split['people'][v['person']]['partition']=='dev' for v in videos)
 ns,sourcehash=classical_functions();windows=[];inputhash={};excluded=[]
 sos=signal.butter(1,[.7,3.],btype='bandpass',fs=30,output='sos')
 for v in videos:
  d=OLD/'dev'/v['drive_id'];src=read(d/'SOURCE.json');assert src['source_sha256']==v['sha256']
  if src['source_span_s']<10:excluded.append(dict(recording=v['recording'],reason='source_under_10s'));continue
  c=read(d/'native/COMPLETE.json');mp=d/'native/meta.npz';assert sha(mp)==c['meta_sha256'];inputhash[str(mp)]=c['meta_sha256']
  with np.load(mp) as m:t=m['t'];valid=m['valid'];rgb=m['rgb_mean']
  predfile=OLD/'predictions'/v['drive_id']/'native/classical.json';old=read(predfile)['rows'];inputhash[str(predfile)]=sha(predfile)
  starts=sorted({r['start'] for r in old if r['scheme']=='10s'})
  for start in starts:
   if not supported(t,valid,start,start+10):excluded.append(dict(recording=v['recording'],start=start,reason='frozen_image_support'));continue
   q=start+np.arange(300)/30;x=np.stack([np.interp(q,t,rgb[:,k]) for k in range(3)],1)
   try:
    if not np.isfinite(x).all() or np.min(abs(x.mean(0)))<1e-8:raise ValueError('RGB mean unsupported')
    cn=x/x.mean(0)-1;cn=signal.sosfiltfilt(sos,cn,axis=0);pos=ns['pos'](x,30)
    if not np.isfinite(pos).all() or np.std(pos)<1e-10:raise ValueError('flat/nonfinite POS cannot define blind signature frequency')
    f,power=signal.periodogram(pos,fs=30,window='hann',nfft=4096,detrend='linear');band=(f>=.7)&(f<=3.);peak=f[band][power[band].argmax()];hit=band&(abs(f-peak)<=.1)
    snr=10*np.log10(max(float(power[hit].sum()),1e-30)/max(float(power[band&~hit].sum()),1e-30))
    z=np.sum(cn*np.hanning(300)[:,None]*np.exp(-2j*np.pi*peak*np.arange(300)/30)[:,None],axis=0)
    if abs(z[1])<1e-12:raise ValueError('G phase unsupported')
    vector=(z*np.exp(-1j*np.angle(z[1]))).real;vector/=np.linalg.norm(vector)
    green=decode_wave(cn[:,1]);pair={method:next(r['prediction'] for r in old if r['scheme']=='10s' and r['start']==start and r['method']==method) for method in ['POS','CHROM']}
    windows.append(dict(video=v,start=start,cn=cn,signature=vector,snr=float(snr),G=green,**pair))
   except ValueError as e:excluded.append(dict(recording=v['recording'],start=start,reason=str(e)))
 fits={};predictions={}
 def fitted(exclude):
  key=tuple(sorted(exclude))
  if key in fits:return np.array(fits[key]['signature'])
  train=[w for w in windows if w['video']['fold'] not in exclude];threshold=float(np.quantile([w['snr'] for w in train],.75));top=[w for w in train if w['snr']>=threshold];per=defaultdict(list)
  for w in top:per[w['video']['person']].append(w['signature'])
  assert len(per)>=3,'Insufficient training people for PBV signature'
  s=np.mean([np.median(v,axis=0) for v in per.values()],axis=0);assert np.linalg.norm(s)>1e-10;s/=np.linalg.norm(s)
  people=sorted({w['video']['person'] for w in train});assert set(people).isdisjoint(v['person'] for v in videos if v['fold'] in exclude)
  fits[key]=dict(excluded_folds=list(key),signature=s.tolist(),top_SNR_threshold=threshold,training_people=people,signature_people=sorted(per),top_windows=len(top));return s
 def predict(exclude,fold):
  key=tuple(sorted(exclude)),fold
  if key in predictions:return predictions[key]
  s=fitted(exclude);per=defaultdict(list)
  for w in windows:
   if w['video']['fold']!=fold:continue
   cn=w['cn'];cov=cn.T@cn/len(cn);cov+=np.eye(3)*max(float(np.trace(cov)),1e-12)*1e-6
   weights=np.linalg.solve(cov,s);wave=cn@weights
   try:hr=decode_wave(wave)
   except ValueError:continue
   per[w['video']['recording']].append(dict(PBV=hr,**{k:w[k] for k in ['G','snr','POS','CHROM']}))
  result=[]
  for rec,rr in per.items():
   v=next(v for v in videos if v['recording']==rec);r={k:v[k] for k in ['recording','person','fold','group','task','skin']};r['truth']=v['pr_bpm'];r['windows']=len(rr)
   for k in ['PBV','G','snr','POS','CHROM']:r[k]=float(np.median([x[k] for x in rr if x[k] is not None])) if any(x[k] is not None for x in rr) else None
   result.append(r)
  predictions[key]=result;return result
 out=[];stacks=[]
 for outer in range(5):
  val=predict({outer},outer);train=[]
  for inner in range(5):
   if inner!=outer:train+=predict({outer,inner},inner)
  keys=['POS','CHROM','PBV','snr'];train=[r for r in train if all(r[k] is not None for k in keys)];val=[r for r in val if all(r[k] is not None for k in keys)]
  assert set(r['person'] for r in train).isdisjoint(r['person'] for r in val)
  x=np.array([[r[k] for k in keys] for r in train]);y=np.array([r['truth'] for r in train]);count={p:sum(r['person']==p for r in train) for p in {r['person'] for r in train}};weight=np.array([1/count[r['person']] for r in train]);mu=np.average(x,axis=0,weights=weight);sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=weight));sd=np.maximum(sd,1e-8)
  design=np.c_[np.ones(len(x)),(x-mu)/sd];coef=np.linalg.lstsq(design*np.sqrt(weight[:,None]),y*np.sqrt(weight),rcond=1e-10)[0];assert len(coef)==5
  stacks.append(dict(fold=outer,features=keys,parameters_including_intercept=5,coef=coef.tolist(),mean=mu.tolist(),std=sd.tolist(),training_people=sorted(count),PBV_crossfit_excludes_outer_and_inner=True))
  constant=float(np.median([v['pr_bpm'] for v in videos if v['fold']!=outer]))
  for r in val:
   stack=float(np.r_[1.,(np.array([r[k] for k in keys])-mu)/sd]@coef)
   for method,pred in [('constant',constant),('POS',r['POS']),('CHROM',r['CHROM']),('G',r['G']),('PBV_trainfold',r['PBV']),('stack5_nested',stack)]:out.append(dict(**{k:r[k] for k in ['recording','person','fold','group','task','skin','truth','windows']},method=method,prediction=pred,condition='native'))
 with (P/'PBV_STACK_VIDEOS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 save(P/'PBV_SIGNATURES.json',list(fits.values()));save(P/'STACK_FITS.json',stacks);save(P/'PBV_EXCLUSIONS.json',excluded)
 # No change to selection gates and no BTC fine-tuning from these CPU results.
 save(P/'PBV_STACK_COMPLETE.json',dict(rows=len(out),videos=len({r['recording'] for r in out}),people=len({r['person'] for r in out}),input_hashes=inputhash,vendor_hashes=sourcehash,holdout_evaluations=0,method='team PBV signature estimator from top-quartile blind SNR; person-balanced signatures; nested linear5'))
 print('PBV/G/nested stack completed',len(out),'rows',flush=True)
if __name__=='__main__':main()
