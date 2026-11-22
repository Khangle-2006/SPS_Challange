"""Integrate supplied b04 into the public cached-waveform pipeline.
Run on Linux, one CPU process by default; saves fits and asserts deployed-readout parity.
Does not open BTC or any raw video. All original source/results stay untouched.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import argparse,json,ast,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
import public_readout

def main():
 p=argparse.ArgumentParser();p.add_argument('--spectra',required=True,type=Path);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
 import fcntl
 args.out.mkdir(parents=True,exist_ok=True);lock=(args.out/'evaluate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (args.out/'COMPLETE.json').exists()
 # Load source without its file IO/main program, then bind the actual cached data.
 tree=ast.parse((P/'source/b04_public_final.py').read_text())
 keep=[]
 for n in tree.body:
  if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)):keep.append(n)
  elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['GW','SIG','TAU','PEN'] for t in n.targets):keep.append(n)
 with np.load(args.spectra,allow_pickle=False) as z:
  raw=np.stack([z[k] for k in ['pad','pos','chrom']]);bpm=z['bpm'].copy()
  ns=dict(BPM=bpm)
  for target,key in [('ds','dataset'),('cond','cond'),('person','person'),('fold','fold'),('recid','recording'),('start','start'),('truth','truth')]:ns[target]=z[key].copy()
 assert set(ns['ds'])=={'UBFC2','MCD'} and set(ns['fold'])==set(range(5))
 assert np.array_equal(bpm,public_readout.BPM) and np.isfinite(raw).all() and (raw>=0).all()
 log=np.log(raw.astype(float)+1e-30);sd=log.std(2,keepdims=True);assert (sd>1e-12).all(),'Degenerate public spectrum: inspect rather than report HR'
 ns['Z']=(log-log.mean(2,keepdims=True))/sd
 exec(compile(ast.Module(body=keep,type_ignores=[]),'supplied_b04_without_io','exec'),ns)
 results=[];fus=np.full(len(ns['truth']),np.nan);trk=fus.copy()
 for condition in ['native','c5prime_bf0','c5prime_default','c5']:
  for fold in range(5):
   _,_,r=ns['task']((condition,fold));cfg={}
   for ds in ['UBFC2','MCD']:
    tr=(ns['cond']==condition)&(ns['ds']==ds)&(ns['fold']!=fold);te=(ns['cond']==condition)&(ns['ds']==ds)&(ns['fold']==fold)
    assert not set(ns['person'][tr])&set(ns['person'][te])
    ii,pred=r['fus'][ds];fus[ii]=pred;ii,pred=r['trk'][ds];trk[ii]=pred
    for run in ns['runs_of'](ii):
     inp=raw[:,run].transpose(1,0,2)
     deployed=public_readout.decode(inp,r['w'],ns['start'][run],r['cfg'][ds])
     assert np.array_equal(deployed['predictions'],trk[run]),'Deployment differs from supplied b04'
    cfg[ds]=r['cfg'][ds]
   results.append(dict(condition=condition,fold=fold,weights=r['w'],tracking=cfg))
   (args.out/'STATUS.json').write_text(json.dumps(results[-1]))
 np.savez_compressed(args.out/'PREDICTIONS.npz',fusion=fus,tracked=trk,truth=ns['truth'],dataset=ns['ds'],condition=ns['cond'],person=ns['person'],fold=ns['fold'],recording=ns['recid'],start=ns['start'])
 (args.out/'FITS.json').write_text(json.dumps(results,indent=2))
 table=[]
 for c in sorted(set(ns['cond'])):
  for d in ['UBFC2','MCD']:
   s=(ns['cond']==c)&(ns['ds']==d);people=ns['person'][s];truth=ns['truth'][s]
   base=bpm[ns['Z'][0,s].argmax(1)];_,be=ns['pmae'](abs(base-truth),people)
   for name,pr in [('padding',base),('fusion',fus[s]),('fusion_tracking',trk[s])]:
    pp,errors=ns['pmae'](abs(pr-truth),people);ix=np.random.default_rng(0).integers(len(pp),size=(10000,len(pp)))
    table.append(dict(condition=c,dataset=d,method=name,people=len(pp),windows=int(s.sum()),MAE=float(errors.mean()),CI=np.quantile(errors[ix].mean(1),[.025,.975]).tolist(),delta_padding=float((errors-be).mean()),delta_CI=np.quantile((errors-be)[ix].mean(1),[.025,.975]).tolist()))
 (args.out/'METRICS.json').write_text(json.dumps(table,indent=2))
 (args.out/'COMPLETE.json').write_text(json.dumps(dict(passed=True,source_sha256=hashlib.sha256((P/'source/b04_public_final.py').read_bytes()).hexdigest(),spectra_sha256=hashlib.sha256(args.spectra.read_bytes()).hexdigest(),holdout_evaluations=0,offline_noncausal=True,source_parity=True)))
if __name__=='__main__':main()
