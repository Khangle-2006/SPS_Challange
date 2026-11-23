"""Replay frozen BTC native and public R1 readers on authorized dev caches. No fit, GPU or raw video reads."""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
import numpy as np
from common import load_model,sha
from reader_v4 import evidence,predG,MODELS

def pmae(p,y,people):return float(np.mean([np.abs(p[people==x]-y[people==x]).mean() for x in np.unique(people)]))
def runs(rec,start):
 for name in dict.fromkeys(rec.tolist()):
  ix=np.flatnonzero(rec==name);ix=ix[np.argsort(start[ix])];cuts=np.flatnonzero(np.abs(np.diff(start[ix])-10)>=.6)+1
  yield from np.split(ix,cuts)
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--server-root',type=Path,default=Path('/mnt/samsung_t5/khangle/facevitals'));a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 conf=json.loads((ROOT/'configs/server-replay-r1.json').read_text());pack=args.server_root/conf['server_relative_path'];mf=pack/'MANIFEST.json'
 if not mf.exists():raise SystemExit('Private dev replay cache missing; see docs/REPRODUCE_SERVER.md. No dataset scan or download is performed.')
 if sha(mf)!=conf['manifest_sha256']:raise RuntimeError('Dev cache manifest changed')
 m=json.loads(mf.read_text());assert m['scope']=='dev-only' and m['btc_videos']==218 and m['btc_people']==158
 for row in m['files']:
  rel=Path(row['path'])
  if rel.name!=row['path'] or rel.suffix!='.npz':raise RuntimeError('Unexpected dev cache path')
  if sha(pack/rel)!=row['sha256']:raise RuntimeError('Dev cache hash mismatch: '+str(rel))
 args.output.mkdir(parents=True,exist_ok=False);(args.output/'.gitignore').write_text('*\n')
 rows=[]
 with np.load(pack/'btc_native.npz',allow_pickle=False) as d:
  Z=evidence(d['C'],d['P'],{n:d[n] for n in MODELS});pred=np.full(218,np.nan)
  for f in range(5):
   sel=np.flatnonzero(d['fold']==f);model,_=load_model(f);pred[sel]=predG(model,Z[sel],d['low'][sel],d['low'][sel],d['task'][sel])
  delta=float(np.max(abs(pred-d['reference'])));assert delta<=1e-9,('BTC parity failed',delta)
  rows.append(dict(dataset='BTC',condition='native',samples=len(pred),people=len(np.unique(d['person'])),mae=pmae(pred,d['truth'],d['person']),max_prediction_delta=delta))
 assets=ROOT/'checkpoints/r1/facevitals-inference-assets-r1-20261124';sys.path.insert(0,str(assets/'code'))
 from public_fixed_math import qscale,marg,point,BPM
 from importlib.util import spec_from_file_location,module_from_spec
 sp=spec_from_file_location('asset_check',ROOT/'tools/check_inference_assets.py');checker=module_from_spec(sp);sp.loader.exec_module(checker);checker.verify(assets)
 pub=ROOT/'models/public_p4m_r1.json';pm=json.loads((ROOT/'models/PUBLIC_MANIFEST.json').read_text());assert sha(pub)==pm['sha256'][pub.name]
 fits=json.loads(pub.read_text())
 with np.load(pack/'public_native.npz',allow_pickle=False) as native:
  native_keys={(str(rec),round(float(start),3)) for rec,start in zip(native['rec'],native['start'])}
 with (ROOT/'reports/rerun_stateless_20261123/z13_public_rerun.csv').open() as f:
  expected={(row['condition'],row['dataset']):float(row['P4m']) for row in csv.DictReader(f)}
 for row in m['files']:
  if not row['path'].startswith('public_'):continue
  cond=row['path'][7:-4]
  with np.load(pack/row['path'],allow_pickle=False) as d:
   pred=np.full(len(d['truth']),np.nan)
   for f in range(5):
    for ds in ['UBFC2','MCD']:
     ix=np.flatnonzero((d['fold']==f)&(d['ds']==ds))
     if not len(ix):continue
     g,s,tau,est,ref,_=fits[f'ro_fold{f}_{ds}'];w=np.array(fits[f'w_fold{f}'])
     for sub in runs(d['rec'][ix],d['start'][ix]):
      jj=ix[sub];E=np.tensordot(w,d['B'][:,jj],1);pred[jj]=point(marg(qscale(E,g,ref),s,tau,np.zeros(len(BPM))),'median' if est==1 else 'mean')
   assert np.isfinite(pred).all();delta=float(np.max(abs(pred-d['reference'])));assert delta<=1e-9,(cond,'Public parity failed',delta)
   for ds in ['UBFC2','MCD']:
    matched=np.array([(str(rec),round(float(start),3)) in native_keys for rec,start in zip(d['rec'],d['start'])])
    sel=(d['ds']==ds)&matched
    if sel.any():
     mae=pmae(pred[sel],d['truth'][sel],d['person'][sel]);assert abs(mae-expected[(cond,ds)])<=1e-9,(cond,ds,'Matched-cohort metric mismatch')
     rows.append(dict(dataset=ds,condition=cond,samples=int(sel.sum()),people=len(np.unique(d['person'][sel])),mae=mae,max_prediction_delta=delta))
 for row in rows:print(f"{row['dataset']:5} {row['condition']:24} n={row['samples']:5} people={row['people']:3} MAE={row['mae']:.9f} parity={row['max_prediction_delta']:.3g}",flush=True)
 with (args.output/'RESULTS.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 receipt={'rows':rows,'holdout_reads':0,'refit':False,'raw_video_inference':False,'scope':'BTC native reader and public all-condition readout replay from frozen pre-reader evidence; not raw frontend/CNN rerun','manifest_sha256':sha(mf)}
 (args.output/'REPLAY_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
