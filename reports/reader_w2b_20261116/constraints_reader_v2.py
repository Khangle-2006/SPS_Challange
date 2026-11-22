import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe')
import sys,json,fcntl,tempfile,traceback,hashlib,time,shutil
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;V=R/'round1_v4_20261105/btc_zeroshot';W=R/'w2_btc_20261116';Q=R/'reader_v1_20261116'
sys.path[:0]=[str(V),str(Q)]
import preprocess as prep
import simulate
import reader_v1 as api
from gated_reader import evaluate
def save(p,x):prep.save(p,x)
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def guarded_frames(src,ram):
 for frame in src:
  assert size(ram)<380_000_000,'STOP_SHARED_RAM_HEADROOM'
  yield frame
def main():
 shard=int(sys.argv[1]);count=4;lock=(P/f'constraints_shard{shard}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.load(open(P/'PREFLIGHT.json'))['passed']
 ids=set(json.load(open(W/'PRIVATE_EXPORT_MAPPING.json'))['video']);dev=[v for v in prep.read(V/'dev_manifest.json')['videos'] if v['recording'] in ids];assert len(dev)==218
 conditions=[('c1_050','c1',.5,None),('c1_025','c1',.25,None),('c1_0125','c1',.125,None),('c4_QVGA','c4',None,(320,240)),('c4_VGA','c4',None,(640,480))]
 assert simulate.PROTOCOL['c1']['gain_levels']==[.5,.25,.125]
 if shard==0:save(P/'C_PROTOCOL_V2.json',dict(source_simulate_sha256=prep.sha256(simulate.__file__),source_protocol_sha256=prep.sha256(simulate.HERE/'protocol.json'),conditions=conditions,frontend='frozen OvalCrop on transformed full frames; no native ROI fallback',clock='source timestamp per encoded frame, not CFR mux clock',temporary_storage='/dev/shm managed ephemeral400MB, existing files untouched',C1_limitation=simulate.PROTOCOL['c1']['limitation']))
 for position,v in enumerate(dev):
  if position%count!=shard:continue
  assert prep.sha256(v['path'])==v['sha256']
  for name,kind,gain,sz in conditions:
   d=P/'constraints'/name/v['recording'];d.mkdir(parents=True,exist_ok=True)
   if (d/'COMPLETE.json').exists():continue
   assert not (d/'meta.npz').exists(),'Inspect partial; no blind restart'
   assert size(P)<200_000_000,'STOP_FOLLOWUP_200MB'
   save(P/f'C_STATUS_SHARD{shard}.json',dict(video=v['recording'],condition=name,pid=os.getpid()))
   # All existing data remain immutable; only this newly-created RAM scratch is ephemeral.
   ram=Path('/dev/shm/reader_C_parallel_20261116');ram.mkdir(exist_ok=True)
   c4lease=(P/'C4_ram.lock').open('a')
   if kind=='c4':fcntl.flock(c4lease,fcntl.LOCK_EX)
   tmp=Path(tempfile.mkdtemp(prefix=f'shard{shard}_',dir=ram))
   try:
    src=prep.Video(v['path'],d/'source_decode.log');video,receipt=simulate.simulate(guarded_frames(src,ram),src.t,tmp/'encoded',kind,size=sz,gain=gain,seed=simulate.PROTOCOL['seed'],budget_root=ram,budget_bytes=400_000_000)
    assert size(tmp)<=400_000_000,'STOP_RAM400MB';encoded=prep.Video(video,d/'derived_decode.log');assert len(encoded.t)==len(src.t)
    detector=prep.OvalCrop();valid=[];means=[];boxes=[]
    try:
     for frame in encoded:
      ok,small,box,_=detector.step(frame);valid.append(bool(ok));means.append(small.astype(float).mean((0,1)));boxes.append(box)
    finally:detector.close()
    assert len(valid)==len(src.t);np.savez_compressed(d/'meta.npz',t=src.t,valid=valid,rgb_mean=means,boxes=boxes)
    save(d/'ENCODING.json',dict(receipt,ephemeral_video_sha256=prep.sha256(video),source_sha256=v['sha256']))
   except Exception:
    save(d/'PRESERVED_RAM_PARTIAL.json',dict(path=str(tmp)));raise
   else:
    assert tmp.resolve().is_relative_to(ram.resolve());shutil.rmtree(tmp)
   finally:c4lease.close()
   z=api.scores(src.t,np.array(valid),np.array(means));save(d/'COMPLETE.json',dict(meta_sha256=prep.sha256(d/'meta.npz'),frames=len(valid),valid_frames=sum(valid),signal_abstain=z is None,z=None if z is None else z.tolist()))
 save(P/f'C_SHARD{shard}_COMPLETE.json',dict(passed=True))
 if shard!=0:return
 while not all((P/f'C_SHARD{i}_COMPLETE.json').exists() for i in range(count)):
  if any((P/f'C_ERROR_SHARD{i}.json').exists() for i in range(count)):return
  time.sleep(30)
 results=[]
 for name,kind,gain,sz in conditions:
  records=[];zs=[]
  for v in dev:
   d=P/'constraints'/name/v['recording'];a=prep.read(d/'COMPLETE.json');assert prep.sha256(d/'meta.npz')==a['meta_sha256'];zs.append(None if a['z'] is None else np.array(a['z']));records.append(dict(recording=v['recording'],person=v['person'],fold=v['fold'],group=v['group'],task=v['task'],truth=float(v['pr_bpm'])))
  r=evaluate(name,records,zs);r['coverage']=dict(available_videos=218,total_videos=218,signal_videos=sum(z is not None for z in zs),prior_only_videos=sum(z is None for z in zs));results.append(r)
 save(P/'C_RESULTS.json',results);lines=['# C1/C4 reader with fixed train-gain gate','Full-frame transforms use frozen simulate.py including C1 storage encoder. Synthetic gain is not measured lux; C1 also includes compression. Source clock retained. No holdout.','', '|Condition|Group|Method|Videos/people|MAE|Bias|Spearman|Delta constant CI95|Delta native reader CI95|','|---|---|---|---:|---:|---:|---:|---|---|']
 for a in results:
  for r in a['metrics']:lines.append(f"|{a['condition']}|{r['group']}|{r['method']}|{r['videos']}/{r['people']}|{r['MAE']:.4f}|{r['bias']:.4f}|{r['Spearman']}|{r['delta_constant']:.4f} {r['delta_constant_CI']}|{r['delta_native']:.4f} {r['delta_native_CI']}|")
 (P/'C_REPORT.md').write_text('\n'.join(lines));save(P/'C_COMPLETE.json',dict(videos=218,conditions=5,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception:save(P/f'C_ERROR_SHARD{sys.argv[1]}.json',dict(traceback=traceback.format_exc()));raise

