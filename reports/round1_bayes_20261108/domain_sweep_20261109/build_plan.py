"""CPU-only fold-safe plan. No training, GPU initialization or holdout paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,math
from pathlib import Path
import numpy as np
from codec_provider import Provider,read,sha,ROOT,V3
from t8_plan import pools
P=Path(__file__).resolve().parent;OLD=P.parent.parent/'round1_v4_20261105';BTC=OLD/'btc_zeroshot'
SPLIT='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
INDEX=ROOT/'t8_oval_20261031/training_index.json'
assert sha(BTC/'btc_split.json')==SPLIT
index=read(INDEX);dev=read(BTC/'dev_manifest.json');split=read(BTC/'btc_split.json')
assert len(dev['videos'])==221 and dev['split_sha256']==SPLIT
assert all(split['people'][v['person']]['partition']=='dev' for v in dev['videos'])
plan=dict(index_sha256=sha(INDEX),btc_manifest_sha256=sha(BTC/'dev_manifest.json'),split_sha256=SPLIT,seed=2027,epochs=10,batch_size=8,btc_aux_batch_size=4,learning_rate=.0001,weight_decay=0.,optimizer='AdamW',scheduler=None,selection='final_epoch_10_no_validation_selection',early_gate='prior pilot gate passed; new fine-tune epoch5 trend report only; nonfinite/static failures stop',memory_fraction=.35,threads=2,codec_cache_bytes=8_000_000_000,btc_cache_bytes=1_000_000_000,output_reserve_bytes=160_000_000,deadline_utc='2026-11-22T16:59:59+00:00',official_MATLAB_parity=False,btc_aux_weight={"2":.1,"3":1.,"4":1.},runs=[])
for fold in range(5):
 train=[v for v in dev['videos'] if v['fold']!=fold];val=[v for v in dev['videos'] if v['fold']==fold]
 trainpeople=sorted({v['person'] for v in train});valpeople=sorted({v['person'] for v in val});assert set(trainpeople).isdisjoint(valpeople)
 bpp=[]
 for v in train:
  m=v['media'];kbps=float(m['video_kbps'] or m['mux_kbps']);bpp.append(kbps*1000/(float(m['width'])*float(m['height'])*float(m['avg_fps'])))
 bounds=np.quantile(bpp,[.25,.75]).tolist();provider=Provider(index,bounds);pool=pools(index,fold);jobs={};excluded=[]
 for ds,rr in pool.items():
  jobs[ds]=[]
  for rid,start in rr:
   rng=np.random.default_rng(int(hashlib.sha256(f'2027/{fold}/{ds}/{rid}/{start}'.encode()).hexdigest()[:12],16))
   recipe=['h264','hevc','c5_proxy'][int(rng.integers(3))];lowres=bool(rng.integers(2));rates=provider.feasible_rate(ds,rid,start,recipe)
   job=dict(dataset=ds,recording=rid,start=float(start),recipe=recipe,lowres=lowres,rates=rates)
   if rates is None:excluded.append(dict(**job,reason='No continuous valid image/PPG support for minimum 8s query; original 200ms gap contract retained'))
   else:jobs[ds].append(job)
  assert jobs[ds]
 initial=OLD/('v4_padding_pilot' if fold==0 else 'v4_repaired')/f'fold{fold}/seed2027/joint'
 status=read(initial/'status.json');prov=read(initial/'provenance.json');history=read(initial/'history.json')
 assert status['state']=='complete' and status['completed_epochs']==30 and [h['epoch'] for h in history]==list(range(1,31))
 assert prov['config']['fold']==fold and prov['config']['seed']==2027 and prov['index_sha256']==sha(INDEX)
 assert prov['train_pool_sha256']==hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()
 assert sha(initial/'final.pt')==status['checkpoint_sha256']
 plan['runs'].append(dict(fold=fold,checkpoint=str(initial/'final.pt'),checkpoint_sha256=status['checkpoint_sha256'],checkpoint_provenance_sha256=sha(initial/'provenance.json'),initial_padding='trained_replicate' if fold==0 else 'same_fold_repaired_zero_padding_weights_converted_to_replicate',btc_train_people=trainpeople,btc_val_people=valpeople,btc_train_records=[v['recording'] for v in train],btc_val_records=[v['recording'] for v in val],btc_bpp_quartiles=bounds,public_jobs=jobs,public_exclusions=excluded,steps_per_epoch=math.ceil(sum(map(len,jobs.values()))/8),btc_training='weak_CE' if fold==2 else 'SiNC' if fold in (3,4) else 'none'))
 del provider
dest=P/'PLAN.json';assert not dest.exists(),'Never overwrite a frozen or inspected plan'
dest.write_text(json.dumps(plan,indent=2,allow_nan=False))
print(json.dumps([dict(fold=r['fold'],jobs={k:len(v) for k,v in r['public_jobs'].items()},excluded=len(r['public_exclusions']),steps=r['steps_per_epoch'],bpp=r['btc_bpp_quartiles']) for r in plan['runs']],indent=2))
