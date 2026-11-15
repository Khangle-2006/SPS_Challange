"""Frozen per-fold codec jobs; excluded constraint failures are recorded, never retuned."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,hashlib,traceback,time
import numpy as np
from runtime import *
from provider import SideProvider
os.environ['CUDA_VISIBLE_DEVICES']=''
def main():
    lease=lock(P/'aug.lock');assert not (P/'AUG_STARTED.json').exists(),'Inspect existing launch/partials'
    assert read(P/'PREP_COMPLETE.json')['index_sha256']==sha(P/'SIDE_INDEX.json');budget(fresh=True)
    save(P/'AUG_STARTED.json',dict(utc=now(),pid=os.getpid(),code_sha256=sha(__file__)))
    prior=read(E/'PLAN_ELIGIBLE.json');index=read(old.p.INDEX);side=read(P/'SIDE_INDEX.json');runs=[];excluded=[]
    for f in range(5):
        base=prior['runs'][f];provider=SideProvider(base['btc_bpp_quartiles']);jobs=[];spec=index['folds']['folds'][f]
        train=set(spec['mcd_train_subjects']);val=set(spec['mcd_val_subjects'])
        for record in side:
            if record['subject_id'] not in train:assert record['subject_id'] in val;continue
            assert record['fold']!=f
            for start in record['windows']:
                key=[2027,f,record['recording_id'],start];rng=np.random.default_rng(int(hashlib.sha256(repr(key).encode()).hexdigest()[:12],16));recipe=['hevc','h264','c5_proxy'][int(rng.integers(3))];lowres=bool(rng.integers(2))
                rates=provider.feasible_rate('MCD_SIDE',record['recording_id'],start,recipe)
                j=dict(recording=record['recording_id'],person=record['subject_id'],camera=record['camera'],start=start,recipe=recipe,lowres=lowres,rates=rates)
                if rates is None:excluded.append(dict(fold=f,**j,reason='no_valid_resample_support'));continue
                jobs.append(j)
        assert all(any(j['camera']==v for j in jobs) for v in ['IriunWebcam','USBVideo']),'A fold lacks usable sideview domain; inspect, do not change sampling silently'
        assert all(j['person'] not in val for j in jobs)
        run=dict(fold=f,public_jobs=base['public_jobs'],side_jobs=jobs,btc_bpp_quartiles=base['btc_bpp_quartiles'],btc_train_people=base['btc_train_people'],btc_val_people=base['btc_val_people'],steps_per_epoch=int(np.ceil(old.p.steps_per_epoch(old.p.pools(index,f))/4)))
        runs.append(run)
    save(P/'AUG_JOBS_FROZEN.json',dict(runs=runs,excluded=excluded))
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import threading
    local=threading.local()
    def encode(f,j):
        if not hasattr(local,'providers'):local.providers={}
        if f not in local.providers:local.providers[f]=SideProvider(runs[f]['btc_bpp_quartiles'])
        provider=local.providers[f]
        try:
            provider.prepare('MCD_SIDE',j['recording'],j['start'],j['recipe'],j['lowres'],f)
            for rate in j['rates']:
                x,y,enc=provider.clip('MCD_SIDE',j['recording'],j['start'],j['recipe'],j['lowres'],f,rate);assert x.shape==(301,3,36,36) and y.shape==(300,) and np.isfinite(x).all() and np.isfinite(y).all()
            # Disk is authoritative; don't retain multiple 300MB RAM caches per worker.
            provider.cache.clear();provider.bytes=0
            return f,j,None
        except RuntimeError as e:
            if not str(e).startswith('C5 proxy bitrate failed'):raise
            return f,j,str(e)
    accepted={f:[] for f in range(5)};done=0;total=sum(len(r['side_jobs']) for r in runs)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(encode,r['fold'],j) for r in runs for j in r['side_jobs']]
        for future in as_completed(futures):
            f,j,reason=future.result();done+=1
            if reason:excluded.append(dict(fold=f,**j,reason=reason))
            else:accepted[f].append(j)
            save(P/'AUG_STATUS.json',dict(pid=os.getpid(),completed=done,total=total,exclusions=len(excluded),fold=f,recording=j['recording']))
    for r in runs:
        r['side_jobs']=sorted(accepted[r['fold']],key=lambda j:(j['recording'],j['start']))
        assert all(any(j['camera']==v for j in r['side_jobs']) for v in ['IriunWebcam','USBVideo']),'A fold lacks a sideview after constraint checks'
    plan=dict(seed=2027,epochs=30,batch_size=8,lr=.009,weight_decay=0.,scheduler='OneCycleLR',initialization='same-fold seed2027 random; no trained cross-fold weights',loss='unchanged padding pilot repair_loss.phase_loss',sampling='4 UBFC +4 MCD per batch; MCD view chosen uniformly among FullHD/Iriun/USB then uniform eligible encoded window within view',resample_range=[.8,1.2],runs=runs,excluded=excluded,offsets_sha256=sha(P/'OFFSETS_V2.json'),side_index_sha256=sha(P/'SIDE_INDEX.json'),source_public_plan_sha256=sha(E/'PLAN_ELIGIBLE.json'),selection='final_epoch30',static_each_epoch=True,no_training_after_V5=True,holdout_evaluations=0)
    save(P/'TRAIN_PLAN.json',plan);save(P/'AUG_COMPLETE.json',dict(utc=now(),plan_sha256=sha(P/'TRAIN_PLAN.json'),side_jobs=sum(len(r['side_jobs']) for r in runs),excluded=len(excluded),budget=budget()))
if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'AUG_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
