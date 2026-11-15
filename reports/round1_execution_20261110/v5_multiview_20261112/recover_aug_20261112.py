"""Audited recovery of one missing immutable codec job; no training restart."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,hashlib,traceback,shutil
import numpy as np
from runtime import *
from provider import SideProvider

def main():
    leases=[lock(P/'aug.lock'),lock(P/'ready.lock'),lock(P/'aug_recovery.lock')]
    assert not (P/'AUG_RECOVERY_STARTED.json').exists()
    frozen=read(P/'FROZEN.json')
    assert all(sha(k)==v for k,v in frozen.items())
    err=read(P/'AUG_ERROR.json')
    assert "du" in err['error'] and "non-zero exit status 1" in err['error']
    log=(P/'prepare_aug.log').read_text()
    assert "AUG_STATUS.json.tmp': No such file or directory" in log
    assert read(P/'READY_ERROR.json')['error']=='Augmentation failed, no automatic continuation'
    source=read(P/'AUG_JOBS_FROZEN.json');assert not source['excluded']
    early=read(P/'TRAIN_PLAN_EARLY.json');assert early['runs']==source['runs']
    jobs=[];missing=[];existing={}
    for run in source['runs']:
        f=run['fold'];existing[f]={}
        for j in run['side_jobs']:
            key=json.dumps(['MCD_SIDE',j['recording'],float(j['start']),j['recipe'],bool(j['lowres']),f],separators=(',',':'))
            rp=P/'side_aug'/str(f)/(hashlib.sha256(key.encode()).hexdigest()+'.json');file=rp.with_suffix('.npz')
            jobs.append((f,j,key,rp,file))
            if rp.exists():
                r=read(rp);assert r['key']==key and sha(file)==r['sha256'];existing[f][str(rp)]=sha(rp)
            else:
                assert not file.exists(),'Partial file: stop, preserve';missing.append((f,j))
    assert len(jobs)==672 and len(missing)==1 and missing[0][0]==4
    for f in range(4):
        r=read(P/f'FOLD{f}_READY.json');assert r['cache_receipt_hashes']==existing[f]
    fresh=budget(fresh=True)
    archive=P/'AUG_ATTEMPT1';archive.mkdir(exist_ok=False)
    for name in ['AUG_STARTED.json','AUG_STATUS.json','AUG_ERROR.json','READY_STARTED.json','READY_STATUS.json','READY_ERROR.json']:
        shutil.copy2(P/name,archive/name)
    save(P/'AUG_RECOVERY_STARTED.json',dict(utc=now(),pid=os.getpid(),code_sha256=sha(__file__),budget=fresh,existing_verified=671,missing=missing,frozen_training_unchanged=True))
    for f,j in missing:
        provider=SideProvider(source['runs'][f]['btc_bpp_quartiles'])
        provider.prepare('MCD_SIDE',j['recording'],j['start'],j['recipe'],j['lowres'],f)
        for rate in j['rates']:
            x,y,_=provider.clip('MCD_SIDE',j['recording'],j['start'],j['recipe'],j['lowres'],f,rate)
            assert x.shape==(301,3,36,36) and y.shape==(300,) and np.isfinite(x).all() and np.isfinite(y).all()
        provider.cache.clear();provider.bytes=0
    bindings={f:{} for f in range(5)}
    for f,j,key,rp,file in jobs:
        r=read(rp);assert r['key']==key and sha(file)==r['sha256'];bindings[f][str(rp)]=sha(rp)
    for f in range(4):assert read(P/f'FOLD{f}_READY.json')['cache_receipt_hashes']==bindings[f]
    plan=dict(seed=2027,epochs=30,batch_size=8,lr=.009,weight_decay=0.,scheduler='OneCycleLR',initialization='same-fold seed2027 random; no trained cross-fold weights',loss='unchanged padding pilot repair_loss.phase_loss',sampling='4 UBFC +4 MCD per batch; MCD view chosen uniformly among FullHD/Iriun/USB then uniform eligible encoded window within view',resample_range=[.8,1.2],runs=source['runs'],excluded=source['excluded'],offsets_sha256=sha(P/'OFFSETS_V2.json'),side_index_sha256=sha(P/'SIDE_INDEX.json'),source_public_plan_sha256=sha(E/'PLAN_ELIGIBLE.json'),selection='final_epoch30',static_each_epoch=True,no_training_after_V5=True,holdout_evaluations=0)
    final_budget=budget(fresh=True)
    assert all(sha(k)==v for k,v in frozen.items())
    save(P/'TRAIN_PLAN.json',plan)
    save(P/'FOLD4_READY.json',dict(utc=now(),run=source['runs'][4],cache_receipt_hashes=bindings[4],source_jobs_sha256=sha(P/'AUG_JOBS_FROZEN.json'),all_side_cache_hashes_verified=True,excluded_jobs=0))
    save(P/'AUG_COMPLETE.json',dict(utc=now(),plan_sha256=sha(P/'TRAIN_PLAN.json'),side_jobs=672,excluded=0,budget=final_budget,recovery_code_sha256=sha(__file__)))
    save(P/'READY_COMPLETE.json',dict(utc=now(),folds=list(range(5)),recovery=True))
    for name in ['AUG_ERROR.json','READY_ERROR.json']:
        assert sha(P/name)==sha(archive/name)
        (P/name).rename(archive/('original_'+name))
    save(P/'AUG_RECOVERY_COMPLETE.json',dict(utc=now(),pid=os.getpid(),reused_hash_verified=671,new_encoded=1,all_hash_verified=672,training_bindings_unchanged=True,recipe_unchanged=True,holdout_evaluations=0))
if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'AUG_RECOVERY_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
