"""Read-only cache finalizer: release each complete fold without waiting for later folds."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import time,json,hashlib,traceback
from runtime import *
os.environ['CUDA_VISIBLE_DEVICES']=''
def main():
    lease=lock(P/'ready.lock');assert not (P/'READY_STARTED.json').exists();save(P/'READY_STARTED.json',dict(pid=os.getpid(),utc=now()))
    while not (P/'AUG_JOBS_FROZEN.json').exists():
        assert not (P/'AUG_ERROR.json').exists();hard_stop();time.sleep(10)
    source=read(P/'AUG_JOBS_FROZEN.json');plan=dict(seed=2027,epochs=30,batch_size=8,lr=.009,weight_decay=0.,runs=source['runs'],source_jobs_sha256=sha(P/'AUG_JOBS_FROZEN.json'),offsets_sha256=sha(P/'OFFSETS_V2.json'),no_training_after_V5=True)
    save(P/'TRAIN_PLAN_EARLY.json',plan);seen={}
    while True:
        hard_stop();assert not (P/'AUG_ERROR.json').exists(),'Augmentation failed, no automatic continuation';ready=[]
        for run in plan['runs']:
            f=run['fold'];target=P/f'FOLD{f}_READY.json'
            if target.exists():ready.append(f);continue
            resolved=run
            if (P/'AUG_COMPLETE.json').exists():resolved=read(P/'TRAIN_PLAN.json')['runs'][f]
            bindings={};missing=0
            for j in resolved['side_jobs']:
                key=json.dumps(['MCD_SIDE',j['recording'],float(j['start']),j['recipe'],bool(j['lowres']),f],separators=(',',':'));digest=hashlib.sha256(key.encode()).hexdigest();rp=P/'side_aug'/str(f)/(digest+'.json')
                if not rp.exists():missing+=1;continue
                r=read(rp);assert r['key']==key;file=rp.with_suffix('.npz')
                if str(rp) not in seen:assert sha(file)==r['sha256'];seen[str(rp)]=sha(rp)
                else:assert sha(rp)==seen[str(rp)]
                bindings[str(rp)]=seen[str(rp)]
            if not missing:
                assert all(any(j['camera']==v for j in resolved['side_jobs']) for v in ['IriunWebcam','USBVideo'])
                save(target,dict(utc=now(),run=resolved,cache_receipt_hashes=bindings,source_jobs_sha256=sha(P/'AUG_JOBS_FROZEN.json'),all_side_cache_hashes_verified=True,excluded_jobs=len(run['side_jobs'])-len(resolved['side_jobs'])));ready.append(f)
        save(P/'READY_STATUS.json',dict(utc=now(),folds=ready,pid=os.getpid()))
        if ready==list(range(5)):save(P/'READY_COMPLETE.json',dict(utc=now(),folds=ready));return
        time.sleep(20)
if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'READY_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
