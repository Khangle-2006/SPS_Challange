"""One CPU worker, streaming matched controls, with durable per-checkpoint outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,sys,time,json,hashlib,fcntl
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
from common import *
sys.path.insert(0,str(V3CODE))
sys.path.insert(0,str(REPORT/'tsm_padding'))
import torch
from EfficientPhys_fixed import EfficientPhys
from t8_evaluate import decode_difference
from polarity_loader import load_canonical_clip
from test_static_tsm import static_result

HERE=Path(__file__).resolve().parent
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def now():return datetime.now(timezone.utc).isoformat()
def model_for(run):
    m=EfficientPhys(frame_depth=10,img_size=36)
    ck=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model']);m.eval()
    return m
def infer(model,x):
    with torch.inference_mode():w=model(torch.from_numpy(x[None]))[0].numpy()
    assert np.isfinite(w).all(),'Nonfinite output is not a legitimate abstention'
    try:return float(decode_difference(w))
    except ValueError as e:
        assert str(e)=='invalid_prediction_waveform',str(e)
        return None
def run_one(run,frozen,index):
    name=run.parent.parent.name+'_'+run.parent.name;dest=HERE/name;dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        done=read(dest/'COMPLETE.json');assert done['checkpoint_sha256']==sha(run/'final.pt')
        return
    allrows=load_run(run,frozen);rows=[r for r in allrows if r['accepted']]
    model=model_for(run);static=static_result(model)
    save(dest/'STATIC.json',dict(utc=now(),checkpoint_sha256=sha(run/'final.pt'),result=static,pass_static=static['hr'] is None))
    # Deterministic person derangement, same condition; no HR-dependent donor selection.
    groups=defaultdict(list)
    for r in rows:groups[r['condition'],r['subject_id']].append(r)
    donor={}
    for cond in CONDITIONS:
        ids=sorted(s for c,s in groups if c==cond);assert len(ids)>1
        for i,s in enumerate(ids):
            source=sorted(groups[cond,s],key=key);other=sorted(groups[cond,ids[(i+1)%len(ids)]],key=key)
            for j,r in enumerate(source):donor[cond,key(r)]=other[j%len(other)]
    fn=dest/'windows.jsonl'
    existing=[json.loads(s) for s in fn.read_text().splitlines()] if fn.exists() else []
    assert len(existing)<=len(rows)
    for a,b in zip(existing,rows):assert (a['condition'],key(a))==(b['condition'],key(b))
    output=existing
    with fn.open('a',buffering=1) as stream:
        for i,r in enumerate(rows[len(existing):],len(existing)):
            cond=r['condition'];ds,var=CONDITIONS[cond]
            x,*_=load_canonical_clip(Path(index['roots'][ds])/r['recording_id']/var/'crop',r['window_start_s'],ds)
            static_x=np.repeat(x[:1],len(x),axis=0)
            if i<5:
                actual=infer(model,static_x)
                assert actual==static['hr'],'Static zero-difference invariance failed'
                # Reproduce native inference on CPU; a mismatch stops the audit.
                assert infer(model,x)==r['prediction_hr_bpm'],'CPU/GPU HR parity failed'
            rng=np.random.default_rng(int(hashlib.sha256(repr((cond,key(r))).encode()).hexdigest()[:16],16))
            noise=static_x+rng.normal(0,.01,x.shape).astype(np.float32)
            perm=rng.permutation(len(x));shuffled=x[perm].copy()
            d=donor[cond,key(r)]
            row={k:r[k] for k in ('subject_id','recording_id','window_start_s','condition','reference_hr_bpm','prediction_hr_bpm')}
            row.update(static=static['hr'],static_noise=infer(model,noise),shuffled=infer(model,shuffled),
                       wrong_person_reference=d['reference_hr_bpm'],donor_person=d['subject_id'],donor_record=d['recording_id'],donor_start=d['window_start_s'])
            stream.write(json.dumps(row,allow_nan=False)+'\n');output.append(row)
            if (i+1)%20==0:
                save(HERE/'STATUS.json',dict(utc=now(),state='running',checkpoint=name,windows_done=i+1,windows_total=len(rows),pid=os.getpid()))
    summary={}
    for cond in CONDITIONS:
        rr=[r for r in output if r['condition']==cond]
        summary[cond]={test:metrics(rr,pred,ref) for test,pred,ref in (
            ('main','prediction_hr_bpm','reference_hr_bpm'),('static','static','reference_hr_bpm'),
            ('static_noise','static_noise','reference_hr_bpm'),('shuffled','shuffled','reference_hr_bpm'),
            ('wrong_person_label','prediction_hr_bpm','wrong_person_reference'))}
        summary[cond]['source_candidates']=sum(r['condition']==cond for r in allrows)
        summary[cond]['source_eligible']=sum(r['condition']==cond and r['reference_eligible'] for r in allrows)
    save(dest/'COMPLETE.json',dict(utc=now(),checkpoint_sha256=sha(run/'final.pt'),source_windows_sha256=sha(FROZEN),
         windows_sha256=sha(fn),protocol_sha256=sha(HERE/'PROTOCOL.md'),code_sha256=sha(__file__),pass_static=static['hr'] is None,conditions=summary))
    print('COMPLETE',name,json.dumps(summary),flush=True)

def main(watch):
    lock=(HERE/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=read(FROZEN);index=read(ROOT/'t8_oval_20261031/training_index.json')
    while True:
        runs=completed()
        # Immediate cheap static screen on all available checkpoints, before long controls.
        for run in runs:
            d=HERE/(run.parent.parent.name+'_'+run.parent.name);d.mkdir(exist_ok=True)
            if not (d/'STATIC.json').exists():
                assert sha(run/'final.pt')==read(run/'status.json')['checkpoint_sha256']
                m=model_for(run);res=static_result(m)
                save(d/'STATIC.json',dict(utc=now(),checkpoint_sha256=sha(run/'final.pt'),result=res,pass_static=res['hr'] is None))
                print('STATIC',d.name,json.dumps(res),flush=True)
        for run in runs:run_one(run,frozen,index)
        done=len(list(HERE.glob('fold*_seed*/COMPLETE.json')))
        save(HERE/'STATUS.json',dict(utc=now(),state='complete' if done==15 else 'waiting_new_checkpoint',checkpoints_complete=done,total=15,pid=os.getpid()))
        if done==15 or not watch:return
        time.sleep(30)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    try:main(a.watch)
    except Exception as e:
        save(HERE/'ERROR.json',dict(utc=now(),error=repr(e)));raise
