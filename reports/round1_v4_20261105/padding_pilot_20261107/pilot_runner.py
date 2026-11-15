"""V4 phase branch: frozen V3 examples, bounded-lag loss, balanced batch8/GN4."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,gc,hashlib,json,math,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE.parent/'code'))
V3CODE=Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102')
sys.path.insert(0,str(V3CODE))
from t8_plan import read,write,sha,code_fingerprint,pools,steps_per_epoch,draw,SEEDS
import t8_run as baseline
import polarity_guard as guard
ROOT=Path('/mnt/samsung_t5/khangle/facevitals')
ORIGINAL=ROOT/'t8_oval_20261031/runs';V3=ROOT/'t8_polarity_v3_20261102'
REPORT=ROOT/'reports/round1_v4_20261105';OUT=REPORT/'v4_padding_pilot'
INDEX=ROOT/'t8_oval_20261031/training_index.json'
def extra_bytes(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0
def storage(index,receipt,fresh=False):
    r=read(receipt);age=(datetime.now(timezone.utc)-datetime.fromisoformat(r['measured_utc'].replace('Z','+00:00'))).total_seconds()
    assert -60<=age and (not fresh or age<=3600),'PROJECT_WAIT: fresh cross-server inventory required'
    base=guard.storage_guard(index,ORIGINAL.parent,V3)
    extra=extra_bytes(REPORT)+extra_bytes(Path('/data_pool/khangle/facevitals/reports/round1_v4_20261105'))+r['a100_v4_bytes']+r.get('V4_CPU_reserved_bytes',0)
    total=base['total_bytes']+extra
    assert total<=64_900_000_000,'STORAGE_WAIT: T8 whole65GB including V4 and100MB headroom'
    project=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in (ROOT,Path('/data_pool/khangle/facevitals')))+r['a100_bytes']
    # Account for the entire pending raw-copy allocation, not just its current partial bytes.
    project+=r.get('remaining_raw_copy_reservation_bytes',0)
    assert project<=499_900_000_000,'PROJECT_WAIT: 500GB project'
    return dict(T8_bytes=total,project_bytes_conservative=project,base=base,V4_bytes=extra,receipt_sha256=sha(receipt))
def own_gpu_memory():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    values=[int(line.split(',')[1].strip()) for line in raw.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
    return max(values) if values else None

def make_model(gn=False):
    import torch
    from EfficientPhys_fixed import EfficientPhys
    from replicate_tsm import use_replicate_padding
    model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36))
    if gn:
        assert isinstance(model.batch_norm,torch.nn.BatchNorm2d)
        model.batch_norm=torch.nn.GroupNorm(1,3)
    return model.cuda()

def validation_clips(index,fold,seed):
    import numpy as np
    from polarity_loader import load_canonical_clip
    base=V3/f'fold{fold}'/f'seed{seed}'/'joint/evaluation/UBFC2_native/waveforms.npz'
    with np.load(base) as archive:
        keys=[(str(r),float(t)) for r,t in archive['keys']]
    ids=sorted({r for r,_ in keys},key=lambda s:int(s.removeprefix('subject')))[:5]
    clips=[]
    for rid in ids:
        start=min(t for r,t in keys if r==rid)
        x,y,*_=load_canonical_clip(Path(index['roots']['UBFC2'])/rid/'native/crop',start,'UBFC2')
        clips.append((x,y))
    return clips

def wait_for_gpu(receipt):
    while True:
        try:
            baseline.resource_guard()
            return
        except RuntimeError as error:
            if not str(error).startswith('RESOURCE_WAIT:'):
                raise
            age=datetime.now(timezone.utc)-datetime.fromisoformat(read(receipt)['measured_utc'])
            if age.total_seconds()>3000:
                raise
            print(str(error),flush=True)
            time.sleep(30)
def probe(index,batch,gn):
    import torch
    from repair_loss import phase_loss
    baseline.seed_all(2027,0);pool=pools(index,0)
    picks=[p for step in range(batch//2) for p in draw(pool,2027,0,0,step,'joint')]
    x,y=baseline.batch_arrays(index,picks);m=make_model(gn);opt=torch.optim.AdamW(m.parameters(),lr=.009,weight_decay=0.)
    pred=m(torch.from_numpy(x).cuda());loss=phase_loss(pred,torch.from_numpy(y).cuda()).mean()
    assert torch.isfinite(loss);loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    opt.step();torch.cuda.synchronize()
    return dict(batch_size=batch,group_norm=gn,loss=float(loss.detach()),finite_backward=True,optimizer_step=True,max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),cache_picks=picks)
def choose_batch(index):
    import torch
    result=OUT/'BATCH_PROBE.json'
    if result.exists():return read(result)
    attempts=[]
    for batch,gn in [(8,False),(4,True)]:
        torch.cuda.reset_peak_memory_stats()
        try:
            good=probe(index,batch,gn);good['attempts']=attempts;write(result,good)
            return good
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append(dict(batch_size=batch,group_norm=gn,outcome='CUDA_OOM',error=str(exc)))
        finally:gc.collect();torch.cuda.empty_cache()
    write(result,dict(passed=False,attempts=attempts));raise RuntimeError('Both predeclared batch options OOM')
def run(index,fold,seed,settings,receipt,frozen):
    import torch,numpy as np
    from repair_loss import phase_loss
    from phase_loss import grouped_draws
    from phase_evaluate import evaluate
    dest=OUT/f'fold{fold}'/f'seed{seed}'/'joint'
    if dest.exists():
        s=read(dest/'status.json');assert s['state']=='complete','Partial run preserved; inspect explicitly'
        assert sha(dest/'final.pt')==s['checkpoint_sha256'] and read(dest/'provenance.json')['v4_code']==frozen
        return
    wait_for_gpu(receipt);budget=storage(index,receipt)
    batch=settings['batch_size'];gn=settings['group_norm'];pool=pools(index,fold);oldsteps=steps_per_epoch(pool);steps=math.ceil(oldsteps/(batch//2))
    baseline.seed_all(seed,fold);model=make_model(False);original_initial=baseline.model_hash(model)
    histories=guard.baseline_pair(ORIGINAL,fold,seed,pool,original_initial,baseline.BASELINE_CODE)
    if gn:model.batch_norm=torch.nn.GroupNorm(1,3).cuda()
    original_v3=V3/f'fold{fold}'/f'seed{seed}'/'joint';v3hist=read(original_v3/'history.json')
    assert read(original_v3/'provenance.json')['initial_model_sha256']==original_initial
    dest.mkdir(parents=True);baseline.ADAPTER_RECEIPTS={}
    config=dict(fold=fold,seed=seed,arm='joint',branch='v4_padding_pilot',padding='replicate_edges_training_and_inference',epochs=30,batch_size=batch,group_norm=gn,
        loss='difference MSE + half zero-lag and half best +/-6 signed Pearson on differentiably bandpassed integrated waves',
        steps_per_epoch=steps,frozen_pairs_per_epoch=oldsteps,max_lr=.009,weight_decay=0.,selection='final_epoch_30',
        frontend_sha256=read(V3CODE/'SOURCE_CACHE_SHA256.json')['oval_crop.py'],
        landmark_sha256=read(V3CODE/'SOURCE_CACHE_SHA256.json')['oval_frontend.py'],
        index_sha256=sha(INDEX),D3_opened=False,test_opened=False,offset_seconds=0,
        correction_manifest_sha256=sha(V3CODE/'CORRECTION.json'),v4_code=frozen,
        comparison_reference='v3_joint full5fold3seed; primary first MCDnative9.3591632; POS4.670 matched')
    write(dest/'provenance.json',dict(config=config,v4_code=frozen,index_sha256=sha(INDEX),v3_code=read(V3CODE/'FROZEN_SHA256.json'),original_initial_model_sha256=original_initial,initial_model_sha256=baseline.model_hash(model),storage=budget,train_pool_sha256=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()))
    opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=0.)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.009,epochs=30,steps_per_epoch=steps);history=[]
    val=validation_clips(index,fold,seed)
    for epoch in range(30):
        storage(index,receipt);model.train();started=time.time();order=hashlib.sha256();dl={'UBFC2':[],'MCD':[]};sc={'UBFC2':[],'MCD':[]};codecs={};updates=0
        for pairs,picks in grouped_draws(pool,seed,fold,epoch,oldsteps,batch,draw):
            for pair in pairs:order.update(json.dumps(pair).encode())
            for ds,_,_,variant in picks:codecs[ds+':'+variant]=codecs.get(ds+':'+variant,0)+1
            x,y=baseline.batch_arrays(index,picks);opt.zero_grad(set_to_none=True)
            pred=model(torch.from_numpy(x).cuda());losses=phase_loss(pred,torch.from_numpy(y).cuda());loss=losses.mean()
            assert torch.isfinite(loss),'Non-finite loss';loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Non-finite gradient'
            opt.step();sched.step();updates+=1
            vals=losses.detach().cpu().tolist();std=pred.detach().cumsum(-1).std(-1).cpu().tolist()
            for pick,l,s in zip(picks,vals,std):dl[pick[0]].append(l);sc[pick[0]].append(s)
        assert updates==steps and order.hexdigest()==v3hist[epoch]['draw_sha256'],'Frozen examples/augmentation differ from V3'
        assert codecs==v3hist[epoch]['codec_counts']
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(dl['UBFC2']+dl['MCD'])),dataset_train_loss={k:float(np.mean(v)) for k,v in dl.items()},dataset_predicted_integrated_std={k:float(np.mean(v)) for k,v in sc.items()},dataset_train_clips={k:len(v) for k,v in dl.items()},optimizer_updates=updates,elapsed_s=time.time()-started,draw_sha256=order.hexdigest(),codec_counts=codecs))
        model.eval()
        with torch.no_grad():
            losses_val=[phase_loss(model(torch.from_numpy(x[None]).cuda()),torch.from_numpy(y[None]).cuda()).item() for x,y in val]
        history[-1]['five_UBFC_clean_validation_loss']=float(np.mean(losses_val))
        torch.cuda.synchronize()
        history[-1]['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        history[-1]['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        history[-1]['own_gpu_memory_MiB']=own_gpu_memory()
        baseline.save_checkpoint(dict(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),history=history,config=config,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),dest/'last.pt')
        write(dest/'history.json',history);write(dest/'status.json',dict(state='training',completed_epochs=epoch+1));write(dest/'POLARITY_ADAPTER_RECEIPTS.json',baseline.ADAPTER_RECEIPTS)
        print(json.dumps(dict(fold=fold,seed=seed,**history[-1])),flush=True)
        if (fold,seed,epoch+1)==(0,2027,5):
            from pilot_gate import check_epoch5
            early=check_epoch5(history)
            write(OUT/'EPOCH5_GATE.json',early)
            if not early['passed']:
                stopped=dict(state='stopped_epoch5_loss',completed_epochs=5,gate=early)
                write(dest/'status.json',stopped)
                write(OUT/'queue_status.json',stopped)
                return
    config['polarity_adapter_receipts_sha256']=sha(dest/'POLARITY_ADAPTER_RECEIPTS.json')
    baseline.save_checkpoint(dict(model=model.state_dict(),config=config,epoch=30),dest/'final.pt')
    write(dest/'status.json',dict(state='training_complete_evaluation_pending',completed_epochs=30,checkpoint_sha256=sha(dest/'final.pt')))
    evaluate(model,index,fold,seed,dest/'evaluation',V3,'cuda')
    torch.cuda.synchronize()
    write(dest/'MEMORY_PEAK.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),own_gpu_memory_MiB=own_gpu_memory(),scope='probe_plus_full_training_and_evaluation'))
    from test_static_tsm import static_result
    static=static_result(model.cpu())
    write(dest/'STATIC.json',dict(result=static,pass_static=static['hr'] is None))
    write(dest/'status.json',dict(state='complete',completed_epochs=30,checkpoint_sha256=sha(dest/'final.pt'),no_outer_val_selection=True))
    del model,opt,sched,pred,loss,losses;gc.collect();torch.cuda.empty_cache()
def main(a):
    import fcntl
    index=read(INDEX);assert sha(INDEX)==guard.EXPECTED_INDEX and index['hashes_verified']
    assert code_fingerprint(V3CODE)==read(V3CODE/'FROZEN_SHA256.json')
    validation=read(CODE/'VALIDATION.json')
    assert validation['passed']
    frozen=validation['files']
    assert all(sha(CODE/name)==digest for name,digest in frozen.items())
    assert all(sha(REPORT/'code'/name)==digest for name,digest in validation['source_files'].items())
    assert (CODE/'OLD_QUEUE_STOPPED.txt').is_file()
    lock,controls=guard.acquire_completed_baseline(ORIGINAL)
    vlock=(V3/'queue.lock').open('r+');fcntl.flock(vlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for f in range(5):
        for seed in SEEDS:
            r=V3/f'fold{f}'/f'seed{seed}'/'joint';s=read(r/'status.json')
            assert s['state']=='complete' and s['completed_epochs']==30 and sha(r/'final.pt')==s['checkpoint_sha256']
    OUT.mkdir(parents=True,exist_ok=True);own=(OUT/'queue.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=storage(index,a.receipt,fresh=True)
    write(OUT/'PREFLIGHT.json',dict(index_sha256=sha(INDEX),v4_code=frozen,baseline_controls=controls,storage=budget,raw_UBFC_copy_required=False))
    if a.preflight_only:return
    wait_for_gpu(a.receipt)
    storage(index,a.receipt,fresh=True)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0)
    settings=choose_batch(index)
    from pilot_gate import score_pilot
    all_pairs=[(fold,seed) for fold in range(5) for seed in SEEDS]
    assert not a.remaining,'Padding pilot is exactly one predeclared run'
    pairs=all_pairs[:1]
    if a.remaining:
        decision=score_pilot(REPORT,OUT)
        assert decision['passed'] and decision==read(OUT/'PILOT_GATE.json')
        pairs=all_pairs[1:]
    for f,seed in pairs:
        write(OUT/'queue_status.json',dict(state='running',pid=os.getpid(),fold=f,seed=seed,settings=settings))
        run(index,f,seed,settings,a.receipt,frozen)
        if read(OUT/'queue_status.json')['state']=='stopped_epoch5_loss':
            return
    decision=score_pilot(REPORT,OUT)
    write(OUT/'PILOT_GATE.json',decision)
    write(OUT/'queue_status.json',dict(state='complete' if a.remaining else 'pilot_complete',runs=15 if a.remaining else 1,gate=decision))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');p.add_argument('--remaining',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception as e:
        if OUT.exists():write(OUT/'queue_error.json',dict(error=str(e),traceback=traceback.format_exc()))
        raise
