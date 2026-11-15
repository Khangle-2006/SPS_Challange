"""Last authorized five-fold multiview training, final30, no outcome selection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
import sys,time,gc,hashlib,traceback
import numpy as np
from runtime import *
from provider import SideProvider,DiskProvider
sys.path.insert(0,str(E));from common import BTCData,infer_native
from model_eval import c5_rows
from repair_loss import phase_loss
from test_static_tsm import static_result
from phase_evaluate import evaluate
METHOD='v5_multiview_seed2027'
def frozen():
    paths=[P/n for n in ['train.py','provider.py','runtime.py','TRAIN_PLAN_EARLY.json','PROTOCOL.md','TRAINING_ADDENDUM.md','OFFSETS_V2.json','SIDE_INDEX.json','AUG_JOBS_FROZEN.json','report.py','public_summary.py','validate.py']]
    paths += [E/n for n in ['training_provider.py','common.py','model_eval.py','PLAN_ELIGIBLE.json']]
    return {str(p):sha(p) for p in paths}
def report():
    # Standalone process avoids old aggregation/common module name collisions.
    import subprocess
    subprocess.run([sys.executable,str(P/'report.py')],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
def one(run,plan,bindings):
    import torch
    p=old.p;f=run['fold'];dest=P/'runs'/f'fold{f}/seed2027/joint';assert not dest.exists(),'Partial run: inspect; never restart'
    hard_stop();gpu();budget(fresh=True);p.baseline.seed_all(2027,f);model=p.make_model(False)
    initial=p.baseline.model_hash(model);assert initial==read(p.V3/f'fold{f}/seed2027/joint/provenance.json')['initial_model_sha256']
    dest.mkdir(parents=True);config=dict(fold=f,seed=2027,epochs=30,branch=METHOD,padding='replicate',loss='unchanged repair_loss.phase_loss',plan_sha256=sha(P/'TRAIN_PLAN_EARLY.json'),ready_sha256=sha(P/f'FOLD{f}_READY.json'),validation_sha256=sha(P/f'VALIDATION_FOLD{f}.json'),selection='final_epoch30',frontend_sha256=read(p.V3CODE/'SOURCE_CACHE_SHA256.json')['oval_crop.py'],index_sha256=sha(p.INDEX),D3_opened=False,test_opened=False)
    save(dest/'provenance.json',dict(config=config,initial_model_sha256=initial,bindings=bindings,storage=budget()))
    public=DiskProvider(read(p.INDEX),run['btc_bpp_quartiles'],300_000_000);side=SideProvider(run['btc_bpp_quartiles'])
    choices={'UBFC2':run['public_jobs']['UBFC2'],'FullHDwebcam':run['public_jobs']['MCD']}
    choices.update({view:[j for j in run['side_jobs'] if j['camera']==view] for view in ['IriunWebcam','USBVideo']});assert all(choices.values())
    opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=0.);sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.009,epochs=30,steps_per_epoch=run['steps_per_epoch']);history=[]
    for epoch in range(1,31):
        hard_stop();budget();assert frozen()==bindings;model.train();started=time.monotonic();rng=np.random.default_rng(np.random.SeedSequence([2027,f,epoch]));losses=[];counts={k:0 for k in choices};drawhash=hashlib.sha256()
        for step in range(run['steps_per_epoch']):
            hard_stop();xs=[];ys=[]
            for i in range(8):
                view='UBFC2' if i%2==0 else ['FullHDwebcam','IriunWebcam','USBVideo'][int(rng.integers(3))]
                j=choices[view][int(rng.integers(len(choices[view])))];rate=float(rng.uniform(*j['rates']));ds='UBFC2' if view=='UBFC2' else 'MCD' if view=='FullHDwebcam' else 'MCD_SIDE';provider=side if ds=='MCD_SIDE' else public
                x,y,_=provider.clip(ds,j['recording'],j['start'],j['recipe'],j['lowres'],f,rate);xs.append(x);ys.append(y);counts[view]+=1;drawhash.update(json.dumps(dict(view=view,job=j,rate=rate),sort_keys=True).encode())
            opt.zero_grad(set_to_none=True);pred=model(torch.from_numpy(np.stack(xs)).cuda());loss=phase_loss(pred,torch.from_numpy(np.stack(ys)).cuda()).mean();assert torch.isfinite(loss),'nonfinite_loss';loss.backward()
            assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None),'nonfinite_gradient'
            opt.step();sched.step();losses.append(float(loss.detach()));save(dest/'status.json',dict(state='training',epoch=epoch,step=step+1,steps=run['steps_per_epoch'],pid=os.getpid(),utc=now()))
        model.cpu();static=static_result(model);save(dest/f'STATIC_EPOCH_{epoch:02d}.json',dict(epoch=epoch,result=static,passed=static['hr'] is None))
        entry=dict(epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.monotonic()-started,static_pass=static['hr'] is None,view_clips=counts,draw_sha256=drawhash.hexdigest(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved());history.append(entry)
        p.baseline.save_checkpoint(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),config=config,epoch=epoch,history=history),dest/'last.pt');save(dest/'history.json',history);print(json.dumps(dict(fold=f,**entry)),flush=True)
        if static['hr'] is not None:
            save(dest/'REJECTED_STATIC.json',dict(epoch=epoch,result=static));raise RuntimeError('STATIC_FAILED: preserve and stop V5, no automatic restart')
        model.cuda()
    model.cpu();p.baseline.save_checkpoint(dict(model=model.state_dict(),config=config,epoch=30),dest/'final.pt');save(dest/'STATIC.json',dict(result=static,pass_static=True));save(dest/'status.json',dict(state='training_complete_evaluation_pending',completed_epochs=30,checkpoint_sha256=sha(dest/'final.pt')))
    # BTC negative controls using exact same frozen test, on only this dev fold.
    sys.path.insert(0,str(N));from padding_followup_v3 import btc_static
    data=BTCData(f);btc_static(model,data,f,dest);model.cuda();save(dest/'NATIVE.json',infer_native(model,data,f,METHOD,'cuda'))
    report() # Native is reported immediately, before lengthy public/C5 evaluation.
    save(dest/'C5PRIME.json',c5_rows(model,f,'cuda',METHOD));evaluate(model,read(p.INDEX),f,2027,dest/'evaluation',p.V3,'cuda')
    save(dest/'status.json',dict(state='complete',completed_epochs=30,checkpoint_sha256=sha(dest/'final.pt'),utc=now(),no_outer_val_selection=True));report()
    del model,opt,sched,public,side,data;gc.collect();torch.cuda.empty_cache()
def main():
    lease=lock(P/'train.lock');leases=[lock(x) for x in [N/'queue.lock',E/'sweep.lock',E/'gpu_execution.lock',old.p.ORIGINAL/'queue.lock',old.p.V3/'queue.lock',old.ORIGINAL_OUT/'queue.lock']]
    assert not (P/'TRAIN_STARTED.json').exists(),'Already launched; inspect, never duplicate'
    plan=read(P/'TRAIN_PLAN_EARLY.json');assert [r['fold'] for r in plan['runs']]==list(range(5));bindings=frozen();save(P/'FROZEN.json',bindings);old.bindings();hard_stop()
    def validate_fold(f):
        import subprocess
        target=P/f'VALIDATION_FOLD{f}.json'
        assert not target.exists(),'Unexpected validation receipt; inspect'
        with (P/f'validate_fold{f}.log').open('x') as log:
            subprocess.run(['taskset','-c','8,9',sys.executable,str(P/'validate.py'),'--fold',str(f)],check=True,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        v=read(target);assert v['passed'] and v['ready_sha256']==sha(P/f'FOLD{f}_READY.json')
    validate_fold(0);gpu()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0)
    save(P/'TRAIN_STARTED.json',dict(pid=os.getpid(),utc=now(),budget=budget(fresh=True),bindings=bindings,GPU=0,memory_fraction=.35,threads=2))
    for run in plan['runs']:
        f=run['fold']
        while not (P/f'FOLD{f}_READY.json').exists():
            assert not (P/'AUG_ERROR.json').exists() and not (P/'READY_ERROR.json').exists(),'Cache dependency failed'
            save(P/'TRAIN_STATUS.json',dict(state='waiting_fold_cache',fold=f,utc=now()));hard_stop();time.sleep(20)
        run=read(P/f'FOLD{f}_READY.json')['run']
        if f!=0:validate_fold(f)
        while True:
            hard_stop()
            try:budget(fresh=True);break
            except AssertionError as e:
                if str(e)!='FRESH_CROSS_SERVER_INVENTORY_REQUIRED':raise
                save(P/'TRAIN_STATUS.json',dict(state='waiting_fresh_inventory',next_fold=run['fold'],utc=now()));time.sleep(30)
        save(P/'TRAIN_STATUS.json',dict(state='running',fold=run['fold'],pid=os.getpid(),utc=now()));one(run,plan,bindings)
    save(P/'TRAIN_COMPLETE.json',dict(utc=now(),folds=5,seed=2027,no_training_after_V5=True));save(P/'TRAIN_STATUS.json',dict(state='complete',utc=now()));report()
if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'TRAIN_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
