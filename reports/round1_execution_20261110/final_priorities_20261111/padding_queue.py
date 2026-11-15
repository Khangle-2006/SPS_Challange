"""New authorized four-fold padding baseline; original pilot training math unchanged."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import sys,json,time,hashlib,subprocess,fcntl,traceback,gc,argparse
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent; E=P.parent; R=E.parent.parent
V=R/'reports/round1_v4_20261105'; PC=V/'padding_pilot_20261107'
sys.path.insert(0,str(PC))
import pilot_runner as p
OUT=P/'padding_fixed'; ORIGINAL_OUT=p.OUT
def now():return datetime.now(timezone.utc).isoformat()
def save(f,x):p.write(f,x)
def size(d):
    return sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file()) if Path(d).exists() else 0
def budget(index=None,receipt=None,fresh=False):
    a=p.read(E/'A100_INVENTORY.json')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(a['measured_utc'])).total_seconds()
    assert age>=-60 and (not fresh or age<=3600),'FRESH_CROSS_SERVER_INVENTORY_REQUIRED'
    base=p.guard.storage_guard(p.read(p.INDEX),p.ORIGINAL.parent,p.V3)['total_bytes']
    used=base+size(V)+size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+a['a100_v4_bytes']+size(R/'reports/round1_bayes_20261108')+size(E)
    own=size(P); remaining=max(0,2_000_000_000-own)
    project=sum(int(subprocess.check_output(['du','-sb',str(d)],text=True).split()[0]) for d in (R,Path('/data_pool/khangle/facevitals')))+a['a100_bytes']
    assert own<2_000_000_000 and used+remaining+420_000_000<84_000_000_000,'STOP_STORAGE_84GB'
    assert project+remaining+420_000_000<500_000_000_000,'STOP_PROJECT_500GB'
    return dict(utc=now(),scope_actual_bytes=used,project_actual_bytes=project,new_scope_bytes=own,additional_reserved_bytes=remaining,headroom_bytes=420_000_000,a100_age_seconds=age,a100_sha256=p.sha(E/'A100_INVENTORY.json'))
def hard_stop():
    assert datetime.now(timezone.utc)<datetime(2026,11,22,16,59,59,tzinfo=timezone.utc),'TRAINING_HARD_STOP'
def gpu_wait(receipt=None):
    while True:
        hard_stop(); samples=[]
        for _ in range(3):
            row=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
            idx,free,util=map(int,row.split(','));assert idx==0
            samples.append(dict(index=idx,free_MiB=free,utilization=util));time.sleep(2)
        save(P/'GPU_CHECK.json',dict(utc=now(),samples=samples,required_free_MiB=6000,sharing_authorized=True))
        if all(x['free_MiB']>=6000 for x in samples):return
        save(P/'WAIT.json',dict(utc=now(),reason='GPU0_VRAM',samples=samples));time.sleep(30)
def bindings():
    validation=p.read(PC/'VALIDATION.json')
    assert validation['passed']
    for name,digest in validation['files'].items():assert p.sha(PC/name)==digest, name
    for name,digest in validation['source_files'].items():assert p.sha(V/'code'/name)==digest,name
    assert p.code_fingerprint(p.V3CODE)==p.read(p.V3CODE/'FROZEN_SHA256.json')
    assert p.sha(p.INDEX)==p.guard.EXPECTED_INDEX
    return dict(pilot_validation_sha256=p.sha(PC/'VALIDATION.json'),pilot_files=validation['files'],source_files=validation['source_files'],wrapper_sha256=p.sha(__file__),protocol_sha256=p.sha(P/'PROTOCOL.md'),request_sha256=p.sha(P/'REQUEST.txt'),index_sha256=p.sha(p.INDEX),v3_frozen=p.read(p.V3CODE/'FROZEN_SHA256.json'))
def lock(file):
    file.parent.mkdir(parents=True,exist_ok=True);h=file.open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def main(preflight=False):
    lease=lock(P/'queue.lock')
    assert not (P/'LAUNCH.json').exists(),'Existing launch: inspect, never duplicate/restart'
    baseline_lock,controls=p.guard.acquire_completed_baseline(p.ORIGINAL)
    v3lock=lock(p.V3/'queue.lock');pilotlock=lock(ORIGINAL_OUT/'queue.lock')
    oldfastlock=lock(E/'sweep.lock')
    gate=p.read(ORIGINAL_OUT/'PILOT_GATE.json');assert gate['passed']
    pilot=ORIGINAL_OUT/'fold0/seed2027/joint'
    assert p.sha(pilot/'final.pt')==gate['checkpoint_sha256']=='e0315095fda595be37e4dc3ce3b693016111cfe7da81ef924a839c5ddf32ab0c'
    assert p.read(ORIGINAL_OUT/'EPOCH5_GATE.json')['passed']
    for fold in range(5):
        for seed in p.SEEDS:
            dest=p.V3/f'fold{fold}'/f'seed{seed}'/'joint';state=p.read(dest/'status.json')
            assert state['state']=='complete' and state['completed_epochs']==30 and p.sha(dest/'final.pt')==state['checkpoint_sha256']
    frozen=bindings();settings=p.read(ORIGINAL_OUT/'BATCH_PROBE.json')
    assert settings['batch_size']==8 and not settings['group_norm']
    b=budget(fresh=True)
    save(P/'PREFLIGHT.json',dict(utc=now(),passed=True,storage=b,bindings=frozen,fold0_preserved_sha256=p.sha(pilot/'final.pt'),settings=settings,baseline_controls=controls))
    if preflight:return
    gpu_wait();budget(fresh=True);hard_stop()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0)
    # Checkpoint hook adds static checks without changing training samples/math/RNG/BN.
    original_save=p.baseline.save_checkpoint
    def checked_save(payload,path):
        hard_stop()
        if Path(path).name=='last.pt':
            from EfficientPhys_fixed import EfficientPhys
            from replicate_tsm import use_replicate_padding
            from test_static_tsm import static_result
            with torch.random.fork_rng(devices=[]):
                m=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36))
                m.load_state_dict({k:v.detach().cpu() for k,v in payload['model'].items()})
                result=static_result(m)
            del m
            original_save(payload,path)
            save(Path(path).parent/f"STATIC_EPOCH_{payload['epoch']:02d}.json",dict(utc=now(),epoch=payload['epoch'],result=result,passed=result['hr'] is None,threshold_unchanged=True))
            if result['hr'] is not None:
                save(Path(path).parent/'REJECTED_STATIC.json',dict(utc=now(),epoch=payload['epoch'],result=result))
                raise RuntimeError('STATIC_FAILED: checkpoint preserved; no continuation')
        else:original_save(payload,path)
    p.baseline.save_checkpoint=checked_save;p.wait_for_gpu=gpu_wait;p.storage=budget;p.OUT=OUT
    OUT.mkdir(exist_ok=True)
    save(P/'FROZEN.json',frozen)
    save(P/'LAUNCH.json',dict(utc=now(),pid=os.getpid(),folds=[1,2,3,4],seed=2027,epochs=30,GPU=0,memory_fraction=.35,threads=2,storage=b,not_old_fasttrack=True))
    index=p.read(p.INDEX)
    for fold in [1,2,3,4]:
        assert bindings()==frozen
        hard_stop()
        # New run must have fresh inventory; wait without modifying/rerunning any run.
        while True:
            try:budget(fresh=True);break
            except AssertionError as e:
                if str(e)!='FRESH_CROSS_SERVER_INVENTORY_REQUIRED':raise
                save(P/'WAIT.json',dict(utc=now(),reason=str(e),next_fold=fold));hard_stop();time.sleep(30)
        save(OUT/'queue_status.json',dict(state='running',pid=os.getpid(),fold=fold,seed=2027,utc=now()))
        torch.cuda.reset_peak_memory_stats()
        p.run(index,fold,2027,settings,E/'A100_INVENTORY.json',frozen)
        dest=OUT/f'fold{fold}/seed2027/joint'
        assert p.read(dest/'STATIC.json')['pass_static']
        save(P/f'FOLD{fold}_COMPLETE.json',dict(utc=now(),checkpoint_sha256=p.sha(dest/'final.pt'),status=p.read(dest/'status.json')))
        gc.collect();torch.cuda.empty_cache()
    save(OUT/'queue_status.json',dict(state='complete',runs=4,fold0_reused=True,utc=now()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');args=a.parse_args()
    try:main(args.preflight)
    except Exception as error:
        save(P/'ERROR.json',dict(utc=now(),error=str(error),traceback=traceback.format_exc()));raise
