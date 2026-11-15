"""User-directed scheduling change only: finish unchanged frozen UBFC controls."""
import os,sys,json,hashlib,fcntl,time,traceback
from pathlib import Path
CODE=Path('/data_pool/khangle/facevitals/code/t8_oval_training_20261031')
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/t8_oval_20261031')
sys.path.insert(0,str(CODE))
import t8_run as run
from t8_plan import read,write,sha,code_fingerprint,SEEDS

def storage_guard(index,output):
    roots=[ROOT,Path(read(Path(index['roots']['MCD'])/'status.json')['legacy_root'])]
    roots += [p for p in ROOT.parent.iterdir() if p.is_dir() and p.name.startswith('t8_') and ('polarity' in p.name or 'label_diagnostic' in p.name)]
    unique=[]
    for p in sorted(set(p.resolve() for p in roots),key=lambda x:len(x.parts)):
        if not any(q==p or q in p.parents for q in unique):unique.append(p)
    sizes={str(r):sum(p.stat().st_size for p in r.rglob('*') if p.is_file()) for r in unique}
    u=read(Path(index['roots']['UBFC2'])/'status.json');origin=u['bytes']+u['legacy_bytes']
    assert sum(sizes.values())+origin<=65_000_000_000-100_000_000,'T8 whole65GB guard'
    return dict(counted_roots_bytes=sizes,A100_UBFC_origin_bytes_including_legacy=origin,total_bytes=sum(sizes.values())+origin,approved_T8_cap_bytes=65_000_000_000,operational_wrapper_sha256=sha(__file__))

def main():
    out=ROOT/'runs'
    assert not Path('/proc/2968104').exists(),'Original queue still alive'
    assert (out/'STOP_JOINT_BY_USER_20261102.json').exists()
    lock=(out/'queue.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=read(CODE/'FROZEN_SHA256.json');assert code_fingerprint(CODE)==frozen
    index=read(ROOT/'training_index.json');index_sha=sha(ROOT/'training_index.json')
    assert index_sha=='12a87b829ab5e158b430c67701fa0dd978c9a3d592e0060da3e876069108d505'
    assert all(fp==read(CODE/'SOURCE_CACHE_SHA256.json') for fp in index['cache_fingerprints'].values())
    # Preserve all earlier states/checkpoints. Do not enter any joint run_one.
    binding=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reason='Independent PPG convention diagnostic; user stopped unharmonized joint, not outcome selection',only_arm='ubfc_only',method_changes=False,frozen_code_sha256=frozen,wrapper_sha256=sha(__file__),index_sha256=index_sha)
    dest=out/'CONTROL_ONLY_CONTINUATION_20261102.json';assert not dest.exists(),'Continuation already launched; inspect first'
    write(dest,binding)
    run.storage_guard=storage_guard
    storage_guard(index,out)
    run.resource_guard()
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0)
    for fold in range(5):
        for seed in SEEDS:
            status=out/f'fold{fold}'/f'seed{seed}'/'ubfc_only'/'status.json'
            if status.exists():
                assert read(status)['state']=='complete','Unexpected partial control; preserve and inspect'
                run.run_one(index,out,fold,seed,'ubfc_only',index_sha,frozen)
                continue
            write(out/'queue_status.json',dict(state='running_controls_only',pid=os.getpid(),fold=fold,seed=seed,arm='ubfc_only',index_sha256=index_sha,joint_disposition='stopped_by_user_polarity_error'))
            run.run_one(index,out,fold,seed,'ubfc_only',index_sha,frozen)
    write(out/'queue_status.json',dict(state='controls_complete_joint_stopped',controls_complete=15,original_joint_disposition='polarity_unharmonized_stopped_by_user',index_sha256=index_sha))
    print('ALL15_UBFC_CONTROLS_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/'runs'/'CONTROL_CONTINUATION_ERROR_20261102.json',dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
