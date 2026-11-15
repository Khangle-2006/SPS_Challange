"""Read-only source extraction with frozen manifest, receipts and bounded CPU output."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,fcntl,traceback,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from portable_roi import extract,read,save,sha
from clock_contract import align
P=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def size(p):return sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
def guard(server, fresh=False):
    receipt=read(P/'LAUNCH_BUDGET.json')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(receipt['utc'])).total_seconds()
    if fresh:assert -60<=age<=3600, 'fresh cross-server budget receipt required'
    if server=='P100':
        sys.path.insert(0,str(P.parent));import common
        return common.guard(fresh=fresh,reserve=70_000_000)
    assert size(P)<20_000_000, 'A100 public output cap20MB; preserve partial'
    # Receipt includes an extra20MB allowance; no stale inventory used at launch.
    assert receipt['T8_bytes']+250_000_000+70_000_000<64_900_000_000
    assert receipt['BTC_bytes']+70_000_000<1_200_000_000
    project=int(subprocess.check_output(['du','-sb','/data/khangle/facevitals'],text=True).split()[0])+receipt['p100_project_bytes']
    assert project+70_000_000<499_900_000_000
    return dict(utc=now(),local_bytes=size(P),project_bytes=project)

def main(server):
    assert server in ['P100','A100']
    lock=(P/('worker_'+server+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (P/('COMPLETE_'+server+'.json')).exists(), 'completed worker must not restart'
    frozen=read(P/'CODE_HASHES.json')
    for name,h in frozen.items():assert sha(P/name)==h, ('code hash',name)
    save(P/('PREFLIGHT_'+server+'.json'),guard(server,True))
    manifest=read(P/'MANIFEST.json');records=[r for r in manifest['records'] if r['server']==server]
    assert len(records)==(119 if server=='P100' else 35)
    for ri,r in enumerate(records):
        assert r['recording'] not in manifest['closed_ubfc_test']
        for variant in ['native','f10_b500']:
            guard(server)
            dest=P/'traces'/r['dataset']/r['recording']/variant;cache=Path(r['cache'])/variant
            receipt=read(cache/'complete.json');meta=cache/'crop/meta.npz';cm=cache/'crop/manifest.json'
            assert sha(meta)==receipt['cache_files']['meta.npz'] and sha(cm)==receipt['cache_files']['manifest.json']
            source=Path(r['source'] if variant=='native' else r['legacy_c5'])
            expected=receipt['source_sha256'] if variant=='native' else receipt['encoding']['video_sha256_at_reuse_or_creation']
            if (dest/'PUBLIC_COMPLETE.json').exists():
                done=read(dest/'PUBLIC_COMPLETE.json');assert sha(dest/'trace.npz')==done['trace_sha256'];continue
            save(P/('STATUS_'+server+'.json'),dict(state='extracting',recording=r['recording'],variant=variant,index=ri+1,total=len(records),pid=os.getpid(),utc=now()))
            assert sha(source)==expected, 'source mismatch'
            ans=extract(source,dest)
            z=np.load(meta);original=z['original_t'];canonical=z['t']
            drop=int(read(cm)['metadata'].get('dropped_final_decoded_frames',0)) if variant=='native' else 0
            with np.load(dest/'trace.npz') as trace:actual_t,x=align(original,canonical,trace['rgb'],drop)
            np.savez_compressed(dest/'clock.npz',original_t=original,canonical_t=canonical)
            save(dest/'PUBLIC_COMPLETE.json',dict(recording=r['recording'],dataset=r['dataset'],person=r['person'],fold=r['fold'],variant=variant,source_sha256=expected,trace_sha256=sha(dest/'trace.npz'),clock_sha256=sha(dest/'clock.npz'),meta_sha256=sha(meta),cache_receipt_sha256=sha(cache/'complete.json'),allowed_tail_drop=drop,duplicate_timestamps=len(original)-len(canonical),frames=len(actual_t),protocol_sha256=sha(P/'PROTOCOL.md')))
    save(P/('COMPLETE_'+server+'.json'),dict(utc=now(),records=len(records),conditions=2,holdout_evaluations=0))
if __name__=='__main__':
    server=sys.argv[1]
    try:main(server)
    except Exception as e:save(P/('ERROR_'+server+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
