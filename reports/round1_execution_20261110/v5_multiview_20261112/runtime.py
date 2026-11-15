import os,sys,json,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;E=P.parent;N=E/'final_priorities_20261111';R=E.parent.parent
sys.path.insert(0,str(N));import padding_queue as old
def now():return datetime.now(timezone.utc).isoformat()
read=old.p.read;save=old.save;sha=old.p.sha;lock=old.lock
def budget(fresh=False):
    a=read(E/'A100_INVENTORY.json');age=(datetime.now(timezone.utc)-datetime.fromisoformat(a['measured_utc'])).total_seconds()
    assert -60<=age and (not fresh or age<=3600),'FRESH_CROSS_SERVER_INVENTORY_REQUIRED'
    base=old.p.guard.storage_guard(read(old.p.INDEX),old.p.ORIGINAL.parent,old.p.V3)['total_bytes']
    used=base+old.size(old.V)+old.size('/data_pool/khangle/facevitals/reports/round1_v4_20261105')+a['a100_v4_bytes']+old.size(R/'reports/round1_bayes_20261108')+old.size(E)
    own=old.size(P);alloc=read(P/'STORAGE_PLAN.json')['allocation_bytes'] if (P/'STORAGE_PLAN.json').exists() else 30_000_000
    remain=max(0,alloc-own);project=sum(int(subprocess.check_output(['du','-sb',str(d)],text=True).split()[0]) for d in [R,Path('/data_pool/khangle/facevitals')])+a['a100_bytes']
    assert own<=alloc and used+remain+420_000_000<84_000_000_000,'STOP_STORAGE_84GB'
    assert project+remain+420_000_000<500_000_000_000,'STOP_PROJECT500GB'
    return dict(utc=now(),scope_actual_bytes=used,project_actual_bytes=project,allocation_bytes=alloc,own_bytes=own,remaining_reserved_bytes=remain,reserve_bytes=420_000_000,a100_age_seconds=age,a100_sha256=sha(E/'A100_INVENTORY.json'))
def hard_stop():old.hard_stop()
def gpu():
    import time
    while True:
        hard_stop();samples=[]
        for _ in range(3):
            line=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0];idx,free,util=map(int,line.split(','));assert idx==0;samples.append(dict(index=idx,free_MiB=free,utilization=util));time.sleep(2)
        save(P/'GPU_CHECK.json',dict(utc=now(),samples=samples,sharing_authorized=True,required_free_MiB=6000))
        if all(r['free_MiB']>=6000 for r in samples):budget(fresh=True);return
        save(P/'GPU_WAIT.json',dict(utc=now(),samples=samples));time.sleep(30)
