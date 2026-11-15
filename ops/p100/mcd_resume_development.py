"""Resume only frozen dev60; download audit30 first, retain all raw."""
import json,os,signal,time,fcntl,concurrent.futures,subprocess,shutil
from pathlib import Path
from download_mcd import download,atomic_json,project_bytes,PROJECT_LIMIT,DISK_RESERVE
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
OUT=ROOT.parent.parent/'mapping_audit_20261030'
plan=json.loads((OUT/'development_plan.json').read_text())
manifest={f['path']:f for f in json.loads((ROOT/'download_manifest.json').read_text())['files']}
audit_paths=[r['video'] for r in plan['audit_records']]
all_paths=audit_paths+[r['video'] for r in plan['development_records'] if r['video'] not in audit_paths]
required=sum(max(0,manifest[p]['size']-((ROOT/p).stat().st_size if (ROOT/p).exists() else (ROOT/(p+'.part')).stat().st_size if (ROOT/(p+'.part')).exists() else 0)) for p in all_paths)
assert project_bytes()+required<PROJECT_LIMIT-1_000_000_000
assert shutil.disk_usage(ROOT).free>required+DISK_RESERVE
old=Path('/proc/3168721')
if old.exists():
    assert old.stat().st_uid==os.getuid() and b'download_mcd.py' in (old/'cmdline').read_bytes()
    os.kill(3168721,signal.SIGKILL);time.sleep(1)
lock=(ROOT/'.download.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status={'state':'running_development_subset','revision':plan['pinned_revision'],'files_total':len(all_paths),
        'bytes_total':sum(manifest[p]['size'] for p in all_paths),'completed':0,'errors':[],
        'raw_deletion_enabled':False,'started':time.time(),'full_dataset_download_resumed':False}
atomic_json(OUT/'development_download_status.json',status)
for stage,paths in [('audit30',audit_paths),('remaining_development',[p for p in all_paths if p not in audit_paths])]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(download,ROOT,manifest[p]):p for p in paths}
        for future in concurrent.futures.as_completed(futures):
            p=futures[future]
            try:future.result();status['completed']+=1
            except Exception as e:status['errors'].append({'path':p,'error':str(e)})
            status.update(stage=stage,updated=time.time());atomic_json(OUT/'development_download_status.json',status)
    if status['errors']:raise RuntimeError(status['errors'])
    if stage=='audit30':
        log=Path('/data_pool/khangle/facevitals/logs/mcd_packet_expanded.log')
        with log.open('ab',buffering=0) as f:
            process=subprocess.Popen(['/data_pool/khangle/facevitals/envs/storage_pilot/bin/python','-u',
                '/data_pool/khangle/facevitals/code/mcd_packet_audit.py','--plan',str(OUT/'development_plan.json')],
                stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        status['expanded_audit_pid']=process.pid
status.update(state='complete_development_subset',finished=time.time());atomic_json(OUT/'development_download_status.json',status)
full=json.loads((ROOT/'download_status.json').read_text())
full.update(state='paused_outside_development_subset',note='Development60 acquisition completed separately. Full-dataset file/byte counts are historical; see development_download_status.json.',updated=time.time())
atomic_json(ROOT/'download_status.json',full)
print(json.dumps(status),flush=True)
