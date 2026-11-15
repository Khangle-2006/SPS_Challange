"""Download the reviewed MPU sample and SCAMPS official release, without extraction."""
import base64, concurrent.futures, fcntl, hashlib, json, shutil, time
from pathlib import Path
import requests

ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets')
RESERVE=180*1024**3  # Includes space for remaining MCD acquisition and shared-disk headroom.

def save(path,obj):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def hashes(path):
    a=hashlib.sha256();b=hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):a.update(chunk);b.update(chunk)
    return a.hexdigest(),b.hexdigest()

def run_dataset(name):
    root=ROOT/name
    plan=json.loads((root/'acquisition_plan.json').read_text())
    lock=(root/'.download.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'dataset':name,'state':'running','files_total':len(plan['files']),
            'bytes_total':sum(x.get('size',0) for x in plan['files']),
            'files_complete':0,'bytes_complete':0,'started':time.time(),'errors':[]}
    save(root/'download_status.json',status)
    for item in plan['files']:
        target=root/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        marker=target.with_name(target.name+'.verified.json');part=target.with_name(target.name+'.part')
        status.update(current_file=item['path'],current_partial_bytes=0)
        for attempt in range(12):
            try:
                if marker.exists() and target.exists():
                    v=json.loads(marker.read_text())
                    if target.stat().st_size==v['size'] and (not item.get('size') or v['size']==item['size']):break
                if target.exists():raise RuntimeError('Unverified existing file, preserving it')
                offset=part.stat().st_size if part.exists() else 0
                expected=item.get('size')
                if expected and offset>expected:raise RuntimeError('Oversize partial, preserving it')
                if not expected or offset<expected:
                    url=item['url']+('&' if '?' in item['url'] else '?')+'acquisition='+str(int(time.time()))
                    with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120)) as r:
                        r.raise_for_status()
                        if 'text/html' in r.headers.get('Content-Type',''):raise RuntimeError('Received HTML instead of data')
                        if offset and r.status_code!=206:offset=0
                        if r.status_code==206 and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise RuntimeError('Range mismatch')
                        if not expected:
                            cr=r.headers.get('Content-Range','');expected=int(cr.split('/')[-1]) if '/' in cr else int(r.headers.get('Content-Length',0)) or None
                        last=0;written=offset
                        with part.open('ab' if offset else 'wb') as f:
                            for chunk in r.iter_content(4*1024*1024):
                                if not chunk:continue
                                while shutil.disk_usage(root).free < RESERVE:
                                    status.update(state='paused_low_disk',free_bytes=shutil.disk_usage(root).free,updated=time.time());save(root/'download_status.json',status);time.sleep(30)
                                status['state']='running';f.write(chunk);written+=len(chunk)
                                if time.time()-last>10:
                                    status.update(current_partial_bytes=written,updated=time.time());save(root/'download_status.json',status);last=time.time()
                if expected and part.stat().st_size!=expected:raise RuntimeError('Size mismatch')
                status.update(state='verifying',current_partial_bytes=part.stat().st_size,updated=time.time());save(root/'download_status.json',status)
                sha,md5=hashes(part)
                if item.get('md5') and md5!=item['md5']:
                    part.rename(part.with_name(part.name+'.badmd5.'+str(int(time.time()))));raise RuntimeError('MD5 mismatch, quarantined')
                part.replace(target);save(marker,{'size':target.stat().st_size,'sha256':sha,'md5':md5,'publisher_md5_checked':bool(item.get('md5')),'source':item['url']})
                break
            except Exception as e:
                print(json.dumps({'dataset':name,'file':item['path'],'attempt':attempt+1,'error':str(e)}),flush=True)
                if attempt==11:
                    status['errors'].append({'file':item['path'],'error':str(e)});save(root/'download_status.json',status)
                else:time.sleep(min(60,5*(attempt+1)))
        if target.exists() and marker.exists():
            status['files_complete']+=1;status['bytes_complete']+=target.stat().st_size
        status.update(state='running',updated=time.time());save(root/'download_status.json',status)
        print(json.dumps(status),flush=True)
    status.update(state='complete' if not status['errors'] else 'failed',finished=time.time(),current_file=None,current_partial_bytes=0)
    save(root/'download_status.json',status);print(json.dumps(status),flush=True)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_dataset,n) for n in ['MPU-rPPG_sample','SCAMPS']]
        for job in jobs:job.result()
