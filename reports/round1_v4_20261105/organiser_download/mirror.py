"""Mirror linked public Drive videos and reconcile worksheet rows without copying clinical columns."""
import argparse, base64, csv, fcntl, hashlib, json, os, re, shutil, subprocess, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import requests

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/samsung_t5/khangle/facevitals')
POOL=Path('/data_pool/khangle/facevitals')
DEST=ROOT/'datasets/FaceVitals_BTC_20261107'
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,value):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
def sizes(p): return int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0])
def url(row):
    fid=row['drive_id']; assert re.fullmatch('[A-Za-z0-9_-]+',fid)
    params=dict(id=fid,export='download',confirm='t')
    key=parse_qs(urlparse(row['url']).query).get('resourcekey')
    if key: params['resourcekey']=key[0]
    return params
def probe(row):
    with requests.get('https://drive.usercontent.google.com/download',params=url(row),headers={'Range':'bytes=0-63','Accept-Encoding':'identity'},stream=True,timeout=(12,32)) as r:
        r.raise_for_status()
        m=Message();m['Content-Disposition']=r.headers.get('Content-Disposition','')
        filename=m.get_filename()
        if not filename or Path(filename).name!=filename or Path(filename).suffix.lower() not in {'.mov','.mp4','.avi','.m4v','.mkv'}:raise ValueError('Drive returned no safe video filename')
        if r.status_code==206:
            total=int(re.fullmatch(r'bytes 0-\d+/(\d+)',r.headers['Content-Range']).group(1))
        elif r.status_code==200:
            total=int(r.headers['Content-Length'])
        else:raise ValueError('Unexpected range response')
        if total<128:raise ValueError('Invalid size')
        return dict(filename=filename,size=total,etag=r.headers.get('ETag'),server_hash=r.headers.get('x-goog-hash'))
def retry(fn,n=3):
    for i in range(n):
        try:return fn()
        except (OSError,ValueError,RuntimeError,requests.RequestException) as ex:
            if i==n-1:raise
            time.sleep(2**i)
def guard(remaining,a100):
    used=sizes(ROOT)+sizes(POOL)
    if used+remaining+1_000_000_000>300_000_000_000 or used+remaining+a100+1_000_000_000>500_000_000_000:
        raise RuntimeError(f'Storage guard: P100 {used}, projected extra {remaining}, A100 {a100}')
    if shutil.disk_usage(DEST).free<remaining+2_000_000_000:raise RuntimeError('Insufficient volume space')
    return used
def source_md5(value):
    m=re.search(r'(?:^|,\s*)md5=([^, ]+)',value or '')
    return base64.b64decode(m[1]).hex() if m else None
def fetch(fid,row,meta):
    # Drive-ID subdirectories also preserve distinct files with the same source basename.
    folder=DEST/row['group']/fid;folder.mkdir(parents=True,exist_ok=True)
    dest=folder/meta['filename'];tmp=folder/(meta['filename']+'.part')
    receipt=HERE/'receipts'/(fid+'.json')
    def hashes(p):
        sha=hashlib.sha256();md=hashlib.md5()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):sha.update(block);md.update(block)
        return sha.hexdigest(),md.hexdigest()
    if dest.exists() and receipt.exists():
        rec=read(receipt)
        if dest.stat().st_size!=meta['size'] or hashes(dest)[0]!=rec['sha256']:raise ValueError('Existing verified file changed')
        return rec
    if not dest.exists():
        offset=tmp.stat().st_size if tmp.exists() else 0
        if offset>meta['size']:raise ValueError('Partial larger than source')
        if offset<meta['size']:
            headers={'Accept-Encoding':'identity','Range':f'bytes={offset}-{meta["size"]-1}'}
            with requests.get('https://drive.usercontent.google.com/download',params=url(row),headers=headers,stream=True,timeout=(15,60)) as r:
                r.raise_for_status()
                if 'text/' in r.headers.get('Content-Type',''):raise ValueError('Drive returned text or quota page')
                if meta['etag'] and r.headers.get('ETag') and meta['etag']!=r.headers['ETag']:raise ValueError('Source ETag changed')
                if r.status_code==206:
                    if r.headers.get('Content-Range')!=f'bytes {offset}-{meta["size"]-1}/{meta["size"]}':raise ValueError('Range/total mismatch')
                elif r.status_code==200:offset=0
                else:raise ValueError('Unexpected download status')
                with tmp.open('ab' if offset else 'wb') as f:
                    for block in r.iter_content(2**20):
                        if f.tell()+len(block)>meta['size']:raise ValueError('Payload exceeds declared source size')
                        f.write(block)
    candidate=dest if dest.exists() else tmp
    if candidate.stat().st_size!=meta['size']:raise ValueError('Truncated media')
    sha,md=hashes(candidate);expected=source_md5(meta['server_hash'])
    if expected and md!=expected:raise ValueError('Source MD5 mismatch')
    raw=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height:format=duration','-of','json',str(candidate)],timeout=60,text=True)
    geom=json.loads(raw);stream=geom['streams'][0];duration=float(geom['format']['duration'])
    if duration<=0 or min(stream['width'],stream['height'])<=0:raise ValueError('Invalid video')
    if candidate==tmp:tmp.rename(dest)
    rec=dict(drive_id=fid,filename=meta['filename'],bytes=meta['size'],sha256=sha,md5=md,source_md5=expected,path=str(dest),seconds=duration,**stream)
    save(receipt,rec)
    return rec
def run():
    parser=argparse.ArgumentParser();parser.add_argument('--a100-bytes',required=True,type=int);args=parser.parse_args()
    manifest=read(HERE/'manifest.json');rows=manifest['rows']
    assert len(rows)==295 and {r['group'] for r in rows}=={'Normal','Abnormal'}
    by=defaultdict(list)
    for r in rows:by[r['drive_id']].append(r)
    DEST.mkdir(parents=True,exist_ok=True);(HERE/'receipts').mkdir(exist_ok=True)
    done={};errors={};meta={}
    save(HERE/'status.json',dict(stage='preflight',utc=now(),unique=len(by)))
    with ThreadPoolExecutor(4) as pool:
        tasks={pool.submit(retry,lambda row=v[0]:probe(row)):k for k,v in by.items()}
        for f in as_completed(tasks):
            k=tasks[f]
            try:meta[k]=f.result()
            except Exception as ex:errors[k]=str(ex)
    save(HERE/'source_metadata.json',meta)
    remaining=sum(v['size'] for v in meta.values())
    # A resumed preflight reserves only bytes that do not already occupy this volume.
    for k,v in meta.items():
        d=DEST/by[k][0]['group']/k/v['filename']
        remaining-=min(v['size'],sum(x.stat().st_size for x in [d,d.with_name(d.name+'.part')] if x.is_file()))
    used=guard(remaining,args.a100_bytes)
    save(HERE/'budget.json',dict(p100_bytes=used,a100_bytes=args.a100_bytes,remaining_bytes=remaining,source_bytes=sum(m['size'] for m in meta.values()),utc=now()))
    def status(stage):
        save(HERE/'status.json',dict(stage=stage,utc=now(),verified=len(done),bytes=sum(x['bytes'] for x in done.values()),source_files=len(meta),source_bytes=sum(x['size'] for x in meta.values()),errors=errors))
    status('downloading')
    keys=list(meta)
    for start in range(0,len(keys),24):
        with ThreadPoolExecutor(3) as pool:
            tasks={pool.submit(retry,lambda k=k:fetch(k,by[k][0],meta[k])):k for k in keys[start:start+24]}
            for f in as_completed(tasks):
                k=tasks[f]
                try:done[k]=f.result()
                except Exception as ex:errors[k]=str(ex)
                status('downloading')
        guard(0,args.a100_bytes)
    audit=[]
    for row in rows:
        k=row['drive_id'];rec=done.get(k,{});matched=bool(rec) and Path(rec['filename']).stem.casefold()==row['recording'].casefold()
        audit.append(dict(group=row['group'],sheet_row=row['row'],recording=row['recording'],drive_id=k,shared_rows=len(by[k]),verified=bool(rec),filename_matches=matched,
                          source_filename=rec.get('filename',''),path=rec.get('path',''),bytes=rec.get('bytes',''),sha256=rec.get('sha256',''),error=errors.get(k,'')))
    with (HERE/'reconciliation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=audit[0]);w.writeheader();w.writerows(audit)
    save(HERE/'result.json',dict(rows=len(rows),unique_links=len(by),verified_files=len(done),bytes=sum(x['bytes'] for x in done.values()),
         groups={g:dict(rows=sum(x['group']==g for x in audit),unique_links=len({x['drive_id'] for x in audit if x['group']==g}),matched=sum(x['group']==g and x['filename_matches'] for x in audit)) for g in ['Normal','Abnormal']},
         exceptions=[x for x in audit if not x['filename_matches'] or x['shared_rows']>1],errors=errors))
    status('reconciled')
if __name__=='__main__':
    with (HERE/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as ex:
            save(HERE/'worker_error.json',dict(utc=now(),error=repr(ex)));raise
