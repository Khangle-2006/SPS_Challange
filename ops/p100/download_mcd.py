"""Pinned, resumable MCD-rPPG download. No credentials required."""
import argparse, concurrent.futures, fcntl, hashlib, json, os, shutil, time, subprocess, threading
from pathlib import Path, PurePosixPath
import requests

REPO = 'kyegorov/mcd_rppg'
REV = 'bf59b3cd1414c2a510d83c3125f16bb87d469ebb'
BASE = 'https://huggingface.co'
PROJECT_ROOTS = ['/data_pool/khangle/facevitals', '/mnt/samsung_t5/khangle/facevitals']
PROJECT_LIMIT = 300_000_000_000  # P100 allowance; total server budget is documented separately.
DISK_RESERVE = 180 * 1024**3
_guard_lock = threading.Lock()
_guard_checked = 0

def project_bytes():
    # du does not follow the dataset symlinks, so the two roots are not double counted.
    result = subprocess.check_output(['du', '-s', '-B1', *PROJECT_ROOTS], text=True)
    return sum(int(line.split()[0]) for line in result.splitlines())

def storage_guard(root):
    global _guard_checked
    with _guard_lock:
        if time.monotonic() - _guard_checked < 30:
            return
        while True:
            used = project_bytes()
            free = shutil.disk_usage(root).free
            paused = used >= PROJECT_LIMIT - 1_000_000_000 or free < DISK_RESERVE
            atomic_json(root / 'storage_guard_status.json', {
                'state': 'paused_storage_budget' if paused else 'ok',
                'p100_project_bytes': used, 'p100_limit_bytes': PROJECT_LIMIT,
                'disk_free_bytes': free, 'disk_reserve_bytes': DISK_RESERVE,
                'updated': time.time()})
            if not paused:
                _guard_checked = time.monotonic()
                return
            print('Download paused by shared-storage budget guard', flush=True)
            time.sleep(30)

def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def get_manifest(root):
    url = f'{BASE}/api/datasets/{REPO}/tree/{REV}?recursive=true&expand=false&limit=1000'
    files = []
    while url:
        r = requests.get(url, timeout=90); r.raise_for_status()
        for item in r.json():
            if item['type'] != 'file': continue
            p = PurePosixPath(item['path'])
            if p.is_absolute() or '..' in p.parts: raise ValueError('Unsafe manifest path')
            files.append({'path': str(p), 'size': item['size'], 'lfs_sha256': item.get('lfs', {}).get('oid')})
        url = r.links.get('next', {}).get('url')
    if not files: raise RuntimeError('Empty manifest')
    data = {'repo': REPO, 'revision': REV, 'total_bytes': sum(f['size'] for f in files), 'files': files}
    atomic_json(root / 'download_manifest.json', data)
    return data

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def download(root, item):
    target = root / item['path']; target.parent.mkdir(parents=True, exist_ok=True)
    mark = target.with_name(target.name + '.verified.json')
    if target.exists() and target.stat().st_size == item['size'] and mark.exists(): return item['path']
    part = target.with_name(target.name + '.part')
    for attempt in range(10):
        try:
            if target.exists() and not part.exists():
                if target.stat().st_size == item['size']:
                    actual = digest(target)
                    if not item['lfs_sha256'] or actual == item['lfs_sha256']:
                        atomic_json(mark, {'sha256': actual, 'size': item['size'], 'revision': REV}); return item['path']
                    raise RuntimeError('Existing complete file checksum mismatch; preserving it')
                raise RuntimeError('Unexpected existing file; preserving it')
            offset = part.stat().st_size if part.exists() else 0
            if offset > item['size']: raise RuntimeError('Oversize partial file')
            if offset < item['size']:
                # Fresh cache-busting query avoids stale expired CDN signed redirects.
                url = f"{BASE}/datasets/{REPO}/resolve/{REV}/{item['path']}?download=true&attempt={int(time.time())}"
                with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {}, stream=True, timeout=(30,120)) as r:
                    r.raise_for_status()
                    if offset and r.status_code != 206: offset = 0
                    if r.status_code == 206 and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Range response mismatch')
                    with part.open('ab' if offset else 'wb') as f:
                        for chunk in r.iter_content(2*1024*1024):
                            if chunk:
                                storage_guard(root)
                                f.write(chunk)
            if part.stat().st_size != item['size']: raise RuntimeError('Incomplete size')
            actual = digest(part)
            if item['lfs_sha256'] and actual != item['lfs_sha256']:
                part.rename(part.with_name(part.name + f'.badsha.{int(time.time())}'))
                raise RuntimeError('Checksum mismatch; partial quarantined')
            part.replace(target)
            atomic_json(mark, {'sha256': actual, 'size': item['size'], 'revision': REV})
            return item['path']
        except Exception as e:
            print(json.dumps({'retry':attempt+1,'path':item['path'],'error':str(e)}), flush=True)
            if attempt==9: raise
            time.sleep(min(60, 3*(attempt+1)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--metadata-only',action='store_true');args=ap.parse_args()
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.download.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=get_manifest(root)
    files=[f for f in manifest['files'] if not f['path'].startswith('video/')] if args.metadata_only else manifest['files']
    required=sum(max(0,f['size']-((root/f['path']).stat().st_size if (root/f['path']).exists() else ((root/(f['path']+'.part')).stat().st_size if (root/(f['path']+'.part')).exists() else 0))) for f in files)
    free=shutil.disk_usage(root).free
    if free < required + DISK_RESERVE: raise RuntimeError('Insufficient shared-disk reserve')
    if project_bytes() + required > PROJECT_LIMIT - 1_000_000_000:
        raise RuntimeError('Projected download exceeds P100 project budget')
    status={'state':'running','repo':REPO,'revision':REV,'files_total':len(files),'bytes_total':sum(f['size'] for f in files),'files_complete':0,'bytes_complete':0,'metadata_only':args.metadata_only,'started':time.time()}
    atomic_json(root/'download_status.json',status)
    # Labels and timing metadata first, then video. Two downloads limit server load.
    files.sort(key=lambda f:(f['path'].startswith('video/'),f['path']))
    completed_bytes=0;errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(download,root,f):f for f in files}
        for future in concurrent.futures.as_completed(pending):
            item=pending[future]
            try:future.result();status['files_complete']+=1;completed_bytes+=item['size']
            except Exception as e:errors.append({'path':item['path'],'error':str(e)})
            status.update(bytes_complete=completed_bytes,updated=time.time(),errors=errors)
            atomic_json(root/'download_status.json',status)
            if status['files_complete']%25==0:print(json.dumps(status),flush=True)
    status.update(state='failed' if errors else 'complete',finished=time.time());atomic_json(root/'download_status.json',status)
    print(json.dumps(status),flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
