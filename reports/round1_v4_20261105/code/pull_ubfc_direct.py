"""Run on P100: copy verified raw files directly from A100, preserve source."""
import fcntl,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).parent
DST=Path('/data_pool/khangle/facevitals/datasets/UBFC-rPPG')
SRC='/data/khangle/facevitals/datasets/UBFC-rPPG'
SSH='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile='+shlex.quote(str(ROOT/'known_hosts'))+' -o ConnectTimeout=15'
lock=(ROOT/'direct.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def save(s):
 t=ROOT/'STATUS.tmp';t.write_text(json.dumps(s,indent=2));os.replace(t,ROOT/'STATUS.json')
state=dict(state='copying',pid=os.getpid(),source=SRC,destination=str(DST),total_files=len(rows),verified=[],source_retained=True)
save(state)
try:
 for row in rows:
  dest=DST/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   assert dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256'], 'Existing conflict '+str(dest)
  else:
   part=dest.with_name(dest.name+'.transfer_part_20261105')
   subprocess.run(['nice','-n','15','rsync','--append-verify','--inplace','--timeout=120','--info=progress2','-e',SSH,'khangle@100.82.42.48:'+SRC+'/'+row['path'],str(part)],check=True)
   assert part.stat().st_size==row['bytes'] and sha(part)==row['sha256'], 'Hash mismatch '+str(part)
   assert not dest.exists();part.rename(dest)
  state['verified'].append(row);save(state)
  print('VERIFIED',len(state['verified']),len(rows),row['path'],flush=True)
 state.update(state='complete',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save(state)
 print('COMPLETE',len(rows),sum(x['bytes'] for x in rows),flush=True)
except BaseException as e:
 state.update(state='error',error=repr(e));save(state);raise
