"""Verify and retire only our serial preparation workers; preserve partial files."""
import signal,time
from common import *
assert not (P/'CPU_PARALLEL_DISPOSITION.json').exists()
stopped=[]
for receipt,script in [('warm_LAUNCH.json','warm_training.py'),('c5full_LAUNCH.json','c5prime_v2.py')]:
 a=read(P/receipt);pid=a['pid'];proc=Path('/proc')/str(pid)
 if proc.exists():
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(P/script) in cmd and os.getpgid(pid)==pid
  os.killpg(pid,signal.SIGTERM)
  for _ in range(50):
   if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
   time.sleep(.2)
  assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z'
  stopped.append(dict(pid=pid,command=cmd))
preserved=[]
for d in (P/'c5prime').glob('*/*/*'):
 if not d.is_dir() or (d/'COMPLETE.json').exists():continue
 target=P/'preserved_before_parallel'/d.relative_to(P);target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();d.rename(target);preserved.append(str(target))
for f in (P/'training_cache').glob('*/*.npz'):
 r=f.with_suffix('.json')
 if r.exists():assert sha(f)==read(r)['sha256']
 else:
  target=P/'preserved_before_parallel'/f.relative_to(P);target.parent.mkdir(parents=True,exist_ok=True);f.rename(target);preserved.append(str(target))
save(P/'CPU_PARALLEL_DISPOSITION.json',dict(utc=now(),reason='CPU throughput only: same recipes, clocks, folds and outputs; no metric selection',stopped=stopped,preserved_partials=preserved,no_deletion=True))
