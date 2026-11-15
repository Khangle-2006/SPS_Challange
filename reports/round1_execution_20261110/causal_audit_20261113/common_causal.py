import os,sys,json,hashlib,subprocess,fcntl,time
from pathlib import Path
from datetime import datetime,timezone
C=Path(__file__).resolve().parent;E=C.parent;P=E/'inference_repair_20261113';R=E.parent.parent
sys.path.insert(0,str(P));import common_repair as repair
read=repair.read;save=repair.save;sha=repair.sha;now=repair.now

def lock(name):
 f=(C/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

def stable_repair_budget(fresh=True):
 for attempt in range(4):
  try:return repair.budget(fresh)
  except subprocess.CalledProcessError as e:
   if not isinstance(e.cmd,list) or e.cmd[:1]!=['du'] or e.returncode!=1 or attempt==3:raise
  except FileNotFoundError as e:
   if not str(e.filename).endswith('.tmp') or attempt==3:raise
  time.sleep(.5)

def budget(fresh=True):
 a=stable_repair_budget(fresh);used=sum(f.stat().st_size for f in C.rglob('*') if f.is_file());remaining=max(0,1_800_000_000-used)
 assert used<=1_800_000_000,'CAUSAL_ALLOCATION_1.8GB'
 assert a['scope_actual_bytes']+a['remaining_reserved_bytes']+a['repair_remaining']+remaining+420_000_000<84_000_000_000,'COMMON84GB'
 assert a['project_actual_bytes']+remaining+a['repair_remaining']+420_000_000<500_000_000_000,'PROJECT500GB'
 return dict(**a,causal_actual_bytes=used,causal_remaining_bytes=remaining)

def check_deadline():
 assert datetime.now(timezone.utc)<datetime.fromisoformat(read(C/'PROTOCOL.json')['deadline_utc']),'TWO_DAY_BOUND'

def arrsha(a):
 import numpy as np
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
