"""Explicitly audited recovery of25missing outputs after atomic inventory rename/du race."""
from common_causal import *
import orientation_audit_v2 as orient
import traceback

def main():
 lease=orient.lock('orientation1.lock');assert not (P/'ORIENTATION1_RECOVERY02_STARTED.json').exists();jobs=repair.manifest()[1::2];done=[];partials=[]
 for j in jobs:
  d=P/'orientation'/j['key']
  if (d/'COMPLETE.json').exists():
   a=read(d/'COMPLETE.json');assert a['source_sha256']==j['source_sha256'] and sha(d/'landmarks.npz')==a['landmarks_sha256'];done.append(j['key'])
  elif d.exists():partials.append(j['key'])
 assert len(done)==161 and partials==[];assert not Path('/proc/557946').exists();b=budget(True)
 save(P/'ORIENTATION1_RECOVERY02_STARTED.json',dict(utc=now(),pid=os.getpid(),verified_completed=done,missing=25,partials=partials,budget=b,source_sha256=sha(P/'orientation_audit_v2.py'),protocol_sha256=sha(P/'PROTOCOL.md'),reason='du raced A100_INVENTORY.json.repair_tmp rename; full successful measurement retried, no cap waiver'))
 for j in jobs:
  if j['key'] in done:continue
  check_deadline();save(P/'ORIENTATION1_STATUS.json',dict(utc=now(),pid=os.getpid(),completed=len(done),total=len(jobs),current=j['key'],reviewed_recovery=2));orient.one(j);done.append(j['key']);budget(False)
 save(P/'ORIENTATION1_COMPLETE.json',dict(utc=now(),completed=len(done),total=len(jobs),explicit_recovery=2))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'ORIENTATION1_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
