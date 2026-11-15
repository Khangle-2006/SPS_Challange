"""Explicitly reviewed continuation after inventory JSON publication race, not a retry loop."""
from orientation_audit_v2 import *

def recover():
 lease=lock('orientation1.lock');assert not (P/'ORIENTATION1_RECOVERY_STARTED.json').exists();audit=read(P/'RECOVERY_AUDIT_01.json');assert audit['partial']==[];budget(True);jobs=manifest()[1::2];done=[]
 save(P/'ORIENTATION1_RECOVERY_STARTED.json',dict(utc=now(),pid=os.getpid(),audit_sha256=sha(P/'RECOVERY_AUDIT_01.json'),extractor_sha256=sha(P/'orientation_audit_v2.py'),unchanged_protocol_sha256=sha(P/'PROTOCOL.md'),no_recipe_or_model_change=True))
 for j in jobs:
  d=P/'orientation'/j['key']
  if (d/'COMPLETE.json').exists():
   a=read(d/'COMPLETE.json');assert a['source_sha256']==j['source_sha256'] and sha(d/'landmarks.npz')==a['landmarks_sha256'];done.append(j['key']);continue
  assert not d.exists(),'UNREVIEWED_PARTIAL'
  save(P/'ORIENTATION1_STATUS.json',dict(utc=now(),pid=os.getpid(),completed=len(done),total=len(jobs),current=j['key'],explicit_reviewed_recovery=True));one(j);done.append(j['key']);budget()
 save(P/'ORIENTATION1_COMPLETE.json',dict(utc=now(),completed=len(done),total=len(jobs),recovery_audit_sha256=sha(P/'RECOVERY_AUDIT_01.json')))
if __name__=='__main__':
 try:recover()
 except Exception as e:save(P/'ORIENTATION1_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
