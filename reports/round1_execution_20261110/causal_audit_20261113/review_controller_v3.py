"""Bounded current-task report controller; never starts/restarts compute jobs."""
from common_causal import *
def render():
 subprocess.run([sys.executable,str(C/'report_causal.py')],check=True)
import signal,traceback
def main():
 lease=lock('review.lock');assert not (C/'REVIEW_V3_STARTED.json').exists();save(C/'REVIEW_V3_STARTED.json',dict(pid=os.getpid(),utc=now(),code_sha256=sha(__file__)))
 last=None
 while True:
  errors=[n for n in ['physical_v3/LADDER_ERROR.json','METADATA_ERROR.json','ROUTE_ERROR.json'] if (C/n).exists()]
  complete=[n for n in ['BATCH_COMPLETE.json','ROUTE_COMPLETE.json','METADATA_COMPLETE.json','physical_v3/LADDER_COMPLETE.json','FRONTEND_COMPARISON_COMPLETE.json'] if (C/n).exists()]
  spatial=(P/'report_spatial/COMPLETE.json').exists();stage_count=len(list((C/'physical_v3/ladder').glob('*/*/COMPLETE.json')));prefix_count=len(list((C/'prefix_repair/ladder').glob('*/*/COMPLETE.json')));signature=(tuple(errors),tuple(complete),spatial,stage_count,prefix_count)
  if signature!=last:render();last=signature
  save(C/'REVIEW_STATUS.json',dict(utc=now(),completed=complete,errors=errors,spatial_complete=spatial,ladder_stage_receipts=stage_count,paired_capture_received=False,no_job_restart=True))
  if errors:save(C/'REVIEW_NEEDS_INSPECTION.json',dict(utc=now(),errors=errors,no_automatic_restart=True));break
  if len(complete)==5 and spatial:
   save(C/'COMPUTE_REVIEW_READY.json',dict(utc=now(),scientific_review_pending=True,paired_capture_pending=True,FREEZE_changed=False));break
  deadline=datetime.fromisoformat(read(C/'PROTOCOL.json')['deadline_utc'])
  if datetime.now(timezone.utc)>=deadline:
   stopped=[]
   for filename,expected in [('PHYSICAL_V3_LAUNCH.json','ladder_physical_v3.py'),('METADATA_LAUNCH.json','audit_metadata.py'),('ROUTE_LAUNCH.json','cross_route.py'),('LOCALIZATION_ALL_LAUNCH.json','localization_all.py'),('PREFIX_REPAIR_LAUNCH.json','repair_prefix.py')]:
    if not (C/filename).exists():continue
    pid=read(C/filename)['pid'];cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists() and str(C/expected).encode() in cmd.read_bytes().split(b'\x00'):
     assert os.getpgid(pid)==pid;os.killpg(pid,signal.SIGTERM);stopped.append(pid)
   save(C/'TWO_DAY_BOUND.json',dict(utc=now(),terminated_own_audit_groups_only=stopped,AB_untouched=True,partial_evidence_preserved=True));render();break
  time.sleep(20)
if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'REVIEW_V3_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
