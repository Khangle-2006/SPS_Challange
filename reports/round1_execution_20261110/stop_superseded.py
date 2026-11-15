"""Stop only the inspected superseded sweep process group; preserve all outputs."""
import os, signal, json, time
from pathlib import Path
from datetime import datetime, timezone
root=Path('/mnt/samsung_t5/khangle/facevitals/reports/round1_bayes_20261108/domain_sweep_v2_20261110')
worker=3577370
supervisor=3577369
cmd=Path(f'/proc/{worker}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(root/'sweep.py') in cmd
assert os.getpgid(worker)==supervisor and os.getpgid(supervisor)==supervisor
status=json.loads((root/'runs/fold0/status.json').read_text())
receipt=dict(utc=datetime.now(timezone.utc).isoformat(),reason='User supersedes recipe and dependency order: AdaBN and new C5 prime before exactly five amended runs; not outcome selection.',worker=worker,supervisor=supervisor,last_status=status,checkpoint_present=(root/'runs/fold0/last.pt').exists(),command=cmd)
(root/'STOPPED_BY_USER_REPLAN_20261110.json').write_text(json.dumps(receipt,indent=2))
os.killpg(supervisor,signal.SIGTERM)
for _ in range(30):
 if not Path(f'/proc/{worker}').exists():break
 time.sleep(1)
receipt['worker_exited']=not Path(f'/proc/{worker}').exists()
(root/'STOPPED_BY_USER_REPLAN_20261110.json').write_text(json.dumps(receipt,indent=2))
assert receipt['worker_exited'],'Inspect manually; do not blindly kill further processes'
(root/'QUEUE_STATUS.json').write_text(json.dumps(dict(state='stopped_user_replan',utc=receipt['utc'],disposition='STOPPED_BY_USER_REPLAN_20261110.json'),indent=2))
print(json.dumps(receipt))
