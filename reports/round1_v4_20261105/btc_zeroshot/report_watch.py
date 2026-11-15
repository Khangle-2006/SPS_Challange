"""Write dev reports when sufficient new prediction files arrive; no notifications."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import fcntl,time,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
lock=(HERE/'report.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
last=-1;last_time=0
while True:
    n=len(list((HERE/'predictions').glob('*/*/*.json')))
    status=json.loads((HERE/'INFERENCE_STATUS.json').read_text()) if (HERE/'INFERENCE_STATUS.json').exists() else {}
    if n and (n-last>=50 or time.monotonic()-last_time>900 or status.get('state')=='complete'):
        subprocess.run([sys.executable,str(HERE/'summarize_dev.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'control_summary.py')],check=True)
        last=n;last_time=time.monotonic()
    if status.get('state')=='complete':break
    time.sleep(60)
