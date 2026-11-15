"""Sequential conversion, two CPU threads; stop on any verification failure."""
import json, subprocess, sys
from pathlib import Path
ROOT=Path('/data/khangle/facevitals')
out=ROOT/'reports/ubfc2_ffv1_20261030';out.mkdir(exist_ok=True)
paths=sorted((ROOT/'datasets/UBFC-rPPG/DATASET_2').glob('subject*/vid.avi'),key=lambda p:int(p.parent.name[7:]))
assert len(paths)==42
status={'state':'running','total':len(paths),'completed':0,'saved_bytes':0,'records':[]}
for p in paths:
    report=out/p.parent.name/'ffv1_pilot.json'
    if report.exists() and json.loads(report.read_text()).get('state')=='replaced_verified':
        r=json.loads(report.read_text())
    else:
        rc=subprocess.run([sys.executable,str(ROOT/'code/pilot_ffv1_20261030.py'),'--subject',p.parent.name,'--replace-verified']).returncode
        if rc:
            status.update(state='failed',failed_subject=p.parent.name)
            (out/'status.json').write_text(json.dumps(status,indent=2))
            raise SystemExit(rc)
        r=json.loads(report.read_text())
    status['completed']+=1;status['saved_bytes']+=r['source_bytes']-r['ffv1_bytes']
    status['records'].append({'subject':p.parent.name,'frames':r['frames'],'saved_bytes':r['source_bytes']-r['ffv1_bytes']})
    (out/'status.json').write_text(json.dumps(status,indent=2))
status['state']='complete';(out/'status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status),flush=True)
