import sys,time,json,hashlib,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'round1_execution_20261110/v5_multiview_20261112'))
import runtime
b=runtime.budget(True)
assert b['scope_actual_bytes']+b['remaining_reserved_bytes']+1_000_000_000+420_000_000<84_000_000_000
assert b['project_actual_bytes']+b['remaining_reserved_bytes']+1_000_000_000+420_000_000<500_000_000_000
assert json.loads((P/'CPU_VALIDATION.json').read_text())['passed']
assert not (P/'TRAIN_STARTED.json').exists()
receipt=dict(passed=True,measured_epoch=time.time(),budget=b,new_root_allocation=1_000_000_000,train_sha256=hashlib.sha256((P/'train_w2.py').read_bytes()).hexdigest(),holdout_evaluations=0)
(P/'LAUNCH_GUARDS.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
