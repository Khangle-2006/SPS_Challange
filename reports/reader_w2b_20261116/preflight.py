import sys,json,time
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'round1_execution_20261110/v5_multiview_20261112'))
import runtime
b=runtime.budget(True);reserve=b['remaining_reserved_bytes']+420_000_000+1_000_000_000+100_000_000+200_000_000+400_000_000
assert b['scope_actual_bytes']+reserve<84_000_000_000
assert b['project_actual_bytes']+reserve<500_000_000_000
(P/'PREFLIGHT.json').write_text(json.dumps(dict(passed=True,measured_epoch=time.time(),budget=b,followup_cap=200_000_000,ephemeral_ram_cap=400_000_000,total_reserved=reserve),indent=2));print('PASS')
