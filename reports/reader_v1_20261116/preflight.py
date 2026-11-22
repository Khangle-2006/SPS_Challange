import sys,json
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'round1_execution_20261110/v5_multiview_20261112'))
import runtime
b=runtime.budget(True)
assert b['scope_actual_bytes']+b['remaining_reserved_bytes']+420_000_000+1_000_000_000+100_000_000<84_000_000_000
assert b['project_actual_bytes']+b['remaining_reserved_bytes']+420_000_000+1_100_000_000<500_000_000_000
(P/'PREFLIGHT.json').write_text(json.dumps(dict(passed=True,budget=b,reader_reservation=100_000_000,W2_reservation=1_000_000_000),indent=2));print('PASS')
