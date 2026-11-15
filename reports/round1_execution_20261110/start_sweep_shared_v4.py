"""Keep frozen training implementation; replace only authorized operational guards."""
from runtime_shared_v4 import install, verify_manifest
verify_manifest()
install()
import runpy
from common import P
runpy.run_path(str(P/'start_sweep_eligible_v2.py'),run_name='__main__')
