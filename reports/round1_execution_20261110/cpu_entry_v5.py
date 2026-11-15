"""Apply bounded storage remeasurement, then run the frozen CPU stage unchanged."""
import sys,runpy
from runtime_shared_v4 import install
install()
script=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(script,run_name='__main__')
