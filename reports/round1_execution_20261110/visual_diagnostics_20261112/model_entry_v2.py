"""Path-only prelaunch correction. Original failed import/log preserved; no inference was run."""
import sys
from pathlib import Path
R=Path('/mnt/samsung_t5/khangle/facevitals')
sys.path.insert(0,str(R/'reports/round1_v4_20261105/padding_pilot_20261107'))
import model_maps
if __name__=='__main__':
 try:model_maps.main()
 except Exception as e:
  import traceback
  model_maps.save(model_maps.P/'MODEL_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=model_maps.now()));raise
