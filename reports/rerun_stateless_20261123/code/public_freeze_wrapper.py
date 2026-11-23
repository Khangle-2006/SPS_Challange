"""Run path-only z14 and require its rerun-z13 match; no new selection rule."""
from expert_spectrum import *
import traceback
started=time.monotonic()
try:
 selected_mode();assert (OUT/'PUBLIC_SCORE_COMPLETE.json').exists();assert not (OUT/'STOP.json').exists()
 src=OUT/'code/z14_p4m_freeze.py';exec(compile(src.read_text(),str(src),'exec'),globals())
 assert worst<=1e-9,('z14 vs z13 mismatch',worst)
 save(OUT/'PUBLIC_FREEZE_COMPLETE.json',dict(seconds=time.monotonic()-started,max_z13_mae_delta=worst,sha256=sha(OUT/'z14_p4m_frozen_rerun.npz')))
except BaseException as e:
 save(OUT/'PUBLIC_FREEZE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));save(OUT/'STOP.json',dict(stage='public_freeze',error=repr(e)));raise
