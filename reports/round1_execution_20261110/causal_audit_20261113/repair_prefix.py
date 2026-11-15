"""Repair only public clips whose container-time prefix truncated the true window.
Selection uses timestamps only. Existing valid ladders and all evidence retained.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',EGL_PLATFORM='surfaceless',MESA_LOADER_DRIVER_OVERRIDE='swrast',GALLIUM_DRIVER='llvmpipe',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
from common_causal import *
import ladder_prefix_v3 as ladder
import localization_all as localization
import copy,traceback

def main():
 lease=lock('prefix_repair.lock');assert not (C/'PREFIX_REPAIR_STARTED.json').exists();check_deadline();b=budget(True)
 audit=read(C/'PREFIX_CLOCK_AUDIT.json');cases={x['job']['key']:x for x in read(C/'LADDER_SELECTION.json')}
 selected=[x for x in audit if x['insufficient']]
 root=C/'prefix_repair';root.mkdir(exist_ok=False);(root/'ladder_infer_v2.py').symlink_to(C/'ladder_infer_v2.py')
 ladder.C=root;localization.C=root;ladder.FOLLOWUP=localization.one
 save(C/'PREFIX_REPAIR_STARTED.json',dict(utc=now(),pid=os.getpid(),cases=selected,selection_reason='Acquisition clock window not fully present in container-trimmed prefix; no HR criterion',audit_sha256=sha(C/'PREFIX_CLOCK_AUDIT.json'),code_sha256=sha(__file__),ladder_code_sha256=sha(C/'ladder_prefix_v3.py'),budget=b,original_experiment_unchanged=True,no_training=True))
 for row in selected:
  check_deadline();budget(True);case=copy.deepcopy(cases[row['key']]);case['audit_prefix_end']=row['new_container_end'];ladder.case_run(case)
 save(C/'PREFIX_REPAIR_COMPLETE.json',dict(utc=now(),cases=[x['key'] for x in selected],no_training=True,old_outputs_preserved=True))

if __name__=='__main__':
 try:main()
 except Exception as e:save(C/'PREFIX_REPAIR_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
