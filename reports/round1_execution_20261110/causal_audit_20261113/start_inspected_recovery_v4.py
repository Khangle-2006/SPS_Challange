"""One inspected continuation; never retries automatically."""
from common_causal import *
import shutil,traceback

def main():
 check_deadline();b=budget(True)
 for pid in [865817,869682]:assert not Path(f'/proc/{pid}').exists()
 lease=lock('ladder.lock')
 assert not (C/'ENCODER_RECOVERY_V4_PREREG.json').exists()
 probe=read(C/'ENCODER_CLOCK_PROBE_V4.json');assert probe['results'][1]['max_abs_PTS_difference_s']<2e-5
 archive=C/'ENCODER_CLOCK_ATTEMPT1';archive.mkdir(exist_ok=False)
 for n in ['ladder_physical_v3.py','review_controller_v3.py','physical_v3/LADDER_ERROR.json','physical_v3/LADDER_STARTED.json','REVIEW_STATUS.json','REVIEW_NEEDS_INSPECTION.json']:
  src=C/n
  if src.exists():dst=archive/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 root=C/'physical_v3/ladder';reused={}
 for case in read(C/'physical_v3/LADDER_SELECTION.json'):
  d=root/case['job']['key']
  if (d/'CASE_COMPLETE.json').exists():assert case['job']['dataset']=='UBFC2';folders=[d]
  elif case['job']['key']=='MCD_1115_FullHDwebcam_after':folders=[d/'s0_native']
  else:assert not d.exists();folders=[]
  for folder in folders:
   for f in folder.rglob('*'):
    if f.is_file() and not f.is_symlink():reused[str(f.relative_to(C))]=sha(f)
 failed=root/'MCD_1115_FullHDwebcam_after/s1_hevc'
 assert failed.exists() and not (failed/'COMPLETE.json').exists()
 # Resolved paths explicitly confined to this audit root; preserve incomplete attempt.
 assert failed.resolve().is_relative_to(C.resolve()) and archive.resolve().is_relative_to(C.resolve())
 failed.rename(archive/'s1_hevc_partial')
 for f in (failed.parent).glob('s1_hevc*log'):shutil.copyfile(f,archive/f.name)
 save(C/'ENCODER_RECOVERY_V4_PREREG.json',dict(utc=now(),cause='FFmpeg default encoder timebase used r_frame_rate359/12 instead of source299/10. demux timebase preserves measured source PTS.',repair='libx265 -enc_time_base demux; same guard20us unchanged; no frame interpolation/HR fitting',probe_sha256=sha(C/'ENCODER_CLOCK_PROBE_V4.json'),selection_sha256=sha(C/'physical_v3/LADDER_SELECTION.json'),reused_files=reused,budget=b,old_attempt_preserved=True,no_training=True))
 lease.close()
 for script,cpus,receipt in [('ladder_recovery_v4.py','14,15','LADDER_RECOVERY_V4_LAUNCH.json'),('review_controller_v4.py','12','REVIEW_V4_LAUNCH.json')]:
  assert not (C/receipt).exists();f=open(C/(script+'.log'),'xb');q=subprocess.Popen(['nice','-n','15','taskset','-c',cpus,sys.executable,str(C/script)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(C/receipt,dict(utc=now(),pid=q.pid,script=script,code_sha256=sha(C/script),budget=b));print(script,q.pid,flush=True)
if __name__=='__main__':main()
