import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
import time,traceback

def main():
 lease=lock('render_controller.lock');assert not (P/'RENDER_CONTROLLER_START.json').exists();save(P/'RENDER_CONTROLLER_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__),renderer_sha256=sha(P/'render_repair.py')));began=time.time()
 required=['COMPUTE_COMPLETE.json','CODEC_BAND_V2_COMPLETE.json','BASELINE_AUDIT_COMPLETE.json']
 errors=['REPORT_CONTROLLER_ERROR.json','CODEC_BAND_V2_ERROR.json','BASELINE_AUDIT_ERROR.json','VISUAL_ERROR.json','SPATIAL_ERROR.json','CODEC_FOUR_ERROR.json','REPORT_original_ERROR.json','REPORT_spatial_ERROR.json']
 while not all((P/f).exists() for f in required):
  assert time.time()-began<36*3600,'36H_BOUND';assert not any((P/f).exists() for f in errors),'DEPENDENCY_FAILURE_NO_AUTOMATIC_RESTART';save(P/'RENDER_CONTROLLER_STATUS.json',dict(utc=now(),waiting=[f for f in required if not (P/f).exists()]));time.sleep(30)
 budget(True)
 assert sha(P/'render_repair.py')==read(P/'RENDER_CONTROLLER_START.json')['renderer_sha256']
 with (P/'render_repair.log').open('wb') as log:subprocess.run(['/data_pool/khangle/facevitals/envs/storage_pilot/bin/python','-u',str(P/'render_repair.py')],stdout=log,stderr=log,check=True)
 save(P/'REVIEW_READY.json',dict(utc=now(),state='needs_scientific_and_visual_review_before_FREEZE',no_training=True,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'RENDER_CONTROLLER_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
