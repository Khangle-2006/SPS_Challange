import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
import time,traceback

def wait_for(names,errors,began):
 while not all((P/n).exists() for n in names):
  assert time.time()-began<36*3600,'36H_REPORT_BOUND'
  assert not any((P/n).exists() for n in errors),'DEPENDENCY_ERROR_REQUIRES_INSPECTION'
  save(P/'REPORT_CONTROLLER_STATUS.json',dict(utc=now(),waiting=[n for n in names if not (P/n).exists()]));time.sleep(15)

def main():
 lease=lock('report_controller.lock');assert not (P/'REPORT_CONTROLLER_START.json').exists();save(P/'REPORT_CONTROLLER_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__)));began=time.time()
 for stage,names,errors in [('original',['ORIGINAL_COMPLETE.json','CLASSICAL_ORIGINAL_COMPLETE.json','REFERENCE_COMPLETE.json'],['ORIGINAL_ERROR.json','CLASSICAL_ORIGINAL_ERROR.json','REFERENCE_ERROR.json']),('spatial',['SPATIAL_COMPLETE.json'],['SPATIAL_ERROR.json'])]:
  wait_for(names,errors,began)
  with (P/('report_'+stage+'.log')).open('wb') as log:subprocess.run(['/data_pool/khangle/facevitals/envs/storage_pilot/bin/python',str(P/'report_repair.py'),'--stage',stage],stdout=log,stderr=log,check=True)
 wait_for(['VISUAL_COMPLETE.json','CODEC_FOUR_COMPLETE.json'],['VISUAL_ERROR.json','CODEC_FOUR_ERROR.json'],began)
 save(P/'COMPUTE_COMPLETE.json',dict(utc=now(),state='scientific_review_rendering_and_FREEZE_decision_pending',no_automatic_promotion=True,holdout_evaluations=0))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'REPORT_CONTROLLER_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
