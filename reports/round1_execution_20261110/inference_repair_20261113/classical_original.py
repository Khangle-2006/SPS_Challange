import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
from crop_io import cached
from classical_repair import run
import traceback

def main():
 lease=lock('classical_original.lock');assert not (P/'CLASSICAL_ORIGINAL_START.json').exists();budget(True);save(P/'CLASSICAL_ORIGINAL_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__)));done=0
 for j in manifest():
  for c in ['native','c5','c5prime_default','c5prime_bf0']:
   out=P/'classical_original'/j['key']/c;assert not out.exists();out.mkdir(parents=True)
   save(P/'CLASSICAL_ORIGINAL_STATUS.json',dict(utc=now(),completed=done,total=1488,key=j['key'],condition=c))
   try:rgb,m,d=cached(j,c)
   except ValueError as e:save(out/'EXCLUDED.json',dict(reason=str(e)));done+=1;continue
   run(j,c,'original',rgb,m['t'],m['valid'],out);save(out/'COMPLETE.json',dict(utc=now(),sha256=sha(out/'CLASSICAL.json')));done+=1
  budget()
 save(P/'CLASSICAL_ORIGINAL_COMPLETE.json',dict(utc=now(),completed=done))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'CLASSICAL_ORIGINAL_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
