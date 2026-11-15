from common_repair import *
from classical_repair import waveform_decode
from t8_t1_loader import supported
import numpy as np,traceback

def main():
 lease=lock('reference.lock');assert not (P/'REFERENCE_START.json').exists();save(P/'REFERENCE_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__)));rows=[]
 for j in manifest():
  if j['dataset']=='BTC':continue
  m=np.load(Path(j['crop'])/'meta.npz');t=m['ppg_t'];v=m['ppg'];valid=m['ppg_valid']
  for w in j['windows']:
   old=new=None;reason=None
   try:
    if not supported(t,valid,w['start'],w['start']+10):raise ValueError('ppg_support_failure')
    y=np.interp(w['start']+np.arange(300)/30,t,v);old=waveform_decode(y,.7);new=waveform_decode(y,.6)
   except ValueError as e:reason=str(e)
   if w['eligible']:assert old==w['truth'],f"OLD_REFERENCE_MISMATCH {j['key']} {w['start']} {old}/{w['truth']}"
   rows.append(dict(dataset=j['dataset'],recording=j['recording'],person=j['person'],start=w['start'],old_eligible=w['eligible'],old_truth=w['truth'],recomputed_07=old,new_06=new,reason=reason,meta_sha256=sha(Path(j['crop'])/'meta.npz')))
 save(P/'REFERENCE_06.json',rows);save(P/'REFERENCE_COMPLETE.json',dict(utc=now(),rows=len(rows),old_eligible_exact=sum(r['old_eligible'] for r in rows),changed=sum(r['old_eligible'] and r['old_truth']!=r['new_06'] for r in rows),sha256=sha(P/'REFERENCE_06.json'),no_clock_support_or_raw_label_change=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'REFERENCE_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
