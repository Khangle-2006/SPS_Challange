from common_repair import *
from readout import *
from t8_evaluate import decode_difference
import traceback
def main():
 rows=[];waves=[];t=np.arange(300)/30
 for bpm in [40,60,90,150,175]:
  wave=np.sin(2*np.pi*bpm/60*t+.37);d=np.diff(np.r_[0,wave]);a=all_decodes(d);old=decode_difference(d);assert abs(old-a['power_0.7'])<1e-9
  assert abs(a['power_0.6']-bpm)<1.5
  rows.append(dict(kind='sine_difference',truth=bpm,production=old,**a));waves.append(d)
 rng=np.random.default_rng(0)
 for i in range(100):
  d=rng.normal(size=300);a=all_decodes(d);old=decode_difference(d);assert abs(old-a['power_0.7'])<1e-9;rows.append(dict(kind='white_noise_difference',index=i,production=old,**a));waves.append(d)
 for value in [0,1,15.33]:
  assert all(v is None for v in all_decodes(np.full(300,value)).values())
 save(P/'READOUT_PROBE.json',dict(utc=now(),production_parity=True,rows=rows,static_constant_pass=True,warning='Noise argmax is not pulse detection. Lowering band can increase low-frequency bias; measured not presumed helpful.'))
 np.savez_compressed(P/'READOUT_PROBE_WAVES.npz',difference=waves)
 print('readout production parity/synthetic/static PASS')
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'READOUT_PROBE_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
