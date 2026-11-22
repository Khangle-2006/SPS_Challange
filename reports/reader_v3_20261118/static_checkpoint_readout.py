"""CPU negative control: real dev ROI -> frozen checkpoint -> public readout.
No training or label-based selection. This does not retest raw face detection.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,json,hashlib,argparse,fcntl
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
R=P.parent
sys.path[:0]=[str(R/'round1_bayes_20261108/domain_sweep_v2_20261110'),'/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102',str(R/'round1_v4_20261105/padding_pilot_20261107'),str(P/'source')]
from btc_data import BTCData
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
import lib_btc,public_readout

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 args.out.mkdir(exist_ok=True);lock=(args.out/'static.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (args.out/'COMPLETE.json').exists()
 assert os.environ['CUDA_VISIBLE_DEVICES']==''
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 data=BTCData(0,max_bytes=100_000_000);rows=[];checkpoints=[]
 for f in range(5):
  root=R/'round1_v4_20261105/v4_padding_pilot' if f==0 else R/'round1_execution_20261110/final_priorities_20261111/padding_fixed'
  d=root/f'fold{f}/seed2027/joint';st=json.loads((d/'status.json').read_text());cp=d/'final.pt'
  assert st['state']=='complete' and st['completed_epochs']==30 and sha(cp)==st['checkpoint_sha256']
  model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36))
  model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
  selected=[]
  for v in sorted(data.rows,key=lambda v:v['recording']):
   if v['fold']!=f:continue
   starts=data.starts(v,'native')
   if starts:selected.append((v,starts[0]))
   if len(selected)==5:break
  assert len(selected)==5
  checkpoints.append(dict(fold=f,path=str(cp),sha256=sha(cp)))
  for v,start in selected:
   x=data.clip(v,start);x=np.repeat(x[:1],len(x),axis=0)
   with torch.inference_mode():wave=model(torch.from_numpy(x[None]))[0].double().numpy()
   rgb,t,valid=data.load(v,'native');j=np.searchsorted(t,start,side='right').clip(1,len(t)-1);w=(start-t[j-1])/(t[j]-t[j-1])
   frame=rgb[j-1].astype(float)*(1-w)+rgb[j].astype(float)*w
   means=np.repeat(frame.mean((0,1))[None],300,axis=0)
   with np.errstate(all='ignore'):pos=lib_btc.pos_team(means);chrom=lib_btc.chrom_team(means)
   spectra=public_readout.wave_spectra(wave,pos,chrom)
   tests=[public_readout.decode([spectra],[1,1,1],[0],tracking=cfg) for cfg in [None,(6,.2)]]
   passed=all(z['predictions']==[None] and z['abstain']==[True] for z in tests)
   rows.append(dict(fold=f,recording=v['recording'],start=start,wave_std=float(wave.std()),wave_mean=float(wave.mean()),pos_finite=bool(np.isfinite(pos).all()),chrom_finite=bool(np.isfinite(chrom).all()),pass_static=passed,tests=tests))
   (args.out/'STATUS.json').write_text(json.dumps(dict(cases=len(rows),passed=sum(r['pass_static'] for r in rows))))
   assert passed,'Static image produced HR; retain evidence, do not change thresholds'
 (args.out/'RESULTS.json').write_text(json.dumps(dict(cases=rows,checkpoints=checkpoints,input_hashes=data.hashes,script_sha256=sha(Path(__file__)),readout_sha256=sha(P/'public_readout.py'),scope='Cached real ROI to actual CPU checkpoint and public readout; no raw face-detection rerun'),indent=2))
 (args.out/'COMPLETE.json').write_text(json.dumps(dict(passed=True,cases=len(rows),checkpoints=5,readout_modes=2,holdout_evaluations=0,gpu_used=False)))
if __name__=='__main__':main()
