"""Five real public native ROI inputs, no closed-test files or Python subprocess in MATLAB."""
import os,subprocess,json,traceback
import numpy as np
from scipy.io import savemat,loadmat
from common import *
from portable_python import predict
def main():
 lease=lock('octave_parity.lock');guard(True);assert not (P/'OCTAVE_PARITY.json').exists()
 root=P/'octave_private/prefix';binary=root/'usr/bin/octave-cli';assert binary.exists()
 libs=[root/'usr/lib/x86_64-linux-gnu']+list((root/'usr/lib/x86_64-linux-gnu/octave').glob('*'))
 env=dict(os.environ,OCTAVE_HOME=str(root/'usr'),LD_LIBRARY_PATH=':'.join(map(str,libs)),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 version=subprocess.check_output([str(binary),'--version'],env=env,text=True,stderr=subprocess.STDOUT).splitlines()[0]
 pub=read(B/'public_comparison/MANIFEST.json')['records'];index=read(R/'t8_oval_20261031/training_index.json')
 selected=[r for ds,n in [('UBFC2',2),('MCD',3)] for r in [x for x in pub if x['dataset']==ds][:n]];results=[]
 for r in selected:
  meta=R/'reports/round1_final_20261105/revision_02/frozen_data/spatial_rgb'/r['dataset']/r['recording']/'native.npz'
  receipt=B/'public_comparison/pos_repair_20261110/CORRECTED_INPUT_HASHES.json';assert sha(meta)==read(receipt)[str(meta)]
  with np.load(meta) as z:t=z['t'].copy();rgb=z['rgb'].copy();valid=z['valid'].copy()
  assert len(t)==len(rgb)==len(valid) and np.all(np.diff(t)>0)
  out=P/'matlab_parity'/r['dataset']/r['recording'];out.mkdir(parents=True,exist_ok=False)
  roi=out/'roi_data.mat';savemat(roi,dict(roi_data=dict(t=t[:,None],rgb=rgb,valid=valid[:,None]),facial_video=r['source']),format='5',do_compression=True)
  reference=predict(t,rgb,valid);np.savetxt(out/'python.csv',reference,delimiter=',',fmt='%.12g')
  prefix=out/'octave';escape=lambda s:str(s).replace("'","''")
  expression=f"addpath('{escape(P)}'); process_round1('{escape(roi)}','{escape(prefix)}');"
  cp=subprocess.run([str(binary),'--traditional','--no-gui','--quiet','--eval',expression],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (out/'octave.log').write_text(cp.stdout);assert cp.returncode==0,cp.stdout[-3000:]
  actual=np.loadtxt(str(prefix)+'.csv',delimiter=',',skiprows=1,ndmin=2)
  assert actual.shape==reference.shape and np.array_equal(np.isfinite(actual),np.isfinite(reference))
  finite=np.isfinite(reference);maximum=float(np.max(abs(actual[finite]-reference[finite]))) if finite.any() else 0.
  assert maximum<.01
  payload=loadmat(str(prefix)+'.mat');assert all(k in payload for k in ['facial_video','roi_data','reference_signal','processed_signal','metrics','params'])
  results.append(dict(dataset=r['dataset'],recording=r['recording'],input_meta_sha256=sha(meta),input_manifest_sha256=sha(receipt),roi_mat_sha256=sha(roi),windows=len(reference),valid_windows=int(np.isfinite(reference[:,2]).sum()),max_abs_difference_bpm=maximum,CSV_sha256=sha(str(prefix)+'.csv'),MAT_sha256=sha(str(prefix)+'.mat')))
 save(P/'OCTAVE_PARITY.json',dict(passed=len(results)==5,version=version,mode='--traditional',results=results,matlab_verified=False,scope='toolbox-free POS/Welch packaging baseline, not the selected calibrated BTC estimator',m_sha256=sha(P/'process_round1.m'),python_sha256=sha(P/'portable_python.py'),holdout_evaluations=0))
 print('OCTAVE_PARITY_PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'OCTAVE_PARITY_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
