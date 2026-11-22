import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import savemat,loadmat
import reader_v1 as api
P=Path(__file__).resolve().parent;E=P.parent/'round1_execution_20261110';model=json.loads((P/'reader_model_dev218.json').read_text())
cb,ca=signal.butter(3,[.7,3],fs=30,btype='bandpass');rb,ra=signal.butter(1,[.7,3],fs=30,btype='bandpass')
savemat(P/'reader_model.mat',dict(grid=np.array(model['grid_bpm'])[:,None],prior=np.array(model['prior'])[:,None],beta=float(model['beta']),cb=cb,ca=ca,czi=signal.lfilter_zi(cb,ca),rb=rb,ra=ra,rzi=signal.lfilter_zi(rb,ra)),format='5')
root=E/'octave_private/prefix';binary=root/'usr/bin/octave-cli';libs=sorted({f.parent for f in (root/'usr/lib').rglob('*.so*')});env=dict(os.environ,OCTAVE_HOME=str(root/'usr'),LD_LIBRARY_PATH=':'.join(map(str,libs)))
inputs=sorted((E/'matlab_parity').glob('*/*/roi_data.mat'));assert len(inputs)==5
results=[];dest=P/'octave_parity';dest.mkdir(exist_ok=True)
for i,f in enumerate(inputs):
 a=loadmat(f,simplify_cells=True)['roi_data'];r=api.predict(a['t'],a['valid'],a['rgb'],model);out=dest/f'case{i}'
 cmd=f"addpath('{P}'); process_round1('{f}','{out}','{P/'reader_model.mat'}');"
 cp=subprocess.run([str(binary),'--traditional','--no-gui','--quiet','--eval',cmd],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);out.with_suffix('.log').write_text(cp.stdout);assert cp.returncode==0,cp.stdout
 q=np.loadtxt(str(out)+'.csv',delimiter=',',skiprows=1);delta=abs(float(q[0])-r['prediction']);assert delta<=.5 and bool(q[1])==r['abstain'],(r,q)
 results.append(dict(input=str(f),input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),python=r,octave=float(q[0]),difference_bpm=delta))
(P/'OCTAVE_PARITY.json').write_text(json.dumps(dict(passed=True,cases=results,real_MATLAB_verified=False,model_source_sha256=hashlib.sha256((P/'reader_model_dev218.json').read_bytes()).hexdigest()),indent=2));print('PARITY_PASS')
