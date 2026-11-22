import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import savemat,loadmat
P=Path(__file__).resolve().parent;R=P.parent/'reader_v1_20261116';E=P.parent/'round1_execution_20261110';V=P.parent/'round1_v4_20261105/btc_zeroshot'
sys.path.insert(0,str(R));import reader_v1 as api
m=json.loads((P/'reader_model_dev218_v2.json').read_text());pr=np.array(m['prior'])
records=json.loads((R/'PREDICTIONS.json').read_text());ids=[r['recording'] for r in records if r['condition']=='native' and r['method']=='reader_v1'][:20];dev={r['recording']:r for r in json.loads((V/'dev_manifest.json').read_text())['videos']};static=[]
for ident in ids:
 v=dev[ident];f=V/'dev'/v['drive_id']/'native/meta.npz'
 with np.load(f) as a:rgb=np.tile(a['rgb_mean'][0],(901,1))
 z=api.scores(np.arange(901)/30,np.ones(901,bool),rgb);assert z is None
 flag=v['task'].lower().startswith('task');offset=m['offset_task' if flag else 'offset_rest'];pred=api.ref.posterior_median(z,pr,m['beta'])+offset
 static.append(dict(recording=ident,abstain=True,prediction=pred,prior_median=api.ref.posterior_median(None,pr,0),offset=offset,passed=pred==api.ref.posterior_median(None,pr,0)+offset))
assert all(x['passed'] for x in static)
(P/'STATIC.json').write_text(json.dumps(dict(passed=True,cases=static,interpretation='Conditional prior fallback, not measured pulse; flag abstain remains true'),indent=2))
cb,ca=signal.butter(3,[.7,3],fs=30,btype='bandpass');rb,ra=signal.butter(1,[.7,3],fs=30,btype='bandpass')
savemat(P/'reader_model.mat',dict(grid=np.array(m['grid_bpm'])[:,None],prior=pr[:,None],beta=float(m['beta']),offset_task=m['offset_task'],offset_rest=m['offset_rest'],cb=cb,ca=ca,czi=signal.lfilter_zi(cb,ca),rb=rb,ra=ra,rzi=signal.lfilter_zi(rb,ra)),format='5')
root=E/'octave_private/prefix';binary=root/'usr/bin/octave-cli';libs=sorted({f.parent for f in (root/'usr/lib').rglob('*.so*')});env=dict(os.environ,OCTAVE_HOME=str(root/'usr'),LD_LIBRARY_PATH=':'.join(map(str,libs)))
inputs=sorted((E/'matlab_parity').glob('*/*/roi_data.mat'));assert len(inputs)==5;dest=P/'octave_parity';dest.mkdir(exist_ok=True);results=[]
for i,f in enumerate(inputs):
 a=loadmat(f,simplify_cells=True)['roi_data'];base=api.predict(a['t'],a['valid'],a['rgb'],m)
 # Exercise both categorical branches on each real ROI input. These are numerical
 # branch checks, not claims about the public clip's recorded activity.
 for flag in [False,True]:
  pred=base['prediction']+m['offset_task' if flag else 'offset_rest'];out=dest/f'case{i}_flag{int(flag)}'
  cmd=f"addpath('{P}');process_round1('{f}','{out}','{P/'reader_model.mat'}',{int(flag)});"
  cp=subprocess.run([str(binary),'--traditional','--no-gui','--quiet','--eval',cmd],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);out.with_suffix('.log').write_text(cp.stdout);assert cp.returncode==0,cp.stdout
  q=np.loadtxt(str(out)+'.csv',delimiter=',',skiprows=1);delta=abs(float(q[0])-pred);assert delta<=.5 and bool(q[1])==base['abstain']
  results.append(dict(input=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),test_task_flag=flag,python=pred,octave=float(q[0]),difference_bpm=delta))
(P/'OCTAVE_PARITY.json').write_text(json.dumps(dict(passed=True,real_ROI_videos=5,branch_checks=10,cases=results,real_MATLAB_verified=False),indent=2));print('STATIC20_PASS OCTAVE5_PASS')
