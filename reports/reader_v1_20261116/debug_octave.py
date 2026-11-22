import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import runpy
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import btc_posterior_reader as r
P=Path(__file__).resolve().parent
try:runpy.run_path(str(P/'verify_octave.py'))
except AssertionError as e:print(e)
a=loadmat(P/'octave_parity/case0.mat',simplify_cells=True);x=a['x']
for k,y in [('p',r.pos(x)),('c',r.chrom(x))]:
 print(k,'maxdiff',np.max(abs(y-a[k+'_wave'])),'std',np.std(y),np.std(a[k+'_wave']))
 print('power',np.max(abs(r.window_spectrum(y)-a['p'+k][:,-1])))
print('Zmax',a['Z'].min(),a['Z'].max())
f=sorted((P.parent/'round1_execution_20261110/matlab_parity').glob('*/*/roi_data.mat'))[0];raw=loadmat(f,simplify_cells=True)['roi_data'];t=raw['t'];valid=raw['valid'].astype(bool);rgb=raw['rgb'];pc=[];pp=[];starts=[]
for start in np.arange(0,max(0.,t[-1]-10+1e-8),5):
 if not r.supported(t,valid,start,start+10):continue
 xx=np.stack([np.interp(start+np.arange(300)/30,t,rgb[:,k]) for k in range(3)],axis=1);c=r.chrom(xx);p=r.pos(xx)
 if c.std()<1e-12 or p.std()<1e-12:continue
 pc.append(r.window_spectrum(c));pp.append(r.window_spectrum(p));starts.append(start)
print('count',len(starts),'octave',a['pc'].shape)
if len(starts)==a['pc'].shape[1]:
 print('all_power_max',np.max(abs(np.array(pc).T-a['pc'])),np.max(abs(np.array(pp).T-a['pp'])))
z=r.video_scores(t,valid,rgb);print('z_diff',np.max(abs(z-a['Z'])))
m=loadmat(P/'reader_model.mat',simplify_cells=True);print('posteriorfromOctaveZ',r.posterior_median(a['Z'],m['prior'],m['beta']),'pythonZ',r.posterior_median(z,m['prior'],m['beta']))
