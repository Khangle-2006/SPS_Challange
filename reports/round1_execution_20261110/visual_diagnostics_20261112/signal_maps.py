import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
from vis_common import *
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import time,traceback
from pipeline import pulse
plt.rcParams.update({'font.size':9,'figure.facecolor':'white','axes.facecolor':'white','text.color':'#14232b','axes.labelcolor':'#14232b','xtick.color':'#14232b','ytick.color':'#14232b','savefig.facecolor':'white'})
def ratio(y,pr):
 if not np.isfinite(pr) or not .7<=pr/60<=3:return np.nan
 f,p=signal.periodogram(y,fs=30,window='hann',nfft=4096);sel=(f>=.7)&(f<=3);den=p[sel].sum()
 return float(p[sel & (abs(f-pr/60)<=.05)].sum()/den) if den>1e-20 else np.nan
def wave(d):return signal.sosfiltfilt(signal.butter(3,[.7,3],fs=30,btype='bandpass',output='sos'),signal.detrend(np.cumsum(d)))
def correlation(a,b):
 if not np.isfinite(b).all() or np.std(a)<1e-12 or np.std(b)<1e-12:return None
 b=signal.sosfiltfilt(signal.butter(3,[.7,3],fs=30,btype='bandpass',output='sos'),signal.detrend(b));cc=[]
 for lag in range(-30,31):
  x,y=(a[-lag:],b[:lag]) if lag<0 else ((a[:-lag],b[lag:]) if lag else (a,b));cc.append(float(np.corrcoef(x,y)[0,1]))
 best=int(np.argmax(cc));return dict(zero=cc[30],max_signed=cc[best],lag_seconds=(best-30)/30,curve=cc,lag_convention='positive means PPG sampled later than model',boundary=best in [0,60])
def draw(r):
 d=P/'data'/r['key'];m=P/'model'/r['key'];out=P/'signal'/r['key'];out.mkdir(parents=True);a=dict(np.load(d/'clips.npz'));o=dict(np.load(m/'outputs.npz'));maps=[];pos=[];corr=[]
 for w,rgb in enumerate(a['rgb']):
  rgb=rgb[:300].astype(float);mapping=np.full((2,8,8),np.nan);edges=np.linspace(0,36,9).astype(int)
  for j in range(8):
   for k in range(8):
    tr=rgb[:,edges[j]:edges[j+1],edges[k]:edges[k+1]].mean((1,2))
    for h,method in enumerate(['G','POS']):mapping[h,j,k]=ratio(pulse(tr,30,method),a['pr'][w])
  maps.append(mapping);pos.append(pulse(rgb.mean((1,2)),30,'POS'));corr.append(correlation(wave(o['difference'][w]),a['ppg'][w]))
 np.savez_compressed(out/'signal.npz',pulse_maps=maps,pos=pos)
 save(out/'METRICS.json',dict(correlations=corr,pr=[float(x) if np.isfinite(x) else None for x in a['pr']],selected_windows=len(maps),hash=sha(out/'signal.npz')))
 fig,ax=plt.subplots(2,3,figsize=(13,8),layout='constrained');mean=o['mean_rgb']/255;ax[0,0].imshow(np.clip(mean,0,1));ax[0,0].set_title('Mean model input (36 × 36)')
 for col,key in [(1,'gradient'),(2,'occlusion_signed')]:
  h=abs(o[key]);h=h/np.nansum(h) if np.nansum(h)>0 else h;ax[0,col].imshow(np.clip(mean,0,1));ax[0,col].imshow(h,extent=(-.5,35.5,35.5,-.5),cmap='magma',alpha=.7);ax[0,col].set_title('Absolute '+key+' mass')
 for col,method in enumerate(['G','POS']):
  im=ax[1,col].imshow(np.nanmean(maps,axis=0)[col],vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=ax[1,col],label='Power near PR / band power');ax[1,col].set_title(method+' label-guided 8 × 8 grid')
 geo=np.load(d/'geometry.npz')['regions'];ax[1,2].imshow(np.argmax(geo,axis=0),vmin=0,vmax=3,cmap='tab10');ax[1,2].set_title('Landmark regions (not skin segmentation)')
 for a0 in ax.flat:a0.set_xticks([]);a0.set_yticks([])
 fig.suptitle(r['group']+' — '+r['recording']);fig.savefig(out/'spatial.png',dpi=140);plt.close(fig)
 fig,ax=plt.subplots(3 if r['group']=='MCD Iriun' else 2,1,figsize=(12,8),layout='constrained');vmin=-35;vmax=0
 for j in range(len(o['difference'])):
  start=float(a['starts'][j]);pr=float(a['pr'][j]);ys=[wave(o['difference'][j]),pos[j]]
  for k,y in enumerate(ys):
   ff,tt,ss=signal.spectrogram(y,fs=30,window='hann',nperseg=150,noverlap=135,nfft=1024,mode='psd');band=(ff>=.7)&(ff<=3);z=10*np.log10(np.maximum(ss[band]/max(ss[band].max(),1e-30),1e-8));im=ax[k].pcolormesh(start+tt,ff[band]*60,z,vmin=vmin,vmax=vmax,cmap='magma',shading='auto')
   if np.isfinite(pr):ax[k].plot([start,start+10],[pr,pr],'c--',lw=1.5)
   ax[k].axvline(start,color='gray',lw=.5);ax[k].set_xlim(0,30.2);ax[k].set_ylim(40,182);ax[k].set_ylabel('bpm');ax[k].set_title(['Padding-fixed output; cyan = reference','POS; independently scaled relative power'][k])
  if len(ax)==3 and np.isfinite(a['ppg'][j]).all():
   y=ys[0];gt=signal.sosfiltfilt(signal.butter(3,[.7,3],fs=30,btype='bandpass',output='sos'),signal.detrend(a['ppg'][j]));tt=start+np.arange(300)/30;ax[2].plot(tt,y/max(y.std(),1e-12),lw=.7,label='Prediction' if j==0 else None);ax[2].plot(tt,gt/max(gt.std(),1e-12),lw=.7,alpha=.7,label='Canonical PPG' if j==0 else None);ax[2].set_title('No lag shift applied; each 10 s reconstructed independently');ax[2].legend(loc='upper right')
 fig.colorbar(im,ax=list(ax[:2]),label='dB relative to own-window maximum');ax[-1].set_xlabel('Time (s); blank intervals = missing support / window edges');fig.suptitle(r['group']+' — '+r['recording']);fig.savefig(out/'temporal.png',dpi=140);plt.close(fig)
 save(out/'COMPLETE.json',dict(utc=now(),input_hash=sha(m/'outputs.npz'),metrics_hash=sha(out/'METRICS.json')))
def summary(rows):
 groups=sorted({r['group'] for r in rows});table=[];evms=[]
 fig,ax=plt.subplots(len(groups),4,figsize=(13,3*len(groups)),layout='constrained')
 for j,g in enumerate(groups):
  rr=[r for r in rows if r['group']==g and (P/'signal'/r['key']/'COMPLETE.json').exists()];grad=[];occ=[];maps=[]
  for r in rr:
   o=np.load(P/'model'/r['key']/'outputs.npz');grad.append(o['gradient']/max(np.nansum(o['gradient']),1e-30));h=abs(o['occlusion_signed']);occ.append(h/max(np.nansum(h),1e-30));maps.append(np.nanmean(np.load(P/'signal'/r['key']/'signal.npz')['pulse_maps'],axis=0));metric=read(P/'model'/r['key']/'COMPLETE.json')
   for method,v in metric['regional'].items():table.append(dict(group=g,recording=r['recording'],method=method,**v,non_skin=v['oval_edge']+v['eyes_mouth']+v['exterior']))
   if (P/'data'/r['key']/'evm.npz').exists():evms.append(r)
  for k,arr in enumerate([grad,occ,[v[0] for v in maps],[v[1] for v in maps]]):
   if arr:im=ax[j,k].imshow(np.nanmean(arr,axis=0),cmap='viridis',vmin=0,vmax=1 if k>1 else None);fig.colorbar(im,ax=ax[j,k],shrink=.6)
   ax[j,k].set_title(f'{g} n={len(rr)}: '+['gradient','occlusion','G PR-band ratio','POS PR-band ratio'][k]);ax[j,k].set_xticks([]);ax[j,k].set_yticks([])
 fig.savefig(P/'SPATIAL_SUMMARY.png',dpi=140);plt.close(fig);save(P/'REGIONAL_MASS.json',table)
 trigger=[]
 for method in ['gradient','occlusion']:
  ub=[r['non_skin'] for r in table if r['group']=='UBFC' and r['method']==method];btc=[r['non_skin'] for r in table if r['group'].startswith('BTC') and r['method']==method]
  if ub and btc:trigger.append(dict(method=method,UBFC_mean=float(np.mean(ub)),BTC_mean=float(np.mean(btc)),UBFC_n=len(ub),BTC_n=len(btc),trigger=bool(np.mean(btc)>.4 and np.mean(ub)<.2)))
 save(P/'EDGE_TRIGGER.json',trigger)
 if evms:
  arr=[dict(np.load(P/'data'/r['key']/'evm.npz')) for r in evms];limit=float(np.quantile(np.concatenate([abs(a['yt']).ravel() for a in arr]),.99));fig,ax=plt.subplots(len(groups),3,figsize=(15,3*len(groups)),layout='constrained')
  for j,g in enumerate(groups):
   for k,r in enumerate([r for r in evms if r['group']==g]):
    a=dict(np.load(P/'data'/r['key']/'evm.npz'));im=ax[j,k].imshow(a['yt'].T,extent=[a['t'][0],a['t'][-1],1,0],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit);ax[j,k].set_title(g+' / '+r['recording']);ax[j,k].set_xlabel('Seconds');ax[j,k].set_ylabel('Face vertical coordinate')
  fig.colorbar(im,ax=list(ax.flat),label='50 × filtered green /255, shared scale');fig.savefig(P/'EVM_YT.png',dpi=130);plt.close(fig);save(P/'EVM_SCALE.json',dict(alpha=50,pyramid_level=4,symmetric_limit=limit,scale_rule='pooled99percentile absolute displayed amplitude'))
 features=[];labels=[]
 for r in rows:
  p=P/'model'/r['key']/'outputs.npz'
  if p.exists():
   a=np.load(p)
   for i,f in enumerate(a['features_common']):features.append(f);labels.append(dict(recording=r['recording'],person=r['person'],domain=r['domain'],selection_group=r['group'],window_start=float(a['starts'][i])))
 np.savez_compressed(P/'COMMON_FEATURES.npz',features=features);save(P/'FEATURE_LABELS.json',labels)
def main():
 lease=lock('signal.lock');assert not (P/'SIGNAL_START.json').exists();save(P/'SIGNAL_START.json',dict(pid=os.getpid(),utc=now()));rows=manifest();done=set();errors=[];begin=time.time()
 test=np.sin(2*np.pi*1.5*np.arange(300)/30);assert ratio(test,90)>.5 and ratio(test,130)<.01 and np.isnan(ratio(np.zeros(300),90));save(P/'SIGNAL_TEST.json',dict(passed=True,known90=ratio(test,90),off130=ratio(test,130)))
 while len(done)<len(rows):
  assert time.time()-begin<4*3600
  for r in rows:
   if r['key'] in done:continue
   if (P/'data'/r['key']/'ERROR.json').exists() or (P/'model'/r['key']/'ERROR.json').exists():done.add(r['key']);errors.append(r['key']);continue
   if not (P/'model'/r['key']/'COMPLETE.json').exists():continue
   try:draw(r)
   except Exception as e:errors.append(r['key']);save(P/'signal'/r['key']/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()))
   done.add(r['key']);save(P/'SIGNAL_STATUS.json',dict(utc=now(),completed=len(done),errors=errors))
  if len(done)<len(rows):time.sleep(10)
 summary(rows);save(P/'SIGNAL_COMPLETE.json',dict(utc=now(),total=len(done),errors=errors))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'SIGNAL_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
