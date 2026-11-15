"""CPU trace-to-posterior prototype, protocol DESIGN.md.

POS: Wang et al.2017, doi:10.1109/TBME.2016.2609282.
CHROM: de Haan & Jeanne2013, doi:10.1109/TBME.2013.2266196.
Multiple regions motivated by Kumar et al.2015 distancePPG; not its implementation.
Background/common-light control motivated by Tarassenko et al.2014; no AR replication.
SQI-mixture and log-opinion pool are this project's preregistered model, not claimed
as a verbatim implementation of Li/RePSS or Gideon & Stent2021. Speth2023 SiNC
belongs to a conditional training branch, not this classical module.
"""
import numpy as np
from scipy import signal,optimize,special,stats
GRID=np.arange(42.,180.01,.5)
def pulse(rgb,fs,method):
 rgb=np.asarray(rgb,float)
 if not np.isfinite(rgb).all() or len(rgb)<fs*2:raise ValueError('insufficient finite RGB')
 if np.max(np.std(rgb,axis=0)/np.maximum(abs(rgb.mean(0)),1e-10))<1e-8:return np.zeros(len(rgb))
 if method=='G':x=-rgb[:,1]
 else:
  n=max(4,int(round(1.6*fs)));x=np.zeros(len(rgb));count=np.zeros(len(rgb))
  for i in range(len(rgb)-n+1):
   c=rgb[i:i+n]/np.maximum(rgb[i:i+n].mean(0),1e-10)-1
   if method=='POS':a=c[:,1]-c[:,2];b=c[:,1]+c[:,2]-2*c[:,0];h=a+np.std(a)/max(np.std(b),1e-12)*b
   elif method=='CHROM':
    a=3*c[:,0]-2*c[:,1];b=1.5*c[:,0]+c[:,1]-1.5*c[:,2]
    sos=signal.butter(3,[.7,3],fs=fs,btype='bandpass',output='sos')
    a=signal.sosfiltfilt(sos,a,padlen=min(n-1,21));b=signal.sosfiltfilt(sos,b,padlen=min(n-1,21))
    h=a-np.std(a)/max(np.std(b),1e-12)*b
   else:raise ValueError(method)
   x[i:i+n]+=h-h.mean();count[i:i+n]+=1
  x/=np.maximum(count,1)
 return signal.sosfiltfilt(signal.butter(3,[.7,3],fs=fs,btype='bandpass',output='sos'),signal.detrend(x))
def native_grid(t,rgb):
 t=np.asarray(t,float);rgb=np.asarray(rgb,float);good=np.isfinite(rgb).all(1)
 if good.sum()<2:raise ValueError('no valid trace')
 tg=t[good];rg=rgb[good]
 if np.max(np.diff(tg))>.2:raise ValueError('trace_gap_over_0.2s')
 fs=1/np.median(np.diff(t));duration=tg[-1]-tg[0]
 if duration+1/fs<10:raise ValueError('shorter_than_10s')
 if fs<=6:raise ValueError('Nyquist_insufficient')
 q=tg[0]+np.arange(int(np.floor(duration*fs))+1)/fs
 return q,np.stack([np.interp(q,tg,rg[:,c]) for c in range(3)],axis=1),float(fs)
def psd(x,fs):
 n=min(len(x),int(round((15 if len(x)/fs>=20 else 10)*fs)))
 return signal.welch(x,fs=fs,window='hann',nperseg=n,noverlap=n//2,nfft=max(4096,2**int(np.ceil(np.log2(n)))),detrend='linear')
def peak_info(f,p,k=3):
 band=(f>=.7)&(f<=3);fb=f[band];pb=p[band]
 if not len(pb) or not np.isfinite(pb).all() or pb.max()<1e-20:return []
 ids=signal.find_peaks(pb)[0].tolist()
 # Retain a boundary maximum as an explicit boundary candidate, never hide it.
 if pb[0]>=pb[1]:ids.append(0)
 if pb[-1]>=pb[-2]:ids.append(len(pb)-1)
 ids=sorted(set(ids),key=lambda i:pb[i],reverse=True)[:k];out=[]
 for i in ids:
  hz=float(fb[i]);d=0.
  if 0<i<len(pb)-1:
   yy=np.log(np.maximum(pb[i-1:i+2],1e-30));den=yy[0]-2*yy[1]+yy[2]
   if abs(den)>1e-12:d=float(np.clip(.5*(yy[0]-yy[2])/den,-.5,.5))
  hz+=d*(fb[1]-fb[0]);near=abs(fb-hz)<=.1;power=float(pb[near].sum());rest=max(float(pb[~near].sum()),1e-30)
  out.append(dict(hr=60*hz,hz=hz,power=power,snr_db=float(10*np.log10(max(power,1e-30)/rest)),concentration=float(power/max(pb.sum(),1e-30))))
 return out
def trace_candidates(t,rgb,artifact_hz=(),ablation='none'):
 """rgb [frames,4regions,3]; ROI3 is non-skin background. No label input."""
 candidates=[];waves=[];failures=[];background=[]
 try:
  _,x,fs=native_grid(t,rgb[:,3]);background=peak_info(*psd(pulse(x,fs,'POS'),fs))
 except ValueError as e:failures.append(dict(roi='background',reason=str(e)))
 bg=[p['hz'] for p in background if p['snr_db']>=6]
 for ri,name in enumerate(['forehead','left_cheek','right_cheek']):
  try:q,x,fs=native_grid(t,rgb[:,ri])
  except ValueError as e:failures.append(dict(roi=name,reason=str(e)));continue
  for method in ['POS','CHROM','G']:
   w=pulse(x,fs,method);peaks=peak_info(*psd(w,fs),k=3)
   if not peaks:failures.append(dict(roi=name,method=method,reason='static_or_no_finite_peak'));continue
   for p in peaks:
    p['background_flag']=any(abs(p['hz']-z)<=.05 for z in bg)
    p['codec_flag']=any(abs(p['hz']-z)<=.05 for z in artifact_hz)
    p['weight']=p['power']*(.25 if p['background_flag'] and ablation!='no_background' else 1)*(.25 if p['codec_flag'] and ablation!='no_codec' else 1)
   top=peaks[0];ratio=top['power']/max(peaks[1]['power'],1e-30) if len(peaks)>1 else 1.
   candidates.append(dict(roi=name,method=method,peaks=peaks[:1] if ablation=='K1' else peaks,features=[top['snr_db'],float(np.log(max(ratio,1e-12))),top['concentration'],0.]))
   waves.append((q,w,fs))
 pos=[c['peaks'][0]['hr'] for c in candidates if c['method']=='POS'];disp=float(np.std(pos)) if len(pos)>1 else 60.
 for c in candidates:c['features'][3]=-disp
 if ablation!='no_viterbi' and len(waves)==9 and all(len(w)/fs>=20 for q,w,fs in waves):
  start=max(q[0] for q,w,fs in waves);end=min(q[-1] for q,w,fs in waves);emis=[]
  for st in np.arange(start,end-10+1e-6,1.):
   spectra=[]
   for q,w,fs in waves:
    ww=w[(q>=st)&(q<st+10)];ff,pp=psd(ww,fs);v=np.interp(GRID/60,ff,pp)
    if ablation!='no_background':v*=np.where(np.any(abs(GRID[:,None]/60-np.asarray(bg)[None,:])<=.05,axis=1),.25,1.) if bg else 1.
    if ablation!='no_codec':v*=np.where(np.any(abs(GRID[:,None]/60-np.asarray(artifact_hz)[None,:])<=.05,axis=1),.25,1.) if len(artifact_hz) else 1.
    spectra.append(v/max(v.sum(),1e-30))
   emis.append(np.log(np.maximum(np.mean(spectra,axis=0),1e-30)))
  if emis:
   transition=.05*abs(GRID[:,None]-GRID[None,:]);score=emis[0];back=[]
   for e in emis[1:]:a=score[:,None]-transition;j=np.argmax(a,axis=0);back.append(j);score=a[j,np.arange(len(GRID))]+e
   j=int(np.argmax(score));path=[j]
   for z in back[::-1]:j=int(z[j]);path.append(j)
   hr=float(np.median(GRID[path]));feat=np.mean([c['features'] for c in candidates],axis=0).tolist()
   candidates.append(dict(roi='multiROI',method='Viterbi',features=feat,peaks=[dict(hr=hr,hz=hr/60,power=1.,weight=1.,background_flag=False,codec_flag=False)]))
 return dict(candidates=candidates,background=background,failures=failures)
def person_weights(records):
 counts={p:sum(r['person']==p for r in records) for p in {r['person'] for r in records}}
 return np.array([1/counts[r['person']]/len(counts) for r in records])
def feature_scaler(records):
 xx=[];ww=[];rw=person_weights(records)
 for r,w in zip(records,rw):
  cs=r['candidates']
  for c in cs:xx.append(c['features']);ww.append(w/max(len(cs),1))
 if not xx:return np.zeros(4),np.ones(4)
 xx=np.array(xx);ww=np.array(ww);ww/=ww.sum();mu=np.sum(xx*ww[:,None],axis=0);sd=np.sqrt(np.sum((xx-mu)**2*ww[:,None],axis=0));return mu,np.maximum(sd,1e-6)
def sqi(c,model):return float(np.mean(np.clip((np.array(c['features'])-model['mu'])/model['sd'],-3,3)))
def gaussian_mixture(c,theta,sigma):
 weights=np.array([p['weight'] for p in c['peaks']],float);weights/=max(weights.sum(),1e-30)
 y=np.array([p['hr'] for p in c['peaks']]);theta=np.atleast_1d(theta)
 return np.sum(stats.norm.pdf(theta[:,None],y[None,:],sigma)*weights,axis=1)
def fit(records,ablation='none'):
 """Fit strictly train people. Outlier NLL is a pseudo-likelihood, not calibrated truth."""
 assert records;mu,sd=feature_scaler(records);model=dict(mu=mu,sd=sd,ablation=ablation)
 rw=person_weights(records);truth=np.array([r['truth'] for r in records]);n=len({r['person'] for r in records})
 mean=np.sum(rw*truth);st=np.sqrt(np.sum(rw*(truth-mean)**2));ix=np.argsort(truth);quant=np.interp([.25,.75],np.cumsum(rw[ix]),truth[ix]);iqr=quant[1]-quant[0]
 scale=min(st,iqr/1.34) if iqr>0 else st;bw=max(.9*scale*n**(-.2),1.)
 prior=np.sum(stats.norm.pdf(GRID[:,None],truth[None,:],bw)*rw,axis=1);prior/=prior.sum()
 if ablation=='no_prior':prior=np.ones(len(GRID))/len(GRID)
 observations=[]
 for r,w in zip(records,rw):
  for c in r['candidates']:observations.append((c,float(r['truth']),w/max(len(r['candidates']),1),sqi(c,model)))
 if observations:
  threshold=np.quantile([x[3] for x in observations],.75);errors=[c['peaks'][0]['hr']-y for c,y,w,q in observations if q>=threshold]
  sigma=float(np.clip(1.4826*np.median(abs(np.array(errors)-np.median(errors))),3,30))
  qs=np.array([o[3] for o in observations]);ww=np.array([o[2] for o in observations]);ww/=ww.sum()
  g=np.array([gaussian_mixture(c,y,sigma)[0] for c,y,w,q in observations])
  def fun(z):
   ep=special.expit(z[0]+z[1]*qs);return -np.sum(ww*np.log(np.maximum((1-ep)*g+ep/138,1e-30)))
  result=optimize.minimize(fun,[0.,0. if ablation=='constant_epsilon' else -1.],method='L-BFGS-B',bounds=[(-10,10),(0,0) if ablation=='constant_epsilon' else (-10,0)])
  if not result.success:raise ValueError('epsilon optimization failed:'+result.message)
  a,b=map(float,result.x);loss=float(result.fun)
 else:sigma=30.;a=10.;b=0.;loss=None
 model.update(prior=prior,sigma=sigma,a=a,b=b,bandwidth=bw,training_people=sorted({r['person'] for r in records}),loss=loss,constant=float(np.median(truth)))
 return model
def predict(candidates,model):
 logp=np.log(np.maximum(model['prior'],1e-300));parts=[]
 for c in candidates:
  q=sqi(c,model);eps=float(special.expit(model['a']+model['b']*q));lik=(1-eps)*gaussian_mixture(c,GRID,model['sigma'])+eps/138
  logp+=np.log(np.maximum(lik,1e-300))/len(candidates);parts.append(dict(roi=c['roi'],method=c['method'],sqi=q,epsilon=eps))
 p=np.exp(logp-special.logsumexp(logp));cdf=np.cumsum(p)
 quant=lambda q:float(GRID[min(np.searchsorted(cdf,q),len(GRID)-1)])
 return dict(hr=quant(.5),lower90=quant(.05),upper90=quant(.95),pulse_observed=bool(candidates),contributors=parts,posterior=p)
