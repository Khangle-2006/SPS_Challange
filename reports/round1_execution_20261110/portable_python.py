"""Numerical reference for the toolbox-free POS/Welch packaging baseline only."""
import numpy as np
def pos(x,L=48):
 h=np.zeros(len(x))
 for last in range(L,len(x)+1):
  block=x[last-L:last];m=block.mean(0)
  if np.any(abs(m)<1e-10):continue
  s=np.array([[0,1,-1],[-2,1,1]])@(block/m).T;den=np.std(s[1],ddof=1)
  if den<1e-10:continue
  z=s[0]+np.std(s[0],ddof=1)/den*s[1];h[last-L:last]+=z-z.mean()
 return h
def hr(w):
 if not np.isfinite(w).all() or np.std(w,ddof=1)<1e-10:return np.nan
 tt=np.linspace(-1,1,len(w));w=w-w.mean();w=w-np.dot(tt,w)/np.dot(tt,tt)*tt;window=.5-.5*np.cos(2*np.pi*np.arange(128)/128)
 p=np.mean([abs(np.fft.rfft((w[s:s+128]-w[s:s+128].mean())*window,4096))**2 for s in range(0,len(w)-128+1,64)],axis=0);f=np.fft.rfftfreq(4096,1/30);sel=(f>=.7)&(f<=3)
 return float(60*f[sel][np.argmax(p[sel])]) if np.isfinite(p).all() and p[sel].sum()>1e-30 else np.nan
def predict(t,rgb,valid):
 rows=[]
 for start in np.arange(0,t[-1]-10+1e-8,10):
  j0=np.where(t<=start)[0];j1=np.where(t>=start+10)[0];p=np.nan
  if len(j0) and len(j1):
   a,b=j0[-1],j1[0]
   if valid[a:b+1].all() and (np.diff(t[a:b+1])<=.2+1e-8).all():
    q=start+np.arange(300)/30;x=np.stack([np.interp(q,t,rgb[:,k]) for k in range(3)],1);p=hr(pos(x))
  rows.append([float(start),float(start+10),p,float(np.isfinite(p))])
 return np.array(rows)
