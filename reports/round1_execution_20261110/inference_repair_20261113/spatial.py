import numpy as np,cv2
def turn_points(points,k,w,h):
 p=np.asarray(points).copy();x=p[:,0].copy();y=p[:,1].copy()
 if k==1:p[:,0]=y;p[:,1]=w-1-x
 elif k==2:p[:,0]=w-1-x;p[:,1]=h-1-y
 elif k==3:p[:,0]=h-1-y;p[:,1]=x
 return p
def orientation(points):
 l=points[[33,133]].mean(0);r=points[[362,263]].mean(0);mouth=points[[13,14,61,291]].mean(0);eye=r-l;down=mouth-(l+r)/2
 def rotate(v,k):
  for _ in range(k):v=np.array([v[1],-v[0]])
  return v
 scores=[]
 for k in range(4):
  e=rotate(eye,k);d=rotate(down,k);scores.append(abs(e[0])/max(np.linalg.norm(e),1e-9)+d[1]/max(np.linalg.norm(d),1e-9))
 return int(np.argmax(scores)),scores
def seed_pixels(image,points,oval):
 l=points[[33,133]].mean(0);r=points[[362,263]].mean(0);m=points[[13,14,61,291]].mean(0);e=r-l;down=m-(l+r)/2
 centers=[(l+r)/2-.65*down,l+.55*down-.12*e,r+.55*down+.12*e];mask=np.zeros(oval.shape,np.uint8)
 for c in centers:cv2.circle(mask,tuple(np.rint(c).astype(int)),max(2,round(.12*np.linalg.norm(e))),1,-1)
 return image[(mask>0)&(oval>0)]
def color_fit(pixels):
 a=np.asarray(pixels,float);rg=a[:,:2]/np.maximum(a.sum(1,keepdims=True),1);center=np.median(rg,axis=0);scale=np.maximum(1.4826*np.median(abs(rg-center),axis=0),.01);lum=a.mean(1);lc=float(np.median(lum));ls=max(float(1.4826*np.median(abs(lum-lc))),5.)
 return dict(center=center.tolist(),scale=scale.tolist(),luma_median=lc,luma_mad=ls,seed_pixels=len(a))
def color_mask(image,oval,fit):
 a=image.astype(float);rg=a[:,:,:2]/np.maximum(a.sum(2,keepdims=True),1);dist=(((rg-fit['center'])/fit['scale'])**2).sum(2);lum=a.mean(2);ach=(a.max(2)-a.min(2))/np.maximum(lum,1)
 bad=(dist>9.21)|((ach<.12)&(lum>fit['luma_median']+3*fit['luma_mad']));bad &= oval>0;keep=(oval>0)&~bad
 fraction=float(bad.sum()/max((oval>0).sum(),1));valid=keep.sum()>=.2*(oval>0).sum() and keep.sum()>=16
 out=image.copy()
 if valid:out[bad]=np.rint(a[keep].mean(0)).clip(0,255).astype(np.uint8)
 return out,bad,fraction,bool(valid)
