from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
def summarize(rr,other=None):
 if other is not None:
  by={r['recording']:r for r in other};rr=[r for r in rr if r['recording'] in by]
 per=defaultdict(list)
 for r in rr:per[r['person']].append(r)
 if not per:return dict(people=0,videos=0)
 values=[];truths=[];preds=[]
 for p in sorted(per):
  rs=per[p];e=np.array([r['prediction']-r['truth'] for r in rs]);v=np.array([np.abs(e).mean(),np.sqrt(np.mean(e*e)),e.mean()])
  if other is not None:
   o=np.array([by[r['recording']]['prediction']-r['truth'] for r in rs]);v-=np.array([np.abs(o).mean(),np.sqrt(np.mean(o*o)),o.mean()])
  values.append(v);truths.append(np.mean([r['truth'] for r in rs]));preds.append(np.mean([r['prediction'] for r in rs]))
 values=np.array(values);ix=np.random.default_rng(0).integers(0,len(values),(10000,len(values)));ci=np.quantile(values[ix].mean(1),[.025,.975],axis=0).T
 ans=dict(people=len(per),videos=len(rr),MAE=float(values[:,0].mean()),RMSE_mean_person=float(values[:,1].mean()),bias=float(values[:,2].mean()),CI95=ci.tolist())
 if other is None:
  ans['Pearson_person_means']=float(pearsonr(truths,preds).statistic) if len(per)>2 and np.ptp(truths)>1e-10 and np.ptp(preds)>1e-10 else None
  ans['LoA_person_mean_errors']=(values[:,2].mean()+np.array([-1.96,1.96])*values[:,2].std(ddof=1)).tolist() if len(per)>1 else None
  if ans['Pearson_person_means'] is not None:
   xx=np.array(truths)[ix];yy=np.array(preds)[ix];xx-=xx.mean(1,keepdims=True);yy-=yy.mean(1,keepdims=True);den=np.sqrt(np.sum(xx*xx,axis=1)*np.sum(yy*yy,axis=1));valid=den>1e-12;rho=np.sum(xx[valid]*yy[valid],axis=1)/den[valid]
   ans['CI95_Pearson']=np.quantile(rho,[.025,.975]).tolist() if len(rho) else None;ans['Pearson_bootstrap_valid']=int(valid.sum())
  if len(per)>1:
   eb=values[:,2][ix];means=eb.mean(1);sd=eb.std(1,ddof=1);ans['CI95_LoA_lower']=np.quantile(means-1.96*sd,[.025,.975]).tolist();ans['CI95_LoA_upper']=np.quantile(means+1.96*sd,[.025,.975]).tolist()
 return ans
