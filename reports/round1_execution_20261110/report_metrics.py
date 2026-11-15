import numpy as np
from scipy.stats import spearmanr
def belongs(r,g):
 if g=='all':return True
 if g=='Resting':return r['task']=='Resting'
 if g in ['Task','Walking']:return r['task'].lower().startswith('task') if g=='Task' else r['task']=='Walking'
 return r['task']=='Resting' and r['group']==g.split('-')[0]

def metric(rows,methods):
 look={m:{r['recording']:r for r in rows if r['method']==m and r['prediction'] is not None} for m in methods};keys=sorted(set.intersection(*(set(a) for a in look.values())))
 if not keys:return []
 for k in keys:
  assert len({(look[m][k]['person'],float(look[m][k]['truth'])) for m in methods})==1,'Mismatched paired reference'
 people=sorted({look[methods[0]][k]['person'] for k in keys});arr={};bias={};rho={}
 for m in methods:
  ee=[];bb=[];pp=[];tt=[]
  for person in people:
   rr=[look[m][k] for k in keys if look[m][k]['person']==person];e=np.array([float(r['prediction'])-float(r['truth']) for r in rr]);ee.append(float(abs(e).mean()));bb.append(float(e.mean()));pp.append(float(np.mean([r['prediction'] for r in rr])));tt.append(float(np.mean([float(r['truth']) for r in rr])))
  arr[m]=np.array(ee);bias[m]=np.array(bb);r=float(spearmanr(pp,tt).statistic) if len(pp)>2 and np.std(pp)>0 and np.std(tt)>0 else float('nan');rho[m]=r if np.isfinite(r) else None
 ix=np.random.default_rng(0).integers(len(people),size=(10000,len(people)));ci=lambda a:np.quantile(a[ix].mean(1),[.025,.975]).tolist()
 return [dict(method=m,videos=len(keys),people=len(people),MAE=float(arr[m].mean()),CI95=ci(arr[m]),P95_person_MAE=float(np.quantile(arr[m],.95)),bias=float(bias[m].mean()),bias_CI95=ci(bias[m]),Spearman=rho[m],delta_constant=float((arr[m]-arr['constant']).mean()),delta_constant_CI=ci(arr[m]-arr['constant']),delta_calibrated=float((arr[m]-arr['calibrated']).mean()),delta_calibrated_CI=ci(arr[m]-arr['calibrated'])) for m in methods]
