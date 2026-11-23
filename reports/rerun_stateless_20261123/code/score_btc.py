"""Exact y11 frozen-native scoring logic on the new spectra, with explicit native row."""
from expert_spectrum import *
import csv,traceback
started=time.monotonic()
try:
 mode=selected_mode();assert (OUT/'inference_COMPLETE.json').exists();assert not (OUT/'STOP.json').exists()
 import y06_reader_quality as Q
 from lib_btc import GRID,zlog
 N=Q.N;Zn,okn,qn=Q.Zn,Q.okn,Q.qn;low,lab,per=Q.low,Q.lab,Q.per
 def zmodel(S):
  ok=np.isfinite(S).all(1);Z=np.zeros((N,len(GRID)));Z[ok]=zlog(S[ok]);return Z,ok
 def load_models(cond):
  out={}
  for n in MODELS:
   p=OUT/'native'/(n+'.npz') if cond=='native' else OUT/'btc_conditions'/n/(cond+'.npz')
   with np.load(p) as z:out[n]=zmodel(z['S'])
  return out
 def mix(Zc,ZM):return (2*Zc+sum(ZM[m][0] for m in MODELS))/5
 def frozen(Zte,sub,Ztr):return Q.cv([('seen','seen')],sub,Zte,qn,low,Ztr,qn,low,lab,{})[0]
 NM=load_models('native');okm=np.all([NM[m][1] for m in MODELS],0);sub=okn&okm;assert sub.sum()==218
 Z4n=mix(Zn,NM);S=lambda a,b,m:Q.summarize(a[m],lab[m],per[m],b[m])
 old={r['condition']:r for r in csv.DictReader((W/'out/y11_v4_conditions.csv').open())}
 calmap={r['recording']:float(r['prediction']) for r in Q.rows if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'}
 cal=np.array([calmap[r] for r in Q.rec]);rows=[];predictions={}
 conds=['native','split10','track_c1_g25p','aidtrack_c1_g25gv01','c2_vit','c4_vga','c4_qvga','c6_us1','track_c7_blue']
 for cond in conds:
  if cond=='native':Zc=Zn;CM=NM;m=sub;cov=1.
  else:
   CM=load_models(cond)
   if cond=='split10':
    import y08_fps_sim as F
    # Same F.one and data; serial evaluation changes no numerical operator.
    res=[F.one(k) for k in F.KEYS];d=dict(C=np.array([r['skip10'][0] for r in res]),P=np.array([r['skip10'][1] for r in res]),have=np.ones(N,bool))
   else:d=Q.load('x09' if (W/'out/x09_btc'/(cond+'.npz')).exists() else 'x01',cond)
   Zc,okc,_=Q.evidence(d);m=sub&d['have']&okc&np.all([CM[n][1] for n in MODELS],0)
   cov=float((okc&np.all([CM[n][1] for n in MODELS],0))[sub&d['have']].mean())
   assert int(m.sum())==int(old[cond]['videos']),(cond,'scored video count changed',int(m.sum()),old[cond]['videos'])
  # Preserve y11's exact native-training subset for each matched condition cohort.
  p3=frozen(Zc,m,Zn);p4=frozen(mix(Zc,CM),m,Z4n);n4=frozen(Z4n,m,Z4n)
  a=S(p4,n4,m);c=S(p4,Q.const,m);e=S(p4,p3,m);b=S(Q.const,Q.const,m);cc=S(cal,cal,m)
  g=('a' if cov>=.9 else '-')+('b' if c['dCI'][0]<=0 else '-')+('c' if c['dCI'][1]<0 else '-')
  oldmae=8.350176695910013 if cond=='native' else float(old[cond]['v4_MAE'])
  rows.append(dict(condition=cond,videos=int(m.sum()),coverage=cov,v4_MAE=a['MAE'],lo=a['CI'][0],hi=a['CI'][1],RMSE=a['RMSE'],r=a['r'],d_native=a['delta'],dn_lo=a['dCI'][0],dn_hi=a['dCI'][1],d_const=c['delta'],dc_lo=c['dCI'][0],dc_hi=c['dCI'][1],d_v3=e['delta'],d3_lo=e['dCI'][0],d3_hi=e['dCI'][1],gates=g,old_v4_MAE=oldmae,shift_vs_old=a['MAE']-oldmae,constant_MAE=b['MAE'],calibrated_native_historical_MAE=cc['MAE'],reader_v3_MAE=e['MAE']-e['delta'],mode=mode))
  predictions.update({cond+'__'+k:v for k,v in dict(v4=p4,v3=p3,native_v4=n4,mask=m,constant=Q.const,calibrated=cal).items()})
  if cond=='native':
   expected=9.130137933212104 if mode=='reset' else json.loads((OUT/'MODE.json').read_text())['results']['self-warm (z17)']['mae']
   assert abs(a['MAE']-expected)<=.001,('native mismatch',a['MAE'],expected)
  print(json.dumps(rows[-1]),flush=True)
 with (OUT/'btc_v4_conditions.csv').open('x',newline='') as f:w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(OUT/'BTC_PREDICTIONS_PRIVATE.npz',**predictions,truth=lab,person=per,recording=Q.rec,fold=Q.fold)
 save(OUT/'BTC_SCORE_COMPLETE.json',dict(mode=mode,seconds=time.monotonic()-started,rows=len(rows),native_mae=rows[0]['v4_MAE'],native_exact_check=True))
except BaseException as e:
 save(OUT/'BTC_SCORE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));save(OUT/'STOP.json',dict(stage='BTC_scoring',error=repr(e)));raise
