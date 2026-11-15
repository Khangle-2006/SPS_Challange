"""Descriptive GOP/background association, never a causal codec/label classifier.
Tarassenko2014 motivates background control; this is Welch, not their AR model.
"""
from common import *
import numpy as np
from pipeline import pulse,psd,peak_info,native_grid
sys.path.insert(0,str(OLD));from infer_dev import classical_functions
def main():
 cohort=read(OLD/'diagnosis205_20261108/COHORT.json')['recordings'];videos=dev_manifest();vv={v['recording']:v for v in videos};ns,_=classical_functions();dest=OUT/'association';dest.mkdir(exist_ok=True)
 reports=[];exclusions=[];background=[];order=read(OUT/'TRACE_ORDER.json') if (OUT/'TRACE_ORDER.json').exists() else {}
 selected=set(order.get('background_top20',[])+order.get('background_worst20',[]))
 for rec in cohort:
  gp=OUT/'native_gop'/(rec+'.json')
  if not gp.exists():continue
  g=read(gp);v=vv[rec];out=dest/(rec+'.json')
  if out.exists():r=read(out)
  else:
   try:
    with np.load(OLD/'dev'/v['drive_id']/'native/meta.npz') as z:t=z['t'];rgb=z['rgb_mean'];valid=z['valid'].astype(bool)
    if valid.sum()<2 or np.max(np.diff(t[valid]))>.2:raise ValueError('legacy_whole_video_support_gap')
    q=np.arange(t[valid][0],t[valid][-1],1/30);x=np.stack([np.interp(q,t[valid],rgb[valid,c]) for c in range(3)],axis=1);w=ns['pos'](x,30)
    f,p=psd(w,30);peaks=peak_info(f,p)
    if not peaks:raise ValueError('no_POS_peak')
    peak=peaks[0];hz=[x['hz'] for x in g['artifact_frequencies']];match=any(abs(peak['hz']-a)<=.05 for a in hz)
    train=[x for x in videos if x['fold']!=v['fold']];people={x['person'] for x in train};expected=np.mean([np.mean([any(abs(x['pr_bpm']/60-a)<=.05 for a in hz) for x in train if x['person']==person]) for person in people])
    r=dict(recording=rec,person=v['person'],POS_whole_hr=float(peak['hr']),truth=v['pr_bpm'],signed_error=float(peak['hr']-v['pr_bpm']),codec_match=match,expected_train_person_match=float(expected),fps=g['fps'],keyint_frames=g['keyint_frames'],I_period_s=g['I_period_s'],miniGOP_frames=g['miniGOP_frames'],near60=bool(58<=peak['hr']<=62),I_period_near1s=g['I_period_s'] is not None and abs(g['I_period_s']-1)<=.1,top20=rec in order.get('background_top20',[]),whole_POS_definition='legacy RGB36 interpolated30Hz; Welch full valid span, not median frozen10s predictions')
    save(out,r);csvout(dest/(rec+'_legacy_skin_PSD.csv'),[dict(hz=float(a),power=float(b)) for a,b in zip(f,p)])
   except ValueError as e:exclusions.append(dict(recording=rec,reason=str(e)));continue
  reports.append(r)
  d=OUT/'traces'/rec/'native';bp=dest/(rec+'_background.json')
  if rec not in selected or not (d/'COMPLETE.json').exists():continue
  if bp.exists():background.append(read(bp));continue
  try:
   with np.load(d/'trace.npz') as z:t=z['t'];rgb=z['rgb']
   q,x,fs=native_grid(t,rgb[:,3]);f,p=psd(pulse(x,fs,'POS'),fs);pk=peak_info(f,p)
   if not pk:raise ValueError('background_static_or_absent')
   b=pk[0];row=dict(recording=rec,skin_whole_hr=r['POS_whole_hr'],background_hr=float(b['hr']),background_snr_db=b['snr_db'],delta_hz=float(r['POS_whole_hr']/60-b['hz']),coincident=bool(abs(r['POS_whole_hr']/60-b['hz'])<=.05),strong_background=b['snr_db']>=6,truth=v['pr_bpm'],skin_abserror=abs(r['signed_error']))
   csvout(dest/(rec+'_background_PSD.csv'),[dict(hz=float(a),power=float(b)) for a,b in zip(f,p)]);save(bp,row);background.append(row)
  except ValueError as e:exclusions.append(dict(recording=rec,reason='background:'+str(e)))
 csvout(OUT/'CODEC_ASSOCIATION.csv',reports);csvout(OUT/'BACKGROUND_ASSOCIATION.csv',background);csvout(OUT/'ASSOCIATION_EXCLUSIONS.csv',exclusions)
 def summary(rr):
  return dict(n=len(rr),observed_match=float(np.mean([x['codec_match'] for x in rr])) if rr else None,expected_train_match=float(np.mean([x['expected_train_person_match'] for x in rr])) if rr else None,near60_I1_count=sum(x['near60'] and x['I_period_near1s'] for x in rr))
 save(OUT/'ASSOCIATION_STATUS.json',dict(utc=now(),total_target205=len(cohort),gop_files_available=sum((OUT/'native_gop'/(r+'.json')).exists() for r in cohort),valid_whole_POS=len(reports),all=summary(reports),top20=summary([r for r in reports if r['top20']]),background_available=len(background),background_target=len(selected),complete=False,interpretation='Coincidence is association only; no exclusive codec-versus-light or label causation'))
if __name__=='__main__':main()
