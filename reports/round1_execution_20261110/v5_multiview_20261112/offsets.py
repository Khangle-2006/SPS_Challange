"""Versioned DEV60 sideview phase alignment; no images/labels mutated."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,hashlib,traceback,fcntl
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.signal import find_peaks
P=Path(__file__).resolve().parent;E=P.parent;R=E.parent.parent
sys.path.insert(0,'/data_pool/khangle/facevitals/code')
from mcd_figure5_metrics import filtered,pos_unfiltered
from mcd_errata_audit import timed
A=R/'mapping_audit_20261030';OLD=A/'lag_pos_ppg_v3';DATA=R/'datasets/MCD-rPPG'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(a,indent=2,allow_nan=False));q.replace(p)
def decision(curve,fs):
    curve=np.asarray(curve);shifts=np.arange(-fs,fs+1);best=min(np.flatnonzero(curve>=curve.max()-1e-12),key=lambda j:(abs(int(shifts[j])),int(shifts[j])))
    peaks=find_peaks(curve)[0].tolist()
    if curve[0]>curve[1]:peaks.append(0)
    if curve[-1]>curve[-2]:peaks.append(len(curve)-1)
    secondary=max((float(curve[j]) for j in peaks if abs(shifts[j]-shifts[best])>=np.ceil(.25*fs)),default=None)
    reasons=[]
    if curve[best]<.3:reasons.append('weak_peak')
    if secondary is not None and secondary>=.95*curve[best]:reasons.append('ambiguous_peak')
    if best in (0,len(curve)-1):reasons.append('boundary_unresolved')
    return dict(lag_frames=int(shifts[best]),offset_seconds=float(shifts[best]/fs),peak=float(curve[best]),secondary_peak=secondary,reasons=reasons,curve=curve.tolist())
def curve_of(x,y,fs,do_filter=True):
    if do_filter:x,y=filtered(x,fs),filtered(y,fs)
    core=np.arange(3*fs,len(x)-3*fs);ref=y[core]-y[core].mean();den=np.linalg.norm(ref)
    if len(core)<10*fs or den<1e-12:raise ValueError('degenerate/short reference')
    c=[]
    for k in range(-fs,fs+1):
        z=x[core+k]-x[core+k].mean();d=den*np.linalg.norm(z)
        if d<1e-12:raise ValueError('degenerate POS')
        c.append(float(z@ref/d))
    return c
def tests():
    rng=np.random.default_rng(2027);y=rng.normal(size=2400);x=np.roll(y,7)
    d=decision(curve_of(x,y,30,False),30);assert d['lag_frames']==7
    # Target(u)=PPG(u-lag); shifting raw PPG timestamps +lag is identical.
    t=np.arange(len(y))/30;q=t[100:200];assert np.allclose(np.interp(q,t+7/30,y),np.interp(q-7/30,t,y))
    assert 'weak_peak' in decision(np.linspace(.01,.2,61),30)['reasons']
    assert 'boundary_unresolved' in decision(np.linspace(0,1,61),30)['reasons']
    z=np.zeros(61);z[20]=1;z[40]=.96;assert 'ambiguous_peak' in decision(z,30)['reasons']
    return dict(passed=True,tests=['positive_delay','target_shift_sign','weak','boundary','ambiguity'])
def one(row,fold):
    rid=Path(row['video']).stem;gate=read(A/'dev60_gate_v3'/f'{rid}.json');receipt=read(OLD/rid/'trace_receipt.json');tr=OLD/rid/'trace.npz'
    assert sha(tr)==receipt['trace_sha256'] and receipt['source_sha256']==gate['source_sha256']
    n=gate['gate']['candidate_prefix_frames']
    with np.load(tr) as z:rgb=z['rgb'][:n].mean(1);valid=z['valid'][:n].copy()
    _,mt=timed(DATA/row['meta']);pv,pt=timed(DATA/row['ppg']);mt=mt[:n]
    t=(mt-mt[0])/1e6;pt=(pt-mt[0])/1e6
    assert np.all(np.diff(t)>0) and np.all(np.diff(pt)>0)
    fs=24 if row['camera']=='IriunWebcam' else 30
    wins=[];curves=[]
    for start in range(0,int(t[-1]//30)*30,30):
        g=start+np.arange(fs*30)/fs;r=np.searchsorted(t,g).clip(0,n-1);l=(r-1).clip(0,n-1);l=np.where(np.isclose(g,t[r],rtol=0,atol=1e-10),r,l)
        w=dict(start=start)
        try:
            assert g[0]>=t[0] and g[-1]<=t[-1] and g[0]>=pt[0] and g[-1]<=pt[-1], 'clock_support'
            assert (valid[l]&valid[r]&((t[r]-t[l])<=.2)).all(),'face_or_gap'
            xx=np.stack([np.interp(g,t,rgb[:,k]) for k in range(3)],1);yy=-np.interp(g,pt,pv)
            c=curve_of(-pos_unfiltered(xx,fs),yy,fs);curves.append(c);w.update(decision(c,fs))
        except (AssertionError,ValueError) as e:w['failure']=str(e)
        wins.append(w)
    result=dict(recording=rid,person=row['patient_id'],camera=row['camera'],step=row['step'],fold=fold,fs=fs,windows=wins,valid_windows=len(curves),source_sha256=gate['source_sha256'],trace_sha256=receipt['trace_sha256'],metadata_sha256=sha(DATA/row['meta']),ppg_sha256=sha(DATA/row['ppg']),raw_row=row)
    if curves:result.update(decision(np.mean(curves,axis=0),fs))
    else:result.update(reasons=['no_valid_windows'],offset_seconds=None)
    if len(curves)<2:result['reasons'].append('fewer_than_two_windows')
    result['eligible']=not result['reasons'];return result
def main():
    lock=(P/'offsets.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (P/'OFFSETS_COMPLETE.json').exists(),'Already complete; no rerun'
    save(P/'OFFSET_TESTS.json',tests());index=read(R/'t8_oval_20261031/training_index.json')
    # Printed/schema checked before production; actual frozen person folds, not official folds.
    folds={str(person):int(f['fold']) for f in index['folds']['folds'] for person in f['mcd_val_subjects']}
    rows=[r for r in read(OLD/'eligible_records.json') if r['camera'] in ('IriunWebcam','USBVideo')]
    assert len(rows)==238 and len(folds)==60
    results=[]
    for row in rows:
        rid=Path(row['video']).stem
        try:a=one(row,folds[str(row['patient_id'])])
        except Exception as e:a=dict(recording=rid,person=row['patient_id'],camera=row['camera'],fold=folds[str(row['patient_id'])],eligible=False,reasons=['processing_error'],error=str(e),traceback=traceback.format_exc())
        save(P/'offsets'/f'{rid}.json',a);results.append(a);save(P/'OFFSET_STATUS.json',dict(pid=os.getpid(),processed=len(results),total=len(rows)))
    save(P/'OFFSETS.json',results)
    summary=[]
    from collections import Counter
    for camera in ['IriunWebcam','USBVideo']:
        rr=[r for r in results if r['camera']==camera];ok=[r for r in rr if r['eligible']];vals=[r['offset_seconds'] for r in ok]
        summary.append(dict(camera=camera,total=len(rr),eligible=len(ok),people=len(set(r['person'] for r in ok)),offset_percentiles=np.percentile(vals,[0,25,50,75,100]).tolist() if vals else None,reasons=dict(Counter(reason for r in rr for reason in r['reasons']))))
    save(P/'OFFSETS_COMPLETE.json',dict(utc=datetime.now(timezone.utc).isoformat(),summary=summary,protocol_sha256=sha(P/'PROTOCOL.md'),code_sha256=sha(__file__),results_sha256=sha(P/'OFFSETS.json'),training_launched=False,holdout_evaluations=0));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'OFFSET_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
