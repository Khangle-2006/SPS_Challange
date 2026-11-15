"""Frozen E0 POS replay, CHROM comparison and reference diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, json, math, sys, traceback, time
from pathlib import Path
import numpy as np
from scipy import signal
from hr_baseline import pos, estimate_hr
from mcd_e0_metrics import sample_grid, ARMS
from mcd_errata_audit import timed

def save(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def chrom(rgb, fs=30):
    rgb=np.asarray(rgb,dtype=float)
    if rgb.ndim!=2 or rgb.shape[1]!=3 or not np.isfinite(rgb).all():raise ValueError('Invalid RGB')
    n=len(rgb); w=math.ceil(1.6*fs); w+=w%2
    if n<w:raise ValueError('Short RGB')
    b,a=signal.butter(3,[.7,3],fs=fs,btype='bandpass')
    starts=list(range(0,n-w+1,w//2))
    if starts[-1]!=n-w:starts.append(n-w)
    out=np.zeros(n)
    for start in starts:
        v=rgb[start:start+w];mean=v.mean(axis=0)
        if np.any(mean<=0):raise ValueError('Nonpositive RGB mean')
        r,g,bl=(v/mean).T
        x=signal.filtfilt(b,a,3*r-2*g)
        y=signal.filtfilt(b,a,1.5*r+g-1.5*bl)
        if np.std(y)<1e-12:raise ValueError('Degenerate chrominance')
        out[start:start+w]+=(x-np.std(x)/np.std(y)*y)*signal.windows.hann(w)
    return out

def tests():
    t=np.arange(300)/30
    # Distinct pulse channel sensitivities plus achromatic slow illumination.
    rgb=np.array([120.,90.,70.])[None,:]*(1+np.sin(2*np.pi*1.2*t)[:,None]*np.array([.003,.01,.002])+ .005*np.sin(2*np.pi*.2*t)[:,None])
    for f in (pos,chrom):
        p=f(rgb,30); assert len(p)==300
        assert abs(estimate_hr(p,30)-72)<1
        assert np.allclose(p,f(rgb*np.array([.6,1.2,.8]),30),atol=1e-8)
    try:chrom(np.ones((300,3)))
    except ValueError:pass
    else:raise AssertionError('constant RGB not rejected')
    print('PASS: known frequency, all300 samples, channel-gain invariance, degeneracy',flush=True)

def run(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    records=json.loads((a.e0/'results.json').read_text())
    cohort={r['video']:r for r in json.loads((a.e0/'cohort.json').read_text())}
    expected=[(r['video'],w['index']) for r in records for w in r['windows'] if w['reference_hr'] is not None and all(w['predictions'][arm] is not None for arm in ARMS)]
    assert len(records)==119 and len(expected)==1860 and len(set(expected))==1860
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'protocol_sha256':sha(a.protocol),'script_sha256':sha(Path(__file__)), 'e0_results_sha256':sha(a.e0/'results.json'),'expected_windows':1860,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save(out/'manifest.json',manifest)
    allrows=[]; failures=[]
    for num,record in enumerate(records,1):
        row=cohort[record['video']];stem=Path(row['video']).stem
        selected=[w for w in record['windows'] if (row['video'],w['index']) in expected]
        try:
            tp=a.e0/stem/'trace.npz';assert sha(tp)==record['trace_sha256']
            rgb=np.load(tp)['rgb'];n=len(rgb)
            assert sha(a.root/row['meta'])==record['metadata_sha256']
            assert sha(a.root/row['ppg_sync'])==record['sync_sha256']
            _,mt=timed(a.root/row['meta']);mt=mt[:n];t=(mt-mt[0])/1e6
            sy=np.loadtxt(a.root/row['ppg_sync'],ndmin=2)[:n,0]
            pv,pt=timed(a.root/row['ppg']);support=(mt>=pt.min())&(mt<=pt.max())
            grid=np.arange(int(np.floor(t[-1]*30))+1)/30
            pg,pok=sample_grid(t,sy,support&np.isfinite(sy),grid)
            rg,rok=sample_grid(t,rgb[:,0].mean(axis=1),np.isfinite(rgb[:,0]).all(axis=(1,2)),grid)
            regions=[sample_grid(t,rgb[:,0,i],np.isfinite(rgb[:,0,i]).all(axis=1),grid) for i in range(3)]
            raw_valid=np.all(np.diff(pt)>0)
            raw,rawok=sample_grid((pt-mt[0])/1e6,pv,np.isfinite(pv),grid) if raw_valid else (np.zeros(len(grid)),np.zeros(len(grid),dtype=bool))
            result=[]
            for w in selected:
                sel=(grid>=w['start_s'])&(grid<w['start_s']+10)
                assert sel.sum()==300 and pok[sel].all() and rok[sel].all()
                ref=estimate_hr(pg[sel],30); pred=estimate_hr(pos(rg[sel],30),30)
                assert abs(ref-w['reference_hr'])<1e-9,('reference replay mismatch',stem,w['index'])
                assert abs(pred-w['predictions']['P0R0'])<1e-9,('POS replay mismatch',stem,w['index'])
                r=dict(video=row['video'],subject=row['patient_id'],step=row['step'],index=w['index'],start_s=w['start_s'],reference_hr=ref,POS=pred,failures={})
                jobs={'CHROM':lambda:estimate_hr(chrom(rg[sel],30),30)}
                for i,(rr,vv) in enumerate(regions):
                    jobs['POS_ROI'+str(i)]=lambda rr=rr,vv=vv:estimate_hr(pos(rr[sel],30),30) if vv[sel].all() else (_ for _ in ()).throw(ValueError('ROI support'))
                for name,f in jobs.items():
                    try:r[name]=f()
                    except ValueError as e:r[name]=None;r['failures'][name]=str(e)
                if rawok[sel].all():r['raw_reference_hr']=estimate_hr(raw[sel],30)
                else:r['raw_reference_hr']=None;r['failures']['raw_reference_hr']='Raw timestamp support/gap/nonmonotonic'
                result.append(r)
            save(out/(stem+'.json'),{'rows':result,'trace_sha256':record['trace_sha256']})
            allrows.extend(result)
        except Exception:
            failures.append({'video':row['video'],'expected_windows':len(selected),'error':traceback.format_exc()})
            save(out/(stem+'.error.json'),failures[-1])
        save(out/'status.json',{'completed_records':num,'expected_records':119,'windows':len(allrows),'errors':len(failures),'complete':num==119})
        print(num,stem,len(allrows),'errors',len(failures),flush=True)
    save(out/'results.json',{'rows':allrows,'failures':failures,'expected_windows':1860})
    if failures or len(allrows)!=1860:raise RuntimeError('Incomplete primary replay; see failures')

if __name__=='__main__':
    if '--test' in sys.argv:tests();sys.exit(0)
    p=argparse.ArgumentParser()
    for name in ['root','e0','output','protocol']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
