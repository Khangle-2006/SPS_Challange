import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy import signal
from t7_pipeline import save
from hr_baseline import pos,estimate_hr
from mcd_e0_metrics import sample_grid
from mcd_errata_audit import timed

def author_hr(x):
    f,p=signal.periodogram(x,fs=30,nfft=2**int(np.ceil(np.log2(len(x)))),detrend=False)
    k=(f>=.5)&(f<=3.5)
    return float(f[k][np.argmax(p[k])]*60)

def author_pos(rgb):
    # Exact projection/OLA endpoint convention in published models/POS.py.
    h=np.zeros(len(rgb))
    for end in range(48,len(rgb)):
        x=rgb[end-48:end];s=np.array([[0,1,-1],[-2,1,1]])@(x/x.mean(0)).T
        if np.std(s[1])<1e-12:raise ValueError('Degenerate author POS')
        p=s[0]+np.std(s[0])/np.std(s[1])*s[1];h[end-48:end]+=p-p.mean()
    b,a=signal.butter(1,[.75,3],fs=30,btype='bandpass')
    return signal.filtfilt(b,a,h)

def main(a):
    original=json.loads((a.output/'results.json').read_text())
    assert len(original['rows'])==1860 and not original['failures']
    cohort=json.loads((a.e0/'cohort.json').read_text());out=[];errors=[]
    for record in cohort:
        old={r['index']:r for r in original['rows'] if r['video']==record['video']}
        pair_starts=[i for i in old if i%2==0 and i+1 in old]
        if not pair_starts:continue
        try:
            rgb=np.load(a.e0/Path(record['video']).stem/'trace.npz')['rgb'][:,0].mean(1)
            _,mt=timed(a.root/record['meta']);mt=mt[:len(rgb)];t=(mt-mt[0])/1e6
            sy=np.loadtxt(a.root/record['ppg_sync'],ndmin=2)[:len(rgb),0]
            grid=np.arange(int(np.floor(t[-1]*30))+1)/30
            rg,rv=sample_grid(t,rgb,np.isfinite(rgb).all(1),grid)
            pg,pv=sample_grid(t,sy,np.isfinite(sy),grid)
            for i in pair_starts:
                sel=(grid>=i*10)&(grid<(i+2)*10)
                assert sel.sum()==600 and rv[sel].all() and pv[sel].all()
                x=rg[sel];y=pg[sel]
                ref=estimate_hr(y,30);pred=estimate_hr(pos(x,30),30)
                # Published processing.py cascades fourth-order low then high pass.
                b,c=signal.butter(4,10,fs=30,btype='lowpass')
                ay=signal.filtfilt(b,c,y)
                b,c=signal.butter(4,.5,fs=30,btype='highpass')
                ay=signal.filtfilt(b,c,ay)
                ar=author_hr(ay);ap=author_hr(author_pos(x))
                out.append(dict(subject=record['patient_id'],video=record['video'],step=record['step'],start_s=i*10,
                    paired10_error=float(np.mean([abs(old[j]['POS']-old[j]['reference_hr']) for j in [i,i+1]])),
                    same_decoder20_error=abs(pred-ref),author_style20_error=abs(ap-ar),
                    same_decoder20_reference=ref,same_decoder20_prediction=pred,author_style20_reference=ar,author_style20_prediction=ap))
        except Exception as e:errors.append(dict(video=record['video'],error=repr(e),expected_pairs=len(pair_starts)))
    summary={}
    for g in ['all','before','after']:
        rr=[r for r in out if g=='all' or r['step']==g];summary[g]={}
        for k in ['paired10_error','same_decoder20_error','author_style20_error']:
            by=[[r[k] for r in rr if r['subject']==s] for s in sorted({r['subject'] for r in rr})]
            summary[g][k]=dict(pairs=len(rr),subjects=len(by),pair_MAE=float(np.mean([r[k] for r in rr])),subject_macro_MAE=float(np.mean([np.mean(v) for v in by])))
    save(a.output/'window_diagnostic.json',dict(rows=out,errors=errors,summary=summary,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest()))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','e0','output','protocol']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
