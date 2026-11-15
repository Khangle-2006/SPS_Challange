"""Bounded native-region comparison, existing landmarks, no fitted parameters."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,json,hashlib,time,traceback,fcntl
from pathlib import Path
import numpy as np
import cv2,av
from t7_pipeline import save
from mcd_errata_audit import timed
from mcd_e0_metrics import sample_grid
from hr_baseline import pos,estimate_hr

POLYS=[[109,10,338,337,151,108],[50,101,205,187,123],[280,330,425,411,352]]
OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
def mean_mask(rgb,points,hull=False):
    if not np.isfinite(points).all():return np.full(3,np.nan)
    pp=np.rint(points).astype(np.int32)
    if hull:pp=cv2.convexHull(pp).reshape(-1,2)
    h,w=rgb.shape[:2];x0,y0=np.maximum(pp.min(0),0);x1,y1=np.minimum(pp.max(0)+1,[w,h])
    if x1<=x0 or y1<=y0:return np.full(3,np.nan)
    mask=np.zeros((y1-y0,x1-x0),np.uint8);cv2.fillPoly(mask,[pp-[x0,y0]],1)
    if mask.sum()<16:return np.full(3,np.nan)
    return np.asarray(cv2.mean(rgb[y0:y1,x0:x1],mask=mask)[:3])

def main(a):
    cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    fingerprint={k:sha(p) for k,p in [('script',Path(__file__)),('protocol',a.protocol),('primary',a.t7/'results.json')]}
    if (a.output/'fingerprint.json').exists():assert json.loads((a.output/'fingerprint.json').read_text())==fingerprint
    else:save(a.output/'fingerprint.json',fingerprint)
    cohort=json.loads((a.e0/'cohort.json').read_text());primary=json.loads((a.t7/'results.json').read_text())['rows']
    assert len(cohort)==119 and len(primary)==1860
    result=[];errors=[];status=dict(pid=os.getpid(),state='running',completed=0,total=119,expected_windows=1860)
    save(a.output/'status.json',status)
    for num,c in enumerate(cohort[:a.limit] if a.limit else cohort,1):
        stem=Path(c['video']).stem;dest=a.output/(stem+'.json');status['current']=stem;save(a.output/'status.json',status)
        selected=[r for r in primary if r['video']==c['video']]
        try:
            if dest.exists():rr=json.loads(dest.read_text());result.extend(rr['rows']);continue
            gate=json.loads((a.audit/'dev60_gate_v3'/(stem+'.json')).read_text())
            tm=json.loads((a.t3/stem/'manifest.json').read_text())
            assert tm['source_video_sha256']==gate['source_sha256']
            tracefile=a.t3/stem/'trace.npz';zz=np.load(tracefile);lm=zz['landmarks'].astype(np.float32)
            n=gate['gate']['candidate_prefix_frames'];assert len(lm)==n==tm['n_frames']
            _,mt=timed(a.root/c['meta']);t=(mt[:n]-mt[0])/1e6
            assert np.allclose(zz['timestamps_s'],t,rtol=0,atol=1e-6)
            if sum(p.stat().st_size for p in a.output.glob('*') if p.is_file())>950000000:raise RuntimeError('1GB output budget')
            rgb=np.full((n,2,3),np.nan,np.float32);seen=0
            with av.open(str(a.root/c['video'])) as container:
                container.streams.video[0].codec_context.thread_count=1
                for frame in container.decode(video=0):
                    if seen>=n:break
                    assert not frame.is_corrupt
                    assert str(frame.pts*frame.time_base)==gate['container_pts_rational'][seen]
                    im=frame.to_ndarray(format='rgb24');points=lm[seen]
                    if np.isfinite(points).all():
                        rgb[seen,0]=np.mean([mean_mask(im,points[p]) for p in POLYS],axis=0)
                        rgb[seen,1]=mean_mask(im,points[OVAL],True)
                    seen+=1
                    if seen%500==0:status['frame']=seen;save(a.output/'status.json',status)
                    if seen==n:break
            assert seen==n
            np.savez_compressed(a.output/(stem+'.npz'),rgb=rgb,t=t)
            grid=np.arange(int(np.floor(t[-1]*30))+1)/30
            sampled=[sample_grid(t,rgb[:,j],np.isfinite(rgb[:,j]).all(1),grid) for j in range(2)]
            rows=[]
            for old in selected:
                r={k:old[k] for k in ['video','subject','step','index','start_s','reference_hr','POS']};r['failures']={}
                sel=(grid>=r['start_s'])&(grid<r['start_s']+10)
                for key,(x,v) in zip(['control_small','large_oval'],sampled):
                    try:
                        if sel.sum()!=300 or not v[sel].all():raise ValueError('ROI/time support missing')
                        r[key]=estimate_hr(pos(x[sel],30),30)
                    except ValueError as e:r[key]=None;r['failures'][key]=str(e)
                rows.append(r)
            save(dest,{'rows':rows,'source_sha256':gate['source_sha256'],'t3_trace_sha256':sha(tracefile)})
            result.extend(rows)
        except Exception:
            error={'video':c['video'],'expected_windows':len(selected),'error':traceback.format_exc()};errors.append(error);save(a.output/(stem+'.error.json'),error)
        finally:
            status.update(completed=num,errors=len(errors),windows=len(result));save(a.output/'results.json',{'rows':result,'errors':errors,'expected_windows':1860});save(a.output/'status.json',status)
            print(num,stem,len(result),'errors',len(errors),flush=True)
    status['state']=('smoke_complete' if a.limit else 'complete') if not errors else 'complete_with_errors';save(a.output/'status.json',status)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','e0','t7','t3','audit','output','protocol']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
