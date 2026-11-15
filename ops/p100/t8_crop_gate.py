"""One POS check of exact saved T8 native cropboxes; no frontend rerun/grid."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,hashlib,time,fcntl,traceback
from pathlib import Path
import numpy as np
import cv2
from data_adapters import MCD
from mcd_e0_metrics import sample_grid
from hr_baseline import pos,estimate_hr


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(4<<20),b''):h.update(x)
    return h.hexdigest()
def save(p,v):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,indent=2,allow_nan=False));temp.replace(p)


def main(a):
    cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[min(2,len(allowed)-1)]})
    code=Path(__file__).parent;frozen=read(code/'SHA256.json')
    assert all(sha(code/f)==h for f,h in frozen.items())
    primary=read(a.t7/'results.json')['rows'];assert len(primary)==1860
    cohort=read(code/'mcd_cohort.json');assert len(cohort)==119
    fingerprint=dict(code=frozen,primary_sha256=sha(a.t7/'results.json'),cache_fingerprint=read(a.cache/'fingerprint.json'))
    if (a.output/'fingerprint.json').exists():assert read(a.output/'fingerprint.json')==fingerprint
    else:save(a.output/'fingerprint.json',fingerprint)
    backend=MCD(a.root,a.audit);status=dict(state='running',pid=os.getpid(),completed=0,total=119,expected_windows=1860,errors=[])
    allrows=[];save(a.output/'status.json',status)
    for row in cohort:
        rid=Path(row['video']).stem;dest=a.output/(rid+'.json');status.update(recording=rid,frame=0)
        save(a.output/'status.json',status)
        try:
            if dest.exists():
                receipt=read(dest);assert receipt['fingerprint']==fingerprint
                allrows.extend(receipt['rows']);status['completed']+=1;continue
            crop=a.cache/rid/'native'/'crop';cache_receipt=read(crop.parent/'complete.json')
            assert cache_receipt['fingerprint']==fingerprint['cache_fingerprint']
            assert sha(crop/'meta.npz')==cache_receipt['cache_files']['meta.npz']
            record=backend.load(rid);meta=np.load(crop/'meta.npz')
            assert cache_receipt['source_sha256']==record.metadata['source_video_sha256']
            np.testing.assert_allclose(record.timestamps,meta['original_t'],rtol=0,atol=1e-9)
            boxes=meta['boxes'];n=len(record.timestamps);assert len(boxes)==n
            t=record.timestamps;rgb=np.full((n,3),np.nan,np.float64);areas=np.zeros(n,np.int32)
            source_valid=np.asarray(record.metadata['frame_valid'],bool)
            retained_valid=meta['valid'][np.searchsorted(meta['t'],t)]
            seen=0
            for i,image in enumerate(record.frames):
                x1,y1,x2,y2=map(int,boxes[i]);h,w=image.shape[:2]
                if source_valid[i] and retained_valid[i] and x2>x1 and y2>y1:
                    assert 0<=x1<x2<=w and 0<=y1<y2<=h
                    rgb[i]=cv2.mean(image[y1:y2,x1:x2])[:3];areas[i]=(x2-x1)*(y2-y1)
                seen+=1
                if seen%500==0:
                    status.update(frame=seen,updated_unix=time.time());save(a.output/'status.json',status)
            assert seen==n
            grid=np.arange(int(np.floor(t[-1]*30))+1)/30
            sampled,valid=sample_grid(t,rgb,np.isfinite(rgb).all(1),grid)
            rows=[]
            for old in [r for r in primary if r['video']==row['video']]:
                r={k:old[k] for k in ('video','subject','step','index','start_s','reference_hr','large_oval','control_small')}
                r.update(crop_POS=None,failure=None)
                select=(grid>=r['start_s'])&(grid<r['start_s']+10)
                try:
                    if select.sum()!=300 or not valid[select].all():raise ValueError('Missing crop/time support')
                    r['crop_POS']=estimate_hr(pos(sampled[select],30),30,band=(.7,3.))
                except ValueError as e:r['failure']=str(e)
                rows.append(r)
            np.savez_compressed(a.output/(rid+'.npz'),rgb=rgb,t=t,area=areas)
            receipt=dict(fingerprint=fingerprint,rows=rows,recording_id=rid,frames=n,
                meta_sha256=sha(crop/'meta.npz'),bbox_area_median=float(np.median(areas[areas>0])) if (areas>0).any() else None)
            save(dest,receipt);allrows.extend(rows);status['completed']+=1
            if sum(p.stat().st_size for p in a.output.glob('*') if p.is_file())>240_000_000:raise RuntimeError('250MB budget')
        except Exception:
            status['state']='blocked_error';status['errors'].append(dict(recording=rid,traceback=traceback.format_exc()))
            save(a.output/'status.json',status);save(a.output/'results.json',dict(rows=allrows,errors=status['errors'],expected_windows=1860));raise
        finally:
            save(a.output/'status.json',status)
        print(json.dumps(dict(recording=rid,completed=status['completed'])),flush=True)
    assert len(allrows)==1860
    save(a.output/'results.json',dict(rows=allrows,errors=[],expected_windows=1860))
    status['state']='complete';save(a.output/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','audit','cache','t7','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
