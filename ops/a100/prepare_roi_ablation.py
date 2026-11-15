"""Causal box smoothing pilot, then matched caches. Never accesses test subjects."""
import argparse, hashlib, json, tarfile
from pathlib import Path
import cv2
import numpy as np
R=Path('/data/khangle/facevitals'); D=R/'datasets/UBFC-rPPG'
BASE=D/'efficientphys36_trainval_v1'; OUT=R/'results/roi_preparation_v1'
TAU=.10
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def smooth(b,t):
    # Only present/past detections, recorded elapsed time; never labels.
    b=b.astype(float); center=(b[:,:2]+b[:,2:])/2; size=b[:,2:]-b[:,:2]
    z=np.concatenate([center,np.log(size)],axis=1); y=z.copy()
    for i in range(1,len(z)):
        alpha=-np.expm1(-max(0.,float(t[i]-t[i-1]))/TAU)
        y[i]=y[i-1]+alpha*(z[i]-y[i-1])
        wh=np.exp(y[i,2:]);candidate=np.rint(np.r_[y[i,:2]-wh/2,y[i,:2]+wh/2])
        inter=np.prod(np.maximum(0,np.minimum(candidate[2:],b[i,2:])-np.maximum(candidate[:2],b[i,:2])))
        union=np.prod(candidate[2:]-candidate[:2])+np.prod(size[i])-inter
        if inter/union<.50:y[i]=z[i]  # Reacquire rather than keep a stale box.
    wh=np.exp(y[:,2:]);return np.rint(np.concatenate([y[:,:2]-wh/2,y[:,:2]+wh/2],axis=1)).astype(np.int32)
def stats(old,new):
    wh=old[:,2:]-old[:,:2]; lag=np.linalg.norm((new[:,:2]+new[:,2:]-old[:,:2]-old[:,2:])/2,axis=1)/np.min(wh,axis=1)
    intersection=np.prod(np.maximum(0,np.minimum(old[:,2:],new[:,2:])-np.maximum(old[:,:2],new[:,:2])),axis=1)
    area=np.prod(wh,axis=1); area2=np.prod(new[:,2:]-new[:,:2],axis=1)
    iou=intersection/(area+area2-intersection)
    return {'p99_center_lag_fraction':float(np.percentile(lag,99)), 'min_iou_to_detector':float(iou.min()),'p01_iou_to_detector':float(np.percentile(iou,1)),
            'old_max_coordinate_step':int(np.abs(np.diff(old,axis=0)).max()),'new_max_coordinate_step':int(np.abs(np.diff(new,axis=0)).max())}
def pilot(split):
    OUT.mkdir(parents=True,exist_ok=True); report={}; worst=[]
    for s in split['train']:
        m=np.load(BASE/(s+'_meta.npz')); b=m['boxes']; n=smooth(b,m['original_t']);report[s]=stats(b,n)
        lag=np.linalg.norm((n[:,:2]+n[:,2:]-b[:,:2]-b[:,2:])/2,axis=1)
        worst.append((float(lag.max()),s,int(lag.argmax()),b,n))
    # Conservative geometric sanity gates, not a ground-truth face coverage claim.
    passed=all(v['p99_center_lag_fraction']<=.15 and v['min_iou_to_detector']>=.50 for v in report.values())
    result={'tau_s':TAU,'reset_if_iou_below':.50,'train_only':True,'subjects':report,'geometry_gate_passed':passed,
            'gate':'each train subject: p99 center lag <= .15 of smaller old-box side; minimum IoU to old box >= .50. This measures detector agreement, not anatomical accuracy.',
            'code_sha256':sha(Path(__file__))}
    (OUT/'pilot.json').write_text(json.dumps(result,indent=2))
    panels=[]
    for _,s,i,b,n in sorted(worst,key=lambda x:x[0],reverse=True)[:8]:
        cap=cv2.VideoCapture(str(D/'DATASET_2'/s/'vid.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,f=cap.read();cap.release();assert ok
        cv2.rectangle(f,tuple(b[i,:2]),tuple(b[i,2:]),(0,255,0),2)
        cv2.rectangle(f,tuple(n[i,:2]),tuple(n[i,2:]),(0,0,255),2)
        cv2.putText(f,f'{s} frame {i}: green old/red smooth',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        panels.append(cv2.resize(f,(640,480)))
    cv2.imwrite(str(OUT/'train_worst_lag.jpg'),np.concatenate([np.concatenate(panels[i:i+2],axis=1) for i in range(0,8,2)],axis=0))
    print(json.dumps(result),flush=True)
def cache(split):
    pilot_result=json.loads((OUT/'pilot.json').read_text());assert pilot_result['geometry_gate_passed']
    assert pilot_result['code_sha256']==sha(Path(__file__))
    configs=[('clean',split['train']+split['val']),('train500',split['train']),('val500',split['val']),('val250',split['val']),('val1000',split['val'])]
    paths=[]
    for condition,subjects in configs:
        dest=D/('roi_ema010_'+condition+'_v1');dest.mkdir(parents=True,exist_ok=True);paths.append(dest)
        rows=[r for r in json.loads((BASE/'clips.json').read_text()) if r['subject'] in subjects]
        (dest/'clips.json').write_text(json.dumps(rows,indent=2));(dest/'subject_split.json').write_text(json.dumps(split,indent=2))
        for s in subjects:
            assert s not in split['test']
            meta=np.load(BASE/(s+'_meta.npz')); times=meta['original_t'];boxes=smooth(meta['boxes'],times)
            if condition=='clean':path=D/'DATASET_2'/s/'vid.avi'
            else:
                folder='training' if condition.startswith('train') else 'validation'
                rate=condition.removeprefix('train').removeprefix('val')
                path=D/f'{folder}_h264_{rate}k_v1'/(s+'.mp4')
            cap=cv2.VideoCapture(str(path));assert cap.isOpened(),str(path);crops=[]
            while True:
                ok,f=cap.read()
                if not ok:break
                x1,y1,x2,y2=boxes[len(crops)];h,w=f.shape[:2]
                assert 0<=x1<x2<=w and 0<=y1<y2<=h
                crops.append(cv2.cvtColor(cv2.resize(f[y1:y2,x1:x2],(36,36),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
            cap.release();assert len(crops)==len(times)
            unique,inverse,counts=np.unique(times,return_inverse=True,return_counts=True)
            merged=np.zeros((len(unique),36,36,3),np.float32);np.add.at(merged,inverse,np.asarray(crops));merged/=counts[:,None,None,None]
            np.save(dest/(s+'_rgb.npy'),merged.astype(np.float16))
            values={k:meta[k] for k in meta.files};values['boxes']=boxes
            np.savez(dest/(s+'_meta.npz'),**values)
            print('CACHE',condition,s,flush=True)
        inventory={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in dest.iterdir() if p.name!='inventory.json'}
        (dest/'inventory.json').write_text(json.dumps(inventory,indent=2))
    archive=D/'roi_ema010_caches_v1.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in paths:tar.add(p,arcname=str(p.relative_to(R)))
    (OUT/'cache_complete.json').write_text(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'test_opened':False},indent=2))
    print('COMPLETE',sha(archive),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','cache'],required=True);a=p.parse_args()
    cv2.setNumThreads(1);split=json.loads((BASE/'subject_split.json').read_text())
    (pilot if a.stage=='pilot' else cache)(split)
