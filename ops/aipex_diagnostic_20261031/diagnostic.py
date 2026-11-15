"""Fixed limited AiPEX diagnostic. No neural model or outcome-driven choices."""
import argparse, csv, hashlib, json, math, os
from pathlib import Path
import cv2
import numpy as np
from hr_baseline import pos, estimate_hr

REGIONS = ('forehead', 'leftcheek', 'rightcheek')
BINS = ('<=-30', '(-30,10]', '(10,28]', '(28,41]', '(41,55]', '>55', 'unknown')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def angle(light, yellow):
    if not math.isfinite(light+yellow) or yellow<=0: return None
    return math.degrees(math.atan((light-50)/yellow))

def bin_ita(value):
    if value is None: return 'unknown'
    for bound, label in zip((-30,10,28,41,55), BINS):
        if value<=bound: return label
    return '>55'

def cheek_samples(video, region):
    cap=cv2.VideoCapture(str(video)); samples=[]
    try:
        if not cap.isOpened(): raise ValueError('Cannot open cheek')
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for q in (.25,.5,.75):
            index=math.floor((n-1)*q)
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(index+1))>.01:
                raise ValueError('Cheek frame seek mismatch')
            h,w=frame.shape[:2]; crop=frame[int(.1*h):int(.9*h),int(.1*w):int(.9*w)]
            lab=cv2.cvtColor(crop.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            light=float(np.median(lab[:,:,0])); yellow=float(np.median(lab[:,:,2]))
            samples.append(dict(roi=region,index=index,frames=n,L=light,b=yellow,
                ita=angle(light,yellow),clipped_channel_fraction=float(np.mean((crop==0)|(crop==255)))))
    finally: cap.release()
    return samples

def process(root,cache,group,subject,truth):
    row=dict(country=group,subject=subject,reference_bpm=truth,prediction_bpm=None,
             ita=None,ita_bin='unknown',hr_error=None,ita_error=None,rois=[],ita_samples=[])
    try:
        for region in REGIONS:
            video=root/'Data'/group/(subject+'_'+region+'.avi')
            marker=video.with_name(video.name+'.complete.json')
            receipt=json.loads(marker.read_text())
            cp=cache/(group+'_'+subject+'_'+region+'_rgb.npz')
            with np.load(cp,allow_pickle=False) as c:
                if str(c['sha256'])!=receipt['sha256']: raise ValueError('RGB source binding mismatch')
                fs=float(c['fs']); rgb=c['rgb']
            if fs!=15: raise ValueError('Unexpected cache FPS; do not fit timing')
            bvp=pos(rgb,fs)
            preds=[estimate_hr(bvp[s:s+150],fs,band=(.7,3.0)) for s in range(24,len(rgb)-24-150+1,150)]
            if not preds: raise ValueError('No full HR windows')
            row['rois'].append(dict(roi=region,frames=len(rgb),fs=fs,pos_samples=24,
                decoder_fs=fs,hr_samples=150,windows=len(preds),prediction_bpm=float(np.mean(preds)),
                window_predictions=preds,cache_sha256=sha(cp),source_sha256=receipt['sha256']))
        row['prediction_bpm']=float(np.median([r['prediction_bpm'] for r in row['rois']]))
    except Exception as e: row['hr_error']=str(e)
    try:
        for region in REGIONS[1:]:
            video=root/'Data'/group/(subject+'_'+region+'.avi')
            row['ita_samples'].extend(cheek_samples(video,region))
        values=[r['ita'] for r in row['ita_samples']]
        if len(values)!=6 or any(v is None for v in values): raise ValueError('Incomplete or nonpositive-b ITA')
        row['ita']=float(np.median(values)); row['ita_bin']=bin_ita(row['ita'])
    except Exception as e: row['ita_error']=str(e)
    return row

def run(a):
    import fcntl
    a.output.mkdir(parents=True,exist_ok=False)
    lock=(a.output/'worker.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cv2.setNumThreads(1)
    bundle=Path(__file__).parent
    manifest=json.loads((bundle/'SHA256.json').read_text())
    for name,digest in manifest.items():
        if sha(bundle/name)!=digest: raise ValueError('Bundle hash mismatch: '+name)
    tasks=[]
    for group in ('India','Sierra_Leone'):
        with (a.root/'Ground_Truth'/(group+'_GT.csv')).open(encoding='utf-8-sig') as f:
            for r in list(csv.reader(f))[1:]:
                if r and r[0].strip(): tasks.append((group,r[0].strip(),float(r[1])))
    if len(tasks)!=88 or len(set((g,s) for g,s,t in tasks))!=88: raise ValueError('Unexpected cohort')
    save(a.output/'provenance.json',dict(bundle=manifest,assumed_fs=15,
        labels={g:sha(a.root/'Ground_Truth'/(g+'_GT.csv')) for g in ('India','Sierra_Leone')}))
    rows=[]
    for group,subject,truth in tasks:
        if sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())>50_000_000:
            raise ValueError('50 MB output cap exceeded')
        row=process(a.root,a.cache,group,subject,truth); rows.append(row)
        save(a.output/(group+'_'+subject+'.json'),row)
        status=dict(phase='running',pid=os.getpid(),done=len(rows),total=88,
            hr_failures=sum(r['hr_error'] is not None for r in rows),ita_unknown=sum(r['ita'] is None for r in rows))
        save(a.output/'status.json',status); print(json.dumps(status),flush=True)
    save(a.output/'records.json',rows)
    status['phase']='complete';save(a.output/'status.json',status)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
