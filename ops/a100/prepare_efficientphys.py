"""Train/validation-only error audit and 36px RGB cache. Never opens test videos."""
import os, csv, json, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from scipy import signal
from hr_baseline import estimate_hr, detrend
R=Path('/data/khangle/facevitals')
OLD=R/'results/ubfc_pos_green_v1'
OUT=R/'datasets/UBFC-rPPG/efficientphys36_trainval_v1'
SPLIT=json.loads((OLD/'subject_split.json').read_text())

def peaks(x,fs):
    s=signal.sosfiltfilt(signal.butter(1,[.75,3],btype='bandpass',fs=fs,output='sos'),detrend(x))
    f,p=signal.periodogram(s,fs=fs,window='hann',nfft=4096)
    inds=signal.find_peaks(p)[0]
    inds=[i for i in inds if .75<=f[i]<=3]
    inds=sorted(inds,key=lambda i:p[i],reverse=True)[:3]
    total=p[(f>=.75)&(f<=3)].sum()
    return [{'bpm':float(f[i]*60),'relative_bin_power':float(p[i]/max(total,1e-20))} for i in inds]

def run(subject, allowed_splits=('train','val')):
    tag=next(k for k in ('train','val','test') if subject in SPLIT[k])
    assert tag in allowed_splits, 'Subject outside requested split'
    raw=R/'datasets/UBFC-rPPG/DATASET_2'/subject
    gt=np.loadtxt(raw/'ground_truth.txt'); times=gt[2]-gt[2,0]
    old=np.load(OLD/(subject+'_signals.npz'))
    windows=[w for w in csv.DictReader((OLD/'windows.csv').open()) if w['subject']==subject]
    fs=(len(times)-1)/times[-1]
    audit=[]; clips=[]
    for i,w in enumerate(windows):
        start=float(w['start_s']); a=round(start*fs); b=a+round(10*fs)
        err=abs(float(w['pos_bpm'])-float(w['gt_ppg_bpm']))
        interval=(times>=start)&(times<start+10)
        gaps=np.diff(times); gap_idx=np.where((times[:-1]<start+10)&(times[1:]>start))[0]
        info={'subject':subject,'split':tag,'start_s':start,'abs_pos_error_bpm':err,
              'pos_bpm':float(w['pos_bpm']),'gt_bpm':float(w['gt_ppg_bpm']),
              'max_gap_s':float(gaps[gap_idx].max()) if len(gap_idx) else 0.,
              'sensor_below40_count':int((gt[1,interval]<40).sum()),
              'approx_half_frequency':abs(float(w['pos_bpm'])/float(w['gt_ppg_bpm'])-.5)<.08}
        if err>5:
            info['ppg_peaks']=peaks(old['ppg'][a:b],fs)
            info['pos_peaks']=peaks(old['pos'][a:b],fs)
        audit.append(info)
        q=start+np.arange(301)/30
        if q[-1]>times[-1]: raise ValueError('Clip beyond available input')
        # Matched 300-sample / 30Hz HR references for this learned baseline.
        ppg=np.interp(q,old['t'],old['ppg'])
        pos_hr=estimate_hr(np.interp(q[:-1],old['t'],old['pos']),30.)
        clips.append({'subject':subject,'split':tag,'window':i,'start_s':start,
                      'gt_hr':estimate_hr(ppg[:-1],30.),'pos_hr':pos_hr,
                      'max_gap_s':info['max_gap_s']})
    # Same causal-by-frame ROI selection as the existing CPU baseline.
    cv2.setNumThreads(1)
    cap=cv2.VideoCapture(str(raw/'vid.avi'))
    nominal=cap.get(cv2.CAP_PROP_FPS)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    crops=[]; boxes=[]; last=None; misses=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        idx=len(crops); h,w=frame.shape[:2]
        if idx%max(1,round(nominal/2))==0 or last is None:
            faces=detector.detectMultiScale(cv2.cvtColor(cv2.resize(frame,(w//2,h//2)),cv2.COLOR_BGR2GRAY),1.1,5,minSize=(30,30))
            if len(faces):
                if last is None: face=max(faces,key=lambda b:int(b[2])*int(b[3]))
                else:
                    lx,ly,lw,lh=last
                    face=min(faces,key=lambda b:(2*b[0]+b[2]-lx-lw/2)**2+(2*b[1]+b[3]-ly-lh/2)**2)
                last=tuple(int(v)*2 for v in face); misses=0
            else: misses+=1
        if last is None or misses>4: raise ValueError('ROI unavailable; explicit repair required')
        x,y,bw,bh=last
        x1=max(0,int(x+.15*bw)); x2=min(w,int(x+.85*bw))
        y1=max(0,int(y+.1*bh)); y2=min(h,int(y+.85*bh))
        crops.append(cv2.cvtColor(cv2.resize(frame[y1:y2,x1:x2],(36,36),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
        boxes.append([x1,y1,x2,y2])
    cap.release()
    assert len(crops)==gt.shape[1]
    crops=np.asarray(crops); boxes=np.asarray(boxes)
    # Duplicate timestamps merged at pixel/label level before interpolation.
    unique,inverse,counts=np.unique(times,return_inverse=True,return_counts=True)
    merged=np.zeros((len(unique),36,36,3),np.float32)
    np.add.at(merged,inverse,crops)
    merged/=counts[:,None,None,None]
    ppg_sum=np.zeros(len(unique)); np.add.at(ppg_sum,inverse,gt[0])
    np.save(OUT/(subject+'_rgb.npy'),merged.astype(np.float16))
    np.savez(OUT/(subject+'_meta.npz'),t=unique,ppg=ppg_sum/counts,boxes=boxes,original_t=times)
    centers=(boxes[:,:2]+boxes[:,2:])/2
    jump=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    for info in audit:
        mask=(times[1:]>=info['start_s'])&(times[1:]<info['start_s']+10)
        info['max_roi_center_step_px']=float(jump[mask].max()) if mask.any() else 0.
    return {'subject':subject,'split':tag,'frames':len(crops),'clips':clips,'windows':audit,
            'sensor_min':float(gt[1].min()),
            'sensor_jumps_over100':int((np.abs(np.diff(gt[1]))>100).sum())}

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'subject_split.json').write_text(json.dumps(SPLIT,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=2) as p:
        jobs=[p.submit(run,s) for s in SPLIT['train']+SPLIT['val']]
        for f in as_completed(jobs):
            r=f.result(); results.append(r)
            print(json.dumps({'done':len(results),'subject':r['subject'],'clips':len(r['clips'])}),flush=True)
    results.sort(key=lambda r:int(r['subject'][7:]))
    (OUT/'error_audit_trainval.json').write_text(json.dumps(results,indent=2))
    (OUT/'clips.json').write_text(json.dumps([c for r in results for c in r['clips']],indent=2))
    inventory={}
    for f in sorted(OUT.glob('*')):
        if f.is_file(): inventory[f.name]={'bytes':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2))
    print(json.dumps({'complete':True,'subjects':len(results),'clips':sum(len(r['clips']) for r in results)}),flush=True)
