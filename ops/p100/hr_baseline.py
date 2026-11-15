"""CPU HR baseline: POS (Wang 2017) and green channel; no label-driven tuning."""
import argparse, csv, hashlib, json, math, re
from pathlib import Path
import cv2
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

def pos(rgb, fs):
    rgb=np.asarray(rgb,dtype=np.float64)
    if rgb.ndim!=2 or rgb.shape[1]!=3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite T x 3 RGB")
    length=math.ceil(1.6*fs)
    if len(rgb)<length: raise ValueError("Clip too short for POS")
    out=np.zeros(len(rgb))
    for end in range(length,len(rgb)+1):
        x=rgb[end-length:end]
        normalized=x/np.maximum(x.mean(axis=0),1e-8)
        s=np.array([[0,1,-1],[-2,1,1]])@normalized.T
        h=s[0]+np.std(s[0])/max(np.std(s[1]),1e-8)*s[1]
        out[end-length:end]+=h-h.mean()
    return out

def detrend(x, lam=100):
    x=np.asarray(x,dtype=float)
    n=len(x)
    d=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format="csc")
    return x-spsolve(sparse.eye(n,format="csc")+lam**2*(d.T@d),x)

def estimate_hr(x, fs, band=(0.7,3.0)):
    if not np.isfinite(x).all() or len(x)<fs*5 or np.std(x)<1e-10: raise ValueError("Invalid/flat/short signal")
    low,high=band
    if not 0<low<high<fs/2: raise ValueError("Band must be inside Nyquist")
    filt=signal.sosfiltfilt(signal.butter(1,[low,high],btype="bandpass",fs=fs,output="sos"),detrend(x))
    freq,power=signal.periodogram(filt,fs=fs,window="hann",nfft=max(4096,2**int(np.ceil(np.log2(len(x))))),detrend=False)
    selected=(freq>=low)&(freq<=high)
    return float(freq[selected][np.argmax(power[selected])]*60)

def read_rgb(path, fullface=False):
    cv2.setNumThreads(1)
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError("Video cannot be opened")
    fs=cap.get(cv2.CAP_PROP_FPS); declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not 1<fs<300: raise ValueError("Invalid FPS")
    if fullface:
        detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        if detector.empty(): raise ValueError("Missing face detector")
    values=[]; boxes=[]; last=None; detection_count=0; attempts=0; misses=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        idx=len(values); h,w=frame.shape[:2]
        if fullface:
            if idx%max(1,round(fs/2))==0 or last is None:
                small=cv2.resize(frame,(w//2,h//2))
                faces=detector.detectMultiScale(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(30,30))
                attempts+=1
                if len(faces):
                    if last is None:
                        face=max(faces,key=lambda b:int(b[2])*int(b[3]))
                    else:
                        lx,ly,lw,lh=last
                        face=min(faces,key=lambda b:(2*b[0]+b[2]-lx-lw/2)**2+(2*b[1]+b[3]-ly-lh/2)**2)
                    last=tuple(int(v)*2 for v in face); detection_count+=1; misses=0
                else: misses+=1
            if last is None or misses>4:
                values.append([np.nan]*3); boxes.append([0]*4); continue
            x,y,bw,bh=last
            x1=max(0,int(x+.15*bw)); x2=min(w,int(x+.85*bw))
            y1=max(0,int(y+.10*bh)); y2=min(h,int(y+.85*bh))
            crop=frame[y1:y2,x1:x2]; boxes.append([x1,y1,x2,y2])
        else: crop=frame
        if crop.size==0: raise ValueError("Empty ROI")
        values.append(cv2.mean(crop)[:3][::-1])
    cap.release()
    rgb=np.asarray(values)
    if len(rgb)!=declared: raise ValueError(f"Decode incomplete: {len(rgb)}/{declared}")
    valid=np.isfinite(rgb).all(axis=1)
    if valid.mean()<.95: raise ValueError(f"Face coverage only {valid.mean():.3f}")
    if not valid.all():
        # Interpolate only rare failed detections, report exact coverage.
        for c in range(3): rgb[:,c]=np.interp(np.arange(len(rgb)),np.where(valid)[0],rgb[valid,c])
    return rgb,float(fs),{"frames":len(rgb),"declared_frames":declared,"face_coverage":float(valid.mean()),"detector_successes":detection_count,"detector_attempts":attempts}

def metrics(rows,pred="pos_bpm",truth="gt_ppg_bpm"):
    if not rows: return {"n":0}
    err=np.asarray([r[pred]-r[truth] for r in rows],float)
    return {"n":len(rows),"MAE_bpm":float(np.mean(abs(err))),"RMSE_bpm":float(np.sqrt(np.mean(err**2))),"bias_bpm":float(np.mean(err))}

def align_timestamps(rgb, gt):
    if gt.shape != (3,len(rgb)): raise ValueError("Frame/GT sample count mismatch; explicit alignment required")
    t=gt[2]-gt[2,0]
    if np.any(np.diff(t)<0) or t[-1]<=0: raise ValueError("Reversed/invalid timestamps")
    fs=(len(t)-1)/t[-1]
    if not 20<fs<40: raise ValueError("Unexpected measured sampling rate")
    unique,inverse,counts=np.unique(t,return_inverse=True,return_counts=True)
    # Quantized duplicate timestamps are merged consistently across all modalities.
    joined=np.column_stack([rgb,gt[0],gt[1]])
    summed=np.zeros((len(unique),5))
    np.add.at(summed,inverse,joined)
    averaged=summed/counts[:,None]
    uniform=np.linspace(0,t[-1],len(t))
    resampled=np.column_stack([np.interp(uniform,unique,averaged[:,c]) for c in range(5)])
    info={"duplicate_timestamps":int(len(t)-len(unique)),"max_timestamp_gap_s":float(np.max(np.diff(unique)))}
    return resampled[:,:3],resampled[:,3],resampled[:,4],uniform,fs,info

def ubfc(root,out):
    manifest=json.loads((root.parent/"drive_manifest.json").read_text())
    subjects=sorted({x["path"].split("/")[0] for x in manifest},key=lambda s:int(s[7:]))
    # Fixed split chosen before looking at results. No fitting is done by POS.
    shuffled=np.random.default_rng(2027).permutation(subjects).tolist()
    split={"seed":2027,"train":shuffled[:28],"val":shuffled[28:35],"test":shuffled[35:]}
    (out/"subject_split.json").write_text(json.dumps(split,indent=2))
    rows=[]; windows=[]; failures=[]
    for subject in subjects:
        try:
            folder=root/subject
            for f in ("vid.avi","ground_truth.txt"):
                if not (folder/(f+".complete.json")).exists(): raise ValueError("Download incomplete: "+f)
            gt=np.loadtxt(folder/"ground_truth.txt")
            assert gt.shape[0]==3 and np.isfinite(gt).all()
            rgb,nominal_fs,info=read_rgb(folder/"vid.avi",fullface=True)
            rgb,ppg,sensor,uniform,fs,timing=align_timestamps(rgb,gt)
            t=uniform
            info.update(timing)
            pred=pos(rgb,fs); trim=math.ceil(1.6*fs); keep=slice(trim,len(t)-trim)
            tag=next(k for k in ("train","val","test") if subject in split[k])
            row={"subject":subject,"split":tag,"pos_bpm":estimate_hr(pred[keep],fs),"green_bpm":estimate_hr(rgb[keep,1],fs),"gt_ppg_bpm":estimate_hr(ppg[keep],fs),"sensor_hr_bpm":float(sensor[keep].mean()),"measured_fs":fs,"nominal_fs":nominal_fs,"duration_s":float(t[-1]),**info}
            rows.append(row)
            for start in range(trim,len(t)-trim-round(10*fs)+1,round(10*fs)):
                end=start+round(10*fs)
                windows.append({"subject":subject,"split":tag,"start_s":start/fs,"pos_bpm":estimate_hr(pred[start:end],fs),"green_bpm":estimate_hr(rgb[start:end,1],fs),"gt_ppg_bpm":estimate_hr(ppg[start:end],fs),"sensor_hr_bpm":float(sensor[start:end].mean())})
            np.savez_compressed(out/(subject+"_signals.npz"),rgb=rgb,ppg=ppg,pos=pred,sensor_hr=sensor,t=uniform)
            print(json.dumps(row),flush=True)
        except Exception as e:
            failures.append({"subject":subject,"error":str(e)}); print("FAILED",subject,str(e),flush=True)
    summary={"dataset":"UBFC-rPPG Dataset 2","expected_subjects":42,"processed":len(rows),"coverage":len(rows)/42,"complete":len(rows)==42,"failures":failures,"protocol":"POS and GREEN; whole recording and nonoverlapping 10s windows; trim 1.6s at both ends; 42-180 bpm; actual GT timestamps; no parameter tuning","record_pos_vs_ppg":metrics(rows),"record_green_vs_ppg":metrics(rows,"green_bpm"),"record_pos_vs_sensor":metrics(rows,truth="sensor_hr_bpm"),"window_pos_vs_ppg":metrics(windows),"test_record_pos_vs_ppg":metrics([r for r in rows if r["split"]=="test"])}
    save(rows,out/"recordings.csv"); save(windows,out/"windows.csv")
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def save(rows,path):
    if rows:
        with path.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def selftest():
    fs=30.; t=np.arange(900)/fs; truth=84.
    pulse=np.sin(2*np.pi*truth/60*t)
    rgb=np.column_stack([100+.2*pulse,110+1.2*pulse,90+.1*pulse])
    pred=estimate_hr(pos(rgb,fs)[48:-48],fs)
    assert abs(pred-truth)<.5,(pred,truth)
    # Frequency calibration at lower and upper portions of the declared band.
    for bpm in (48,72,120,168):
        assert abs(estimate_hr(np.sin(2*np.pi*bpm/60*t),fs)-bpm)<.5
    try: estimate_hr(np.ones(900),fs)
    except ValueError: pass
    else: raise AssertionError("Flat signal must fail")
    duplicate_t=t.copy(); duplicate_t[100]=duplicate_t[99]
    aligned=align_timestamps(rgb,np.vstack([pulse,np.full(len(t),truth),duplicate_t]))
    assert aligned[-1]["duplicate_timestamps"]==1
    assert np.isfinite(aligned[0]).all() and np.all(np.diff(aligned[3])>0)
    print(json.dumps({"synthetic_tests":"passed","known_bpm":truth,"recovered_pos_bpm":pred}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path);p.add_argument("--out",type=Path);p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test: selftest()
    else:
        a.out.mkdir(parents=True,exist_ok=True)
        summary=ubfc(a.root,a.out)
        raise SystemExit(0 if summary["complete"] else 2)

