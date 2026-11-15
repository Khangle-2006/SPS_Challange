"""Matched native-ROI POS before/after full-frame H264, validation subjects only."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np
from scipy import signal
from hr_baseline import align_timestamps,pos,estimate_hr,detrend
R=Path('/data/khangle/facevitals')
BASE=R/'datasets/UBFC-rPPG/efficientphys36_trainval_v1'
OUT=R/'results/pos_h264_validation_v1'

def native_trace(video,boxes):
    cap=cv2.VideoCapture(str(video));assert cap.isOpened();rgb=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        x1,y1,x2,y2=boxes[len(rgb)]
        rgb.append(cv2.mean(f[y1:y2,x1:x2])[:3][::-1])
    cap.release();assert len(rgb)==len(boxes)
    return np.asarray(rgb)

def spectral(x):
    y=signal.sosfiltfilt(signal.butter(1,[.75,3.],btype='bandpass',fs=30,output='sos'),detrend(x))
    f,p=signal.periodogram(y,fs=30,window='hann',nfft=4096,detrend=False)
    ix=[i for i in signal.find_peaks(p)[0] if .75<=f[i]<=3.]
    ix=sorted(ix,key=lambda i:p[i],reverse=True)[:3]
    return [{'bpm':float(f[i]*60),'relative_power':float(p[i]/max(p[ix[0]],1e-30))} for i in ix]

def safe_hr(x):
    try:return estimate_hr(x,30.)
    except ValueError:return None

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['train','val'],default='val');args=parser.parse_args()
    tag=args.split
    if tag=='train':OUT=R/'results/pos_h264_training_v1'
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    split=json.loads((BASE/'subject_split.json').read_text())
    rows=[r for r in json.loads((BASE/'clips.json').read_text()) if r['split']==tag]
    assert len(rows)==(42 if tag=='val' else 161)
    reports=[]
    for s in split[tag]:
        meta=np.load(BASE/(s+'_meta.npz'));boxes=meta['boxes']
        gt=np.loadtxt(R/'datasets/UBFC-rPPG/DATASET_2'/s/'ground_truth.txt')
        raw=native_trace(R/'datasets/UBFC-rPPG/DATASET_2'/s/'vid.avi',boxes)
        encoded=native_trace(R/'datasets/UBFC-rPPG'/('validation_h264_500k_v1' if tag=='val' else 'training_h264_500k_v1')/(s+'.mp4'),boxes)
        clean,ppg,_,t,fs,_=align_timestamps(raw,gt)
        comp,ppg2,_,t2,fs2,_=align_timestamps(encoded,gt)
        np.testing.assert_array_equal(t,t2);np.testing.assert_array_equal(ppg,ppg2);assert fs==fs2
        pc=pos(clean,fs);ph=pos(comp,fs)
        np.savez_compressed(OUT/(s+'_traces.npz'),t=t,ppg=ppg,clean_rgb=clean,h264_rgb=comp,clean_pos=pc,h264_pos=ph)
        for r in [r for r in rows if r['subject']==s]:
            q=r['start_s']+np.arange(300)/30
            yc=np.interp(q,t,pc);yh=np.interp(q,t,ph);yg=np.interp(q,t,ppg)
            reference=estimate_hr(yg,30.);assert abs(reference-r['gt_hr'])<1e-8
            hc=safe_hr(yc);hh=safe_hr(yh)
            # Reproduce the previously used clean POS before interpreting a compression change.
            assert hc is not None and abs(hc-r['pos_hr'])<1e-8,(s,r['start_s'],hc,r['pos_hr'])
            rgbc=np.column_stack([np.interp(q,t,clean[:,k]) for k in range(3)])
            rgbh=np.column_stack([np.interp(q,t,comp[:,k]) for k in range(3)])
            row={**r,'clean_pos_hr':hc,'h264_pos_hr':hh,
                 'clean_pos_error':abs(hc-reference),'h264_pos_error':abs(hh-reference) if hh is not None else 135.,
                 'h264_invalid':hh is None,'rgb_mean_absolute_distortion':float(np.mean(np.abs(rgbc-rgbh))),
                 'rgb_temporal_correlation':[float(np.corrcoef(rgbc[:,k],rgbh[:,k])[0,1]) for k in range(3)],
                 'ppg_peaks':spectral(yg),'clean_pos_peaks':spectral(yc),'h264_pos_peaks':spectral(yh)}
            reports.append(row)
        print('COMPLETE',s,flush=True)
    def summary(key):
        e=np.array([r[key] for r in reports])
        return {'n':len(e),'MAE':float(e.mean()),'RMSE':float(np.sqrt(np.mean(e*e))),
                'by_subject':{s:float(np.mean([r[key] for r in reports if r['subject']==s])) for s in split[tag]}}
    result={'clean':summary('clean_pos_error'),'h264':summary('h264_pos_error'),
            'h264_invalid':sum(r['h264_invalid'] for r in reports),
            'clean_POS_reproduced':True,'test_opened':False,'split':tag,'records':reports,
            'scope':'Same native ROI coordinates, timestamps, PPG reference, 30Hz query grid and estimator. POS formed on each whole recording before slicing identical windows. Compression-induced face detection failures not tested.'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
