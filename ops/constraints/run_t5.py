"""Single CPU streaming cohort run; completed receipts are immutable/reusable."""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from simulate import PROTOCOL, simulate, write_json, sha256

CONDITIONS={
 'c5_500k_10fps':dict(kind='c5',fps=10),
 'c5_500k_30fps':dict(kind='c5',fps=30),
 'c4_QVGA':dict(kind='c4',size=(320,240)),
 'c4_c5':dict(kind='c4_c5',size=(320,240),fps=10),
 'c1_050':dict(kind='c1',gain=.5),
 'c1_025':dict(kind='c1',gain=.25),
 'c1_0125':dict(kind='c1',gain=.125)}


def trace_video(video, timestamps):
    import cv2
    from video_roi_frontend import Tracker
    cv2.setNumThreads(1)
    n=len(timestamps);fs=(n-1)/(timestamps[-1]-timestamps[0]);tracker=Tracker(fs)
    values=np.full((n,3),np.nan); valid=np.zeros(n,bool); boxes=np.zeros((n,4),np.int32)
    cap=cv2.VideoCapture(str(video));i=0
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if i>=n:raise ValueError('Extra derivative frame')
            h,w=bgr.shape[:2]; gray=None
            if tracker.needs_detection(i):
                gray=cv2.cvtColor(cv2.resize(bgr,(w//2,h//2)),cv2.COLOR_BGR2GRAY)
            good,box,_=tracker.step(i,w,h,gray)
            if good:
                x1,y1,x2,y2=box;values[i]=cv2.mean(bgr[y1:y2,x1:x2])[:3][::-1]
                valid[i]=True;boxes[i]=box
            i+=1
    finally:cap.release()
    if i!=n:raise ValueError(f'Derivative decode {i}/{n}')
    return values,valid,boxes,{'decoded_frames':i,'roi_valid_frames':int(valid.sum()),'detector_attempts':tracker.attempts,'detector_successes':tracker.successes}


def merge_duplicates(t,x,valid):
    u,inv,counts=np.unique(t,return_inverse=True,return_counts=True)
    y=np.zeros((len(u),)+x.shape[1:]);np.add.at(y,inv,np.nan_to_num(x));y/=counts.reshape((-1,)+(1,)*(x.ndim-1))
    good=np.ones(len(u),bool);np.logical_and.at(good,inv,valid)
    return u,y,good


def supported_interval(t,valid,start,end,max_gap=.2):
    if len(t)<2 or start<t[0]-1e-9 or end>t[-1]+1e-9:return False,'support_boundary'
    # Match T1 semantics: a bad edge matters only when positively overlapping window.
    edge=(np.diff(t)>max_gap)|~valid[:-1]|~valid[1:]
    if np.any(edge & (t[:-1]<end) & (t[1:]>start)):return False,'gap_or_invalid_roi'
    if np.any(~valid & (t>=start)&(t<=end)):return False,'invalid_roi'
    return True,None


def evaluate(record, base_windows, directory, fs):
    from hr_baseline import pos,estimate_hr
    mapping=np.load(directory/'frame_mapping.npz'); times=mapping['timestamps_s'];indices=mapping['source_frame_indices']
    manifest=json.loads((directory/'manifest.json').read_text())
    if (directory/'trace.npz').exists():
        d=np.load(directory/'trace.npz');rgb=d['rgb'];valid=d['valid']; frontend=json.loads((directory/'frontend.json').read_text())
    else:
        rgb,valid,boxes,frontend=trace_video(manifest['video'],times)
        np.savez_compressed(directory/'trace.npz',rgb=rgb,timestamps_s=times,valid=valid,source_frame_indices=indices,boxes=boxes)
        write_json(directory/'frontend.json',frontend)
    source_valid=np.asarray(record.metadata.get('frame_valid',np.ones(len(record.timestamps),bool)),bool)
    times,rgb,valid=merge_duplicates(times,rgb,valid&source_valid[indices])
    label=record.require_synchronized_ppg()
    pt,ppg,pv=merge_duplicates(label.timestamps,label.values,label.valid_mask)
    waves=[];starts=[];rows=[]
    for w in base_windows['rejected_windows']:
        rows.append({'index':w['index'],'start_s':w['start_s'],'end_s':w['end_s'],'failure':'T1:'+','.join(w['reasons']),'reference_hr':None,'old_hr':None})
    for w in base_windows['valid_windows']:
        start,end=w['start_s'],w['end_s'];r={'index':w['index'],'start_s':start,'end_s':end,'failure':None,'reference_hr':None,'old_hr':None}
        try:
            good,reason=supported_interval(times,valid,start,end)
            if not good:raise ValueError(reason)
            good,reason=supported_interval(pt,pv,start,end)
            if not good:raise ValueError('ppg_'+reason)
            q=start+np.arange(round(10*fs))/fs
            x=np.stack([np.interp(q,times,rgb[:,c]) for c in range(3)],axis=1)
            waveform=pos(x,fs); truth=np.interp(q,pt,ppg)
            # Reference uses fixed 30Hz grid for every condition, not degraded sampling.
            refq=start+np.arange(300)/30
            ref=np.interp(refq,pt,ppg)
            r['reference_hr']=estimate_hr(ref,30,band=(.75,3.))
            r['old_hr']=estimate_hr(waveform,fs,band=(.75,3.))
            r['wave_index']=len(waves);waves.append(waveform);starts.append(start)
        except (ValueError,RuntimeError) as e:r['failure']=str(e)
        rows.append(r)
    rows.sort(key=lambda x:x['index'])
    a=np.asarray(waves,dtype=np.float64).reshape((-1,1,round(10*fs)))
    np.savez_compressed(directory/'waveforms.npz',waves=a,fs=np.asarray(fs),starts=np.asarray(starts))
    # v2 frozen diagnostic only. T2 did not pass acceptance; old decoder remains default.
    v2_info={'available':False,'default':False}
    try:
        from decode_hr_v2 import decode_sequence
        predictions=decode_sequence(a,fs,starts=np.asarray(starts),mode='causal')
        v2_info={'available':True,'default':False,'note':'Frozen diagnostic; T2 failed acceptance on existing paired replay.'}
        for r in rows:
            if 'wave_index' in r:
                p=predictions[r['wave_index']]
                # Contract is a dictionary; retain all diagnostic fields in sidecar.
                r['v2']={k:p[k] for k in ['hr_bpm','confidence','abstain','invalid','reset','top2_peaks_bpm'] if k in p}
                for k,v in list(r['v2'].items()):
                    if isinstance(v,(float,np.floating)) and not np.isfinite(v):r['v2'][k]=None
    except Exception as e:
        v2_info['error']=f'{type(e).__name__}: {e}'
    receipt={'recording_id':record.recording_id,'subject_id':record.subject_id,'dataset':record.metadata['dataset'],
      'condition':directory.parent.name,'fs':fs,'candidate_windows':base_windows['candidate_windows'],
      'base_valid_windows':len(base_windows['valid_windows']),'successful_windows':sum(r['failure'] is None for r in rows),
      'rows':rows,'frontend':frontend,'v2':v2_info,'timestamp_mapping':'source index retained; no offset',
      'protocol_sha256':sha256(Path(__file__).with_name('protocol.json'))}
    write_json(directory/'evaluation.json',receipt)
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['MCD','UBFC2'],required=True)
    ap.add_argument('--root',required=True);ap.add_argument('--audit');ap.add_argument('--output',required=True);ap.add_argument('--shared',required=True)
    ap.add_argument('--limit',type=int);ap.add_argument('--budget-gb',type=float,required=True);args=ap.parse_args()
    sys.path.insert(0,args.shared)
    from data_adapters import MCD,UBFC2
    from window_support import recording_windows
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    protocol_sha=sha256(Path(__file__).with_name('protocol.json'))
    snapshot=out/'protocol.json'
    if snapshot.exists() and sha256(snapshot)!=protocol_sha:raise RuntimeError('Protocol fingerprint mismatch; new output required')
    snapshot.write_bytes(Path(__file__).with_name('protocol.json').read_bytes())
    dataset=MCD(args.root,args.audit) if args.dataset=='MCD' else UBFC2(args.root)
    names=dataset.recordings()
    if args.dataset=='MCD':names=[n for n in names if dataset.rows[n]['camera']=='FullHDwebcam']
    if args.limit:names=names[:args.limit]
    expected=119 if args.dataset=='MCD' else 42
    if not args.limit and len(names)!=expected:raise RuntimeError(f'Cohort count {len(names)} != {expected}')
    write_json(out/'cohort.json',{'dataset':args.dataset,'recordings':names,'expected':expected,'protocol_sha256':protocol_sha})
    status={'state':'running','pid':os.getpid(),'dataset':args.dataset,'started_unix':time.time(),'total_recordings':len(names),'total_condition_records':len(names)*7,
      'completed':0,'failures':[],'protocol_sha256':protocol_sha,'budget_bytes':int(args.budget_gb*1e9)}
    write_json(out/'status.json',status)
    for name in names:
        record=None;base=None
        for cond,params in CONDITIONS.items():
            directory=out/cond/name;directory.mkdir(parents=True,exist_ok=True)
            status.update(current_recording=name,current_condition=cond,updated_unix=time.time());write_json(out/'status.json',status)
            try:
                if (directory/'evaluation.json').exists():
                    old=json.loads((directory/'evaluation.json').read_text())
                    if old['protocol_sha256']!=protocol_sha:raise RuntimeError('Existing evaluation fingerprint mismatch')
                else:
                    if record is None:record=dataset.load(name);base=recording_windows(record)
                    if not (directory/'manifest.json').exists():
                        # Stable per-recording-condition seed; unaffected by traversal/resume order.
                        seed=PROTOCOL['seed']+int(hashlib.sha256((args.dataset+'/'+name+'/'+cond).encode()).hexdigest()[:8],16)
                        simulate(record.frames,record.timestamps,directory,seed=seed,budget_root=out,budget_bytes=int(args.budget_gb*1e9),**params)
                    fs=params.get('fps',30)
                    evaluate(record,base,directory,fs)
                status['completed']+=1
                print(json.dumps({'event':'completed','recording':name,'condition':cond,'completed':status['completed'],'time':time.time()}),flush=True)
            except Exception as e:
                failure={'recording':name,'condition':cond,'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()}
                status['failures'].append(failure);write_json(directory/'failure.json',failure)
                print(json.dumps({'event':'failure',**failure}),flush=True)
                if 'budget' in str(e).lower():
                    status['state']='stopped_budget';write_json(out/'status.json',status);return
            write_json(out/'status.json',status)
    status.update(state='complete' if not status['failures'] else 'complete_with_failures',finished_unix=time.time())
    write_json(out/'status.json',status)
    from report_t5 import report
    report(out)


if __name__=='__main__':main()
