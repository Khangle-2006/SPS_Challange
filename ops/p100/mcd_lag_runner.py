"""Streaming dev60 FaceMesh/POS lag diagnostics. No training or alignment repair."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,csv,json,hashlib,time,traceback,fcntl
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import av,cv2,mediapipe as mp
from mcd_figure5_metrics import pos_unfiltered,lag_metrics,drift_metrics
from mcd_errata_audit import timed,nearest
from mcd_development_gate_v3 import sha,save

POLYS={'forehead':[109,10,338,337,151,108], 'left_cheek':[50,101,205,187,123], 'right_cheek':[280,330,425,411,352]}

def regional_means(rgb,lm):
    h,w=rgb.shape[:2];out=np.full((3,3),np.nan,np.float32);areas=[]
    for j,ids in enumerate(POLYS.values()):
        poly=np.rint(lm[ids]).astype(np.int32)
        # Cropped mask is pixel-identical to full-frame mask; bounds include endpoints.
        x1=max(0,int(poly[:,0].min()));x2=min(w,int(poly[:,0].max())+1)
        y1=max(0,int(poly[:,1].min()));y2=min(h,int(poly[:,1].max())+1)
        if x2<=x1 or y2<=y1:areas.append(0);continue
        mask=np.zeros((y2-y1,x2-x1),np.uint8)
        cv2.fillPoly(mask,[poly-np.array([x1,y1],np.int32)],1)
        area=int(mask.sum());areas.append(area)
        if area>=16:out[j]=cv2.mean(rgb[y1:y2,x1:x2],mask=mask)[:3]
    return out,areas

def trace_video(path,gate,folder,status_callback):
    folder.mkdir(exist_ok=True)
    trace_path=folder/'trace.npz';receipt=folder/'trace_receipt.json'
    if trace_path.exists() and receipt.exists():
        r=json.loads(receipt.read_text())
        assert r['source_sha256']==gate['source_sha256'] and r['trace_sha256']==sha(trace_path)
        with np.load(trace_path) as z:return z['rgb'].copy(),z['valid'].copy(),r
    assert sha(path)==gate['source_sha256'],'Source changed since V3'
    traces=[];valid=[];corrupt=[];counts=[];pts=[];start=time.time()
    with av.open(str(path)) as c,mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,max_num_faces=1,refine_landmarks=False,
        min_detection_confidence=.5,min_tracking_confidence=.5) as mesh:
        c.streams.video[0].codec_context.thread_count=1
        for packet in c.demux(video=0):
            try:frames=packet.decode()
            except av.error.FFmpegError:
                if packet.pos is None or packet.pos<gate['last_media_chunk_pos']:raise
                continue
            for frame in frames:
                rgb=frame.to_ndarray(format='rgb24');h,w=rgb.shape[:2]
                face=mesh.process(rgb);tr=np.full((3,3),np.nan,np.float32);ar=[0,0,0]
                if face.multi_face_landmarks:
                    lm=np.array([(p.x*w,p.y*h) for p in face.multi_face_landmarks[0].landmark],np.float32)
                    tr,ar=regional_means(rgb,lm)
                isbad=bool(frame.is_corrupt)
                traces.append(tr);valid.append(bool(np.isfinite(tr).all() and not isbad))
                corrupt.append(isbad);counts.append(ar)
                pts.append(str(frame.pts*frame.time_base) if frame.pts is not None else None)
                if len(traces)%500==0:status_callback(len(traces),time.time()-start)
    assert len(traces)==gate['gate']['decoded_frames'],('Decode length changed',len(traces))
    assert pts==gate['container_pts_rational'],'Decode presentation sequence changed'
    with trace_path.with_suffix('.npz.tmp').open('wb') as f:
        np.savez_compressed(f,rgb=np.asarray(traces),valid=np.asarray(valid),corrupt=np.asarray(corrupt),areas=np.asarray(counts))
    trace_path.with_suffix('.npz.tmp').replace(trace_path)
    r={'source_sha256':gate['source_sha256'],'trace_sha256':sha(trace_path),'frames':len(traces),
       'valid_frames_before_tail_drop':int(sum(valid)),'seconds':time.time()-start,'polygons':POLYS,
       'versions':{'av':av.__version__,'cv2':cv2.__version__,'mediapipe':mp.__version__}}
    save(receipt,r)
    return np.asarray(traces),np.asarray(valid),r

def grid_signals(t,rgb,ppg,valid,fs):
    if len(t)<2 or np.any(np.diff(t)<=0):raise ValueError('Nonmonotonic metadata clock')
    grid=np.arange(int(np.floor(t[-1]*fs))+1)/fs
    right=np.searchsorted(t,grid,side='left').clip(0,len(t)-1);left=(right-1).clip(0,len(t)-1)
    exact=np.isclose(grid,t[right],rtol=0,atol=1e-10)
    left=np.where(exact,right,left)
    ok=(grid>=t[0])&(grid<=t[-1])&valid[left]&valid[right]&((t[right]-t[left])<=.2)
    rg=np.stack([np.interp(grid,t,rgb[:,j]) for j in range(3)],axis=1)
    pg=np.interp(grid,t,ppg)
    ok &= np.isfinite(rg).all(1)&np.isfinite(pg)
    # Mark every sample in intervals adjoining invalid observations; no dropping/concatenation.
    return grid,rg,pg,ok

def analyze(row,gate,trace,valid,root):
    n=gate['gate']['candidate_prefix_frames'];trace=trace[:n];valid=valid[:n]
    _,mt=timed(root/row['meta']);mt=mt[:n];t=(mt-mt[0])/1e6
    sync_path=Path(row['ppg_sync']);sync_path=sync_path if sync_path.is_absolute() else root/sync_path
    sy=np.loadtxt(sync_path,ndmin=2)
    if len(sy)<n:raise ValueError('Published sync shorter than retained prefix')
    ppg=sy[:n,0];pv,pt=timed(root/row['ppg']);nv,dt=nearest(mt,pt,pv)
    if not np.array_equal(nv,ppg):raise ValueError('Published sync differs from nearest raw PPG prefix')
    if not np.all(abs(dt-sy[:n,1])<=1.000001e-6):raise ValueError('Published sync distance column differs')
    if np.std(ppg)<1e-12:raise ValueError('Constant reference PPG')
    # No endpoint extrapolation outside the contact recording wall-clock coverage.
    valid=valid & (mt>=pt.min()) & (mt<=pt.max())
    # Full record uses common actual PPG/video support; trim only unsupported endpoints,
    # not invalid face or interior observations. Report exact trimmed support.
    support=np.flatnonzero((mt>=pt.min())&(mt<=pt.max()))
    if not len(support):raise ValueError('No common PPG/video clock support')
    lo,hi=int(support[0]),int(support[-1])+1
    fs=24 if row['camera']=='IriunWebcam' else 30
    rg=trace.mean(axis=1)
    grid,gr,gp,ok=grid_signals(t,rg,ppg,valid,fs)
    result={'video':row['video'],'subject':row['patient_id'],'camera':row['camera'],'step':row['step'],
      'official_fold':int(row['official_fold']),'frames_retained':n,'fs':fs,
      'duration_s':float(t[-1]),'reference_prefix_exact':True,'metadata_sha256':sha(root/row['meta']),
      'sync_sha256':sha(sync_path),'raw_ppg_sha256':sha(root/row['ppg']),
      'nearest_ppg_dt_max_s':float(dt.max()),'source_valid_fraction':float(valid.mean()),
      'grid_valid_fraction':float(ok.mean()),'fullrecord':None,'fullrecord_failure':None,'windows':[],
      'common_support_start_s':float(t[lo]),'common_support_end_s':float(t[hi-1])}
    full_sel=(grid>=t[lo])&(grid<=t[hi-1])
    try:
        if not ok[full_sel].all():raise ValueError('Interior invalid face/corruption/interpolation support')
        result['fullrecord']=lag_metrics(pos_unfiltered(gr[full_sel],fs),gp[full_sel],fs)
    except Exception as e:result['fullrecord_failure']=str(e)
    windows=int(np.floor(t[-1]/30))
    for k in range(windows):
        start=k*30;sel=(grid>=start)&(grid<start+30)
        entry={'index':k,'start_s':start,'midpoint_s':start+15,'result':None,'failure':None}
        try:
            if int(sel.sum())!=30*fs or not ok[sel].all():raise ValueError('Window has invalid interpolation/face/reference support')
            entry['result']=lag_metrics(pos_unfiltered(gr[sel],fs),gp[sel],fs)
        except Exception as e:entry['failure']=str(e)
        result['windows'].append(entry)
    vv=[x for x in result['windows'] if x['result'] is not None]
    result['drift']=drift_metrics([x['midpoint_s'] for x in vv],[x['result']['lag_seconds'] for x in vv])
    result['suffix_seconds']=float(t[-1]-windows*30)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--wait-gate',action='store_true');ap.add_argument('--limit',type=int,default=360)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    cv2.setNumThreads(1)
    gate_dir=a.audit/'dev60_gate_v3'
    while True:
        state=json.loads((gate_dir/'status.json').read_text()) if (gate_dir/'status.json').exists() else {}
        if state.get('state')=='complete' and state.get('processed')==360:break
        if not a.wait_gate:raise RuntimeError('Full V3 cohort not completed')
        save(a.output/'status.json',{'state':'waiting_for_v3','gate_processed':state.get('processed',0),'updated_unix':time.time()})
        time.sleep(20)
    plan=json.loads((a.audit/'development_plan_v3_curated.json').read_text())
    runhash=hashlib.sha256((a.audit/'FIGURE5_PROTOCOL.txt').read_bytes()+(a.audit/'LAG_IMPLEMENTATION_NOTES.txt').read_bytes()+
          (a.audit/'development_plan_v3_curated.json').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    receipt={'run_hash':runhash,'gate_protocol_hash':state['protocol_sha256'],'protocol':'FIGURE5_PROTOCOL + LAG_IMPLEMENTATION_NOTES',
             'worker_sha256':sha(Path(__file__)),'kernels_sha256':sha(Path(__file__).with_name('mcd_figure5_metrics.py'))}
    if (a.output/'run_manifest.json').exists():assert json.loads((a.output/'run_manifest.json').read_text())==receipt
    else:save(a.output/'run_manifest.json',receipt)
    eligible=[]
    for row in plan['development_records']:
        gate=json.loads((gate_dir/(Path(row['video']).stem+'.json')).read_text())
        assert gate['protocol_sha256']==state['protocol_sha256']
        if gate['accepted_operational_prefix']:eligible.append((row,gate))
    save(a.output/'eligible_records.json',[x[0] for x in eligible])
    results=[]
    for row,gate in eligible[:a.limit]:
        folder=a.output/Path(row['video']).stem;folder.mkdir(exist_ok=True)
        base={'state':'running','eligible_total':len(eligible),'processed':len(results),'video':row['video'],'run_hash':runhash}
        def callback(frames,elapsed):save(a.output/'status.json',{**base,'frames_current':frames,'current_elapsed_s':elapsed,'updated_unix':time.time()})
        callback(0,0)
        path=folder/'result.json'
        if path.exists():
            result=json.loads(path.read_text());assert result['run_hash']==runhash
        else:
            try:
                trace,valid,tr=trace_video(a.root/row['video'],gate,folder,callback)
                result=analyze(row,gate,trace,valid,a.root);result['trace_receipt']=tr
            except Exception as e:
                result={'video':row['video'],'subject':row['patient_id'],'camera':row['camera'],'step':row['step'],
                  'official_fold':int(row['official_fold']),'error':str(e),'traceback':traceback.format_exc(),'fullrecord':None,'windows':[]}
            result['run_hash']=runhash;save(path,result)
        results.append(result)
        state2={**base,'processed':len(results),'fullrecord_evaluable':sum(x.get('fullrecord') is not None for x in results),
               'processing_errors':sum('error' in x for x in results),'updated_unix':time.time()}
        save(a.output/'status.json',state2);print(json.dumps(state2),flush=True)
    save(a.output/'results.json',results)
    state2.update(state='complete' if len(results)==len(eligible) else 'partial_smoke')
    save(a.output/'status.json',state2)

if __name__=='__main__':main()
