"""Native/H264 pixels x native/H264 ROIs, streaming CPU-only diagnostic E0."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse, json, hashlib, subprocess, time, traceback, fcntl, shutil
from pathlib import Path
from fractions import Fraction
import numpy as np
import av, cv2, mediapipe as mp
from mcd_lag_runner import regional_means, POLYS
from mcd_errata_audit import timed
from mcd_development_gate_v3 import sha, save
from mcd_e0_metrics import ARMS, analyze_windows, summarize


def native_frames(path, gate):
    n = gate['gate']['candidate_prefix_frames']
    count = 0
    with av.open(str(path)) as c:
        c.streams.video[0].codec_context.thread_count = 1
        for frame in c.decode(video=0):
            if count >= n:
                return
            pts = str(frame.pts*frame.time_base) if frame.pts is not None else None
            if pts != gate['container_pts_rational'][count] or frame.is_corrupt:
                raise ValueError('Retained source sequence differs from V3')
            count += 1
            yield frame.to_ndarray(format='rgb24')
            if count == n:
                return  # Do not ask decoder to traverse the already-audited truncated tail.
    if count != n:
        raise ValueError(f'Source prefix length {count} != {n}')


def encode(path, gate, folder):
    out = folder/'h264_500k.mp4'
    receipt = folder/'encode.json'
    if out.exists() and receipt.exists():
        r = json.loads(receipt.read_text())
        if r['source_sha256'] != gate['source_sha256'] or r['encoded_sha256'] != sha(out):
            raise ValueError('Encoded cache hash mismatch')
        return out, r
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        w,h = stream.width,stream.height
        fps = stream.average_rate or stream.base_rate
        if fps is None or not 1 < float(fps) < 100:
            raise ValueError('Invalid source container rate')
    tmp = folder/'h264_500k.partial.mp4'
    cmd = ['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','rawvideo',
           '-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(fps),
           '-i','pipe:0','-an','-c:v','libx264','-preset','medium','-pix_fmt','yuv420p',
           '-b:v','500k','-maxrate','500k','-bufsize','1000k','-g',str(round(float(fps)*2)),
           '-sc_threshold','0','-threads','1','-filter_threads','1','-movflags','+faststart',str(tmp)]
    with (folder/'encode.log').open('wb') as log:
        p = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            count=0
            for rgb in native_frames(path,gate):
                p.stdin.write(rgb.tobytes());count+=1
            p.stdin.close()
            if p.wait(timeout=120) != 0:
                raise RuntimeError('ffmpeg encode failed; inspect encode.log')
        except BaseException:
            if p.poll() is None:p.kill();p.wait()
            raise
    if count != gate['gate']['candidate_prefix_frames']:
        raise ValueError('Encode input count mismatch')
    tmp.replace(out)
    with av.open(str(out)) as c:
        stream=c.streams.video[0]
        if stream.frames != count or stream.width != w or stream.height != h:
            raise ValueError('Encoded header count/dimensions mismatch')
    r=dict(source_sha256=gate['source_sha256'],encoded_sha256=sha(out),frames=count,
           width=w,height=h,container_fps=str(fps),command=cmd,
           file_bytes=out.stat().st_size,container_average_bps=out.stat().st_size*8/(count/float(fps)))
    save(receipt,r)
    return out,r


def cross_means(images, landmarks):
    result=np.full((4,3,3),np.nan,np.float32)
    for pixel in range(2):
        for roi in range(2):
            if landmarks[roi] is not None:
                result[pixel*2+roi],_=regional_means(images[pixel],landmarks[roi])
    return result


def extract(path, encoded, gate, folder, callback):
    ids=sorted({x for indices in POLYS.values() for x in indices})
    allrgb=[];coordinates=[]
    kwargs=dict(static_image_mode=False,max_num_faces=1,refine_landmarks=False,
                min_detection_confidence=.5,min_tracking_confidence=.5)
    with av.open(str(encoded)) as c, mp.solutions.face_mesh.FaceMesh(**kwargs) as mn, mp.solutions.face_mesh.FaceMesh(**kwargs) as mh:
        c.streams.video[0].codec_context.thread_count=1
        compressed=iter(c.decode(video=0))
        for i,native in enumerate(native_frames(path,gate)):
            frame=next(compressed,None)
            if frame is None or frame.is_corrupt:raise ValueError('Encoded prefix missing/corrupt frame')
            h264=frame.to_ndarray(format='rgb24')
            if h264.shape != native.shape:raise ValueError('Cross-cell dimensions differ')
            h,w=native.shape[:2];landmarks=[]
            for image,mesh in [(native,mn),(h264,mh)]:
                face=mesh.process(image)
                lm=np.array([(p.x*w,p.y*h) for p in face.multi_face_landmarks[0].landmark],np.float32) if face.multi_face_landmarks else None
                landmarks.append(lm)
            allrgb.append(cross_means([native,h264],landmarks))
            coordinates.append(np.stack([lm[ids] if lm is not None else np.full((len(ids),2),np.nan,np.float32) for lm in landmarks]))
            if (i+1)%500==0:callback('roi',i+1)
        if next(compressed,None) is not None:raise ValueError('Extra encoded frames')
    arr=np.asarray(allrgb)
    if len(arr)!=gate['gate']['candidate_prefix_frames']:raise ValueError('Trace count mismatch')
    with (folder/'trace.npz.tmp').open('wb') as f:
        np.savez_compressed(f,rgb=arr,landmarks=np.asarray(coordinates),landmark_ids=np.asarray(ids),arms=np.asarray(ARMS))
    (folder/'trace.npz.tmp').replace(folder/'trace.npz')
    return arr


def run_record(row, gate, root, folder, callback):
    path=root/row['video']
    if sha(path)!=gate['source_sha256']:raise ValueError('Source hash changed')
    callback('encode',0)
    encoded,receipt=encode(path,gate,folder)
    rgb=extract(path,encoded,gate,folder,callback)
    n=len(rgb);_,mt=timed(root/row['meta']);mt=mt[:n]
    if len(mt)!=n:raise ValueError('Short metadata')
    t=(mt-mt[0])/1e6
    sy=np.loadtxt(root/row['ppg_sync'],ndmin=2)
    if len(sy)<n:raise ValueError('Short published PPG sync')
    _,pt=timed(root/row['ppg'])
    support=(mt>=pt.min())&(mt<=pt.max())
    windows=analyze_windows(t,rgb,sy[:n,0],support)
    return dict(video=row['video'],subject=row['patient_id'],step=row['step'],camera=row['camera'],
                official_fold=int(row['official_fold']),frames=n,duration_s=float(t[-1]),
                candidate_windows=int(np.floor(t[-1]/10)),windows=windows,encode=receipt,
                valid_trace_fraction={a:float(np.isfinite(rgb[:,i]).all(axis=(1,2)).mean()) for i,a in enumerate(ARMS)},
                trace_sha256=sha(folder/'trace.npz'),metadata_sha256=sha(root/row['meta']),
                sync_sha256=sha(root/row['ppg_sync']),source_labels_unchanged=True,offset_corrected=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--limit',type=int,default=119)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if hasattr(os,'sched_setaffinity'):
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[min(1,len(allowed)-1)]})
    cv2.setNumThreads(1)
    plan_path=a.audit/'development_plan_v3_curated.json'
    allrows=json.loads(plan_path.read_text())['development_records']
    rows=[r for r in allrows if r['camera']=='FullHDwebcam' and r['patient_id']!='4952'
          and not (r['patient_id']=='7412' and r['step']=='after')]
    if len(rows)!=119 or len({r['patient_id'] for r in rows})!=60:raise ValueError('Unexpected E0 cohort')
    names=['mcd_e0_runner.py','mcd_e0_metrics.py','hr_baseline.py','mcd_lag_runner.py',
           'mcd_errata_audit.py','mcd_development_gate_v3.py','mcd_figure5_metrics.py']
    manifest=dict(protocol_sha256=sha(a.protocol),plan_sha256=sha(plan_path),
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in names},
        versions=dict(av=av.__version__,cv2=cv2.__version__,mediapipe=mp.__version__,numpy=np.__version__),
        ffmpeg_version=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        cohort=[r['video'] for r in rows],offset_corrected=False,lag_gate_required=False)
    runhash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest['run_hash']=runhash
    mpth=a.output/'run_manifest.json'
    if mpth.exists():
        if json.loads(mpth.read_text())!=manifest:raise ValueError('Run fingerprint changed; use a versioned output')
    else:save(mpth,manifest)
    save(a.output/'cohort.json',rows)
    records=[];consecutive_errors=0
    for row in rows[:a.limit]:
        folder=a.output/Path(row['video']).stem;folder.mkdir(exist_ok=True)
        base=dict(state='running',expected=119,processed=len(records),video=row['video'],run_hash=runhash,pid=os.getpid())
        def callback(stage,frames):save(a.output/'status.json',{**base,'stage':stage,'frames_current':frames,'updated_unix':time.time()})
        callback('start',0)
        p=folder/'result.json'
        if p.exists():
            record=json.loads(p.read_text())
            if record.get('run_hash')!=runhash:raise ValueError('Cached result fingerprint mismatch')
        else:
            if sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())>19_000_000_000 or shutil.disk_usage(a.output).free<25_000_000_000:
                raise RuntimeError('E0 storage guard; partial results preserved')
            gate=json.loads((a.audit/'dev60_gate_v3'/(Path(row['video']).stem+'.json')).read_text())
            try:
                record=run_record(row,gate,a.root,folder,callback)
            except Exception as e:
                _,mt=timed(a.root/row['meta']);n=gate['gate']['candidate_prefix_frames']
                record=dict(video=row['video'],subject=row['patient_id'],step=row['step'],camera=row['camera'],
                            error=str(e),traceback=traceback.format_exc(),windows=[],
                            candidate_windows=int((mt[n-1]-mt[0])//10_000_000))
            record['run_hash']=runhash;save(p,record)
        records.append(record)
        consecutive_errors=consecutive_errors+1 if 'error' in record else 0
        state={**base,'processed':len(records),'processing_errors':sum('error' in r for r in records),'updated_unix':time.time()}
        save(a.output/'status.json',state);print(json.dumps(state),flush=True)
        if consecutive_errors>=3:raise RuntimeError('Three consecutive record failures; inspect before continuing')
    save(a.output/'results.json',records)
    summary=summarize(records);summary.update(complete=len(records)==119,run_hash=runhash)
    save(a.output/'summary.json',summary)
    save(a.output/'status.json',{**state,'state':'complete' if len(records)==119 else 'partial_smoke'})


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();raise
