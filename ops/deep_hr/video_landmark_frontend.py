"""Label-free FaceMesh 468 frontend. Missing measurements remain NaN.

MediaPipe 0.10.21: Apache-2.0; bundled canonical model retains its upstream
license in assets/MEDIAPIPE_LICENSE. Confidence is NOT available per landmark.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import time
import platform
import os
import numpy as np

REGION_NAMES = ('forehead', 'cheek_left', 'cheek_right', 'periocular')
ROI_INDICES = ([109,10,338,337,151,108], [50,101,205,187,123],
               [280,330,425,411,352], [55,8,285,417,351,6,122,193])
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
             379,378,400,377,152,148,176,149,150,136,172,58,132,93,
             234,127,162,21,54,103,67,109]
EYE_LEFT = [33,160,158,133,153,144]
EYE_RIGHT = [362,385,387,263,373,380]
MOUTH = [61,40,37,0,267,270,291,321,314,17,84,91]
STABLE = [33,133,362,263,168,6,1,98,327,152]
TIMING_NAMES = ['detector_aux_ms','landmark_graph_ms','rasterize_ms',
                'affine_ms','encode_ms','total_with_decode_ms']


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def shrink_polygon(points, fraction=.12):
    points=np.asarray(points,float)
    return points.mean(axis=0)+(1-fraction)*(points-points.mean(axis=0))


def polygons_from_landmarks(landmarks):
    """Pixel XY polygons; slot 0 changes meaning only for flagged small faces."""
    oval=landmarks[FACE_OVAL]
    small=bool(np.min(np.ptp(oval,axis=0))<40)
    if small: return [shrink_polygon(oval), None, None, None], small
    return [shrink_polygon(landmarks[idx]) for idx in ROI_INDICES], small


def rasterize_means(rgb, polygons, holes=None):
    import cv2
    h,w=rgb.shape[:2]; means=np.full((4,3),np.nan,np.float32)
    valid=np.zeros(4,bool); areas=np.zeros(4,np.int32)
    for r,poly in enumerate(polygons):
        if poly is None or not np.isfinite(poly).all():continue
        p=np.rint(poly).astype(np.int32)
        x0,y0=np.maximum(p.min(axis=0),0);x1,y1=np.minimum(p.max(axis=0)+1,[w,h])
        if x1<=x0 or y1<=y0:continue
        mask=np.zeros((y1-y0,x1-x0),np.uint8)
        cv2.fillPoly(mask,[p-[x0,y0]],255)
        if r==0 and holes:
            for hole in holes:cv2.fillPoly(mask,[np.rint(hole).astype(np.int32)-[x0,y0]],0)
        areas[r]=int(np.count_nonzero(mask))
        if areas[r]>=16:
            means[r]=cv2.mean(rgb[y0:y1,x0:x1],mask=mask)[:3];valid[r]=True
    return means,valid,areas


def canonical_template():
    path=Path(__file__).parent/'assets'/'canonical_face_model.obj'
    vertices=np.array([[float(x) for x in line.split()[1:4]]
                       for line in path.read_text().splitlines() if line.startswith('v ')])
    if vertices.shape!=(468,3):raise ValueError('Canonical model must have 468 vertices')
    xy=vertices[:,:2].copy();xy[:,1]*=-1
    lo,hi=xy[FACE_OVAL].min(0),xy[FACE_OVAL].max(0)
    return (xy-(lo+hi)/2)*(60/np.max(hi-lo))+35.5


def least_squares_affine(landmarks, template):
    x=np.column_stack([np.asarray(landmarks)[STABLE],np.ones(len(STABLE))])
    if not np.isfinite(x).all() or np.linalg.matrix_rank(x)<3:
        return np.full((2,3),np.nan,np.float32)
    return np.linalg.lstsq(x,template[STABLE],rcond=None)[0].T.astype(np.float32)


class AlignedWriter:
    def __init__(self,path,fps):
        import av
        self.av=av;self.c=av.open(str(path),'w');self.stream=self.c.add_stream('ffv1',rate=Fraction(float(fps)).limit_denominator(100000))
        self.stream.width=72;self.stream.height=72;self.stream.pix_fmt='bgr0'
        self.stream.codec_context.thread_count=1;self.n=0
    def write(self,rgb):
        frame=self.av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=self.n
        for packet in self.stream.encode(frame):self.c.mux(packet)
        self.n+=1
    def close(self):
        for packet in self.stream.encode():self.c.mux(packet)
        self.c.close()


def save_qa(rgb,polygons,crop,path,recording_id):
    import cv2
    out=rgb.copy();colors=[(255,100,0),(0,255,0),(0,150,255),(255,0,255)]
    for i,p in enumerate(polygons):
        if p is None:continue
        cv2.polylines(out,[np.rint(p).astype(np.int32)],True,colors[i],2)
        xy=np.rint(p.mean(0)).astype(int)
        cv2.putText(out,str(i),tuple(xy),cv2.FONT_HERSHEY_SIMPLEX,.5,colors[i],1)
    panel=np.zeros((out.shape[0]+40,out.shape[1]+160,3),np.uint8)
    panel[:out.shape[0],:out.shape[1]]=out
    panel[:144,out.shape[1]:out.shape[1]+144]=cv2.resize(crop,(144,144),interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel,recording_id+' frame100', (5,out.shape[0]+25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    cv2.imwrite(str(path),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))


def process_recording(recording,out_dir,config=None,qa_path=None,progress=None):
    """Recording.frames/timestamps only; labels are never accessed.

    Returns manifest. trace.npz contains original pixel coordinates, true times,
    measurement validity, and aligned crop transforms. NaN polygon padding is
    structural; consult polygon_lengths. Invalid crop uses black storage frame.
    """
    import cv2
    import mediapipe as mp
    if mp.__version__!='0.10.21':raise RuntimeError('Expected pinned MediaPipe 0.10.21')
    cv2.setNumThreads(1)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    n=len(recording.timestamps);ts=np.asarray(recording.timestamps,dtype=np.float64)
    fps=(n-1)/(ts[-1]-ts[0]) if n>1 else 30
    template=canonical_template()
    data={
      'landmarks':np.full((n,468,2),np.nan,np.float16),
      'landmark_confidence':np.full((n,468),np.nan,np.float16),
      'landmark_confidence_available':np.array(False),
      'polygons':np.full((n,4,36,2),np.nan,np.float16),
      'polygon_lengths':np.zeros((n,4),np.uint8),
      'rgb':np.full((n,4,3),np.nan,np.float32),
      'region_valid':np.zeros((n,4),bool),'polygon_area_px':np.zeros((n,4),np.int32),
      'detected':np.zeros(n,bool),'detector_confidence':np.full(n,np.nan,np.float32),
      'affine':np.full((n,2,3),np.nan,np.float32),'crop_valid':np.zeros(n,bool),
      'timestamps_s':ts,'face_small':np.zeros(n,bool),
      'lower_face_occluded':np.full(n,-1,np.int8),
      'timing_ms':np.zeros((n,6),np.float32)}
    mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,
       refine_landmarks=False,min_detection_confidence=.5,min_tracking_confidence=.5)
    detector=mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=.5)
    writer=AlignedWriter(out_dir/'aligned72.avi',fps);iterator=iter(recording.frames)
    started=time.time();seen=0
    try:
        for i in range(n):
            t0=time.perf_counter();rgb=next(iterator);t=time.perf_counter()
            detection=detector.process(rgb);data['timing_ms'][i,0]=(time.perf_counter()-t)*1000
            if detection.detections:data['detector_confidence'][i]=max(d.score[0] for d in detection.detections)
            t=time.perf_counter();result=mesh.process(rgb);data['timing_ms'][i,1]=(time.perf_counter()-t)*1000
            crop=np.zeros((72,72,3),np.uint8);polygons=[None]*4
            if result.multi_face_landmarks:
                lm=np.array([(p.x*rgb.shape[1],p.y*rgb.shape[0]) for p in result.multi_face_landmarks[0].landmark],np.float32)
                if lm.shape!=(468,2) or not np.isfinite(lm).all():raise ValueError('Invalid FaceMesh landmark shape/values')
                data['landmarks'][i]=lm;data['detected'][i]=True
                t=time.perf_counter();polygons,small=polygons_from_landmarks(lm);data['face_small'][i]=small
                holes=[lm[EYE_LEFT],lm[EYE_RIGHT],lm[MOUTH]] if small else None
                means,valid,areas=rasterize_means(rgb,polygons,holes)
                data['rgb'][i]=means;data['region_valid'][i]=valid;data['polygon_area_px'][i]=areas
                for r,p in enumerate(polygons):
                    if p is not None:data['polygons'][i,r,:len(p)]=p;data['polygon_lengths'][i,r]=len(p)
                data['timing_ms'][i,2]=(time.perf_counter()-t)*1000
                t=time.perf_counter();affine=least_squares_affine(lm,template);data['affine'][i]=affine
                if np.isfinite(affine).all():
                    crop=cv2.warpAffine(rgb,affine,(72,72),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                    data['crop_valid'][i]=True
                data['timing_ms'][i,3]=(time.perf_counter()-t)*1000
            t=time.perf_counter();writer.write(crop);data['timing_ms'][i,4]=(time.perf_counter()-t)*1000
            data['timing_ms'][i,5]=(time.perf_counter()-t0)*1000;seen=i+1
            if qa_path and i==100:save_qa(rgb,polygons,crop,qa_path,recording.recording_id)
            if progress and (i%300==0 or i==n-1):progress(i+1,n)
        try:next(iterator);raise ValueError('More frames than timestamps')
        except StopIteration:pass
    finally:
        writer.close();mesh.close();detector.close()
    if seen!=n:raise ValueError('Incomplete frontend')
    np.savez_compressed(out_dir/'trace.npz',**data)
    manifest={
      'schema_version':1,'frontend':'T3-v1-20261030','recording_id':recording.recording_id,
      'subject_id':recording.subject_id,'n_frames':n,'fps_effective':fps,
      'region_names':list(REGION_NAMES),'small_face_slot0':'whole_face_polygon_with_eye_mouth_holes',
      'detected_frames':int(data['detected'].sum()),'detected_fraction':float(data['detected'].mean()),
      'region_valid_counts':data['region_valid'].sum(0).tolist(),
      'trace_finite_fraction_all_slots':float(np.isfinite(data['rgb']).mean()),
      'trace_finite_fraction_valid_slots':float(np.isfinite(data['rgb'][data['region_valid']]).mean()) if data['region_valid'].any() else None,
      'small_face_frames':int(data['face_small'].sum()),'crop_valid_frames':int(data['crop_valid'].sum()),
      'landmark_confidence_available':False,'C7_status':'not_implemented_no_per_landmark_confidence',
      'timing_ms':{k:{'mean':float(np.mean(data['timing_ms'][:,j])),'p50':float(np.percentile(data['timing_ms'][:,j],50)),
                  'p95':float(np.percentile(data['timing_ms'][:,j],95))} for j,k in enumerate(TIMING_NAMES)},
      'elapsed_s':time.time()-started,'mediapipe_version':mp.__version__,
      'license':'Apache-2.0','canonical_model_sha256':sha256(Path(__file__).parent/'assets/canonical_face_model.obj'),
      'source_video_sha256':recording.metadata.get('source_video_sha256','NA'),
      'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [],
      'timestamp_policy':'Actual input timestamps unchanged; crop AVI time is only indexed storage, use timestamps_s',
      'coordinate_convention':'x horizontal y vertical pixel coordinates in source RGB; polygons stored float16 after full precision rasterization',
      'config':config or {}}
    manifest['output_bytes']=sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    (out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
