"""Measure native-clock landmark crops and traces. Pilot only: never delete raw."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv, json, hashlib, time
from datetime import datetime
from pathlib import Path
import av, cv2, mediapipe as mp, numpy as np
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
OUT=Path('/mnt/samsung_t5/khangle/facevitals/storage_pilot_20261030')
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(1)
POLYGONS={'forehead':[109,10,338,337,151,108],
          'left_cheek':[50,101,205,187,123],
          'right_cheek':[280,330,425,411,352]}
vertices=sorted(set(v for p in POLYGONS.values() for v in p))
rows=list(csv.DictReader((ROOT/'db.csv').open()))
available=[r for r in rows if (ROOT/r['video']).exists() and (ROOT/(r['video']+'.verified.json')).exists()]
selected=[]
for camera in ['FullHDwebcam','USBVideo','IriunWebcam']:
    matches=sorted((r for r in available if r['camera']==camera),key=lambda r:(r['patient_id'],r['step']))
    if matches:selected.append(matches[0])
reports=[]
for row in selected:
    t0=time.time(); source=ROOT/row['video']; key=source.stem
    lines=(ROOT/row['meta']).read_text().splitlines()
    indices=[]; clocks=[]
    for line in lines:
        if not line.strip():continue
        idx, dt=line.split(maxsplit=1)
        indices.append(int(idx));clocks.append(datetime.fromisoformat(dt))
    rel=np.asarray([(d-clocks[0]).total_seconds() for d in clocks],dtype=np.float64)
    assert np.all(np.diff(rel)>=0), 'Nonmonotonic acquisition metadata'
    assert np.all(np.diff(indices)==1), 'Missing metadata frame index'
    crops=[];traces=[];valid=[];pts=[];coords=[];areas=[];frame_hashes=[]
    # One full recording per camera; retain failed frames explicitly.
    with av.open(str(source)) as container, mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,max_num_faces=1,refine_landmarks=False,
        min_detection_confidence=0.5,min_tracking_confidence=0.5) as mesh:
        stream=container.streams.video[0]
        info={'codec':stream.codec_context.name,'pixel_format':stream.codec_context.format.name,
              'average_rate':str(stream.average_rate),'time_base':str(stream.time_base),
              'width':stream.width,'height':stream.height}
        for i,frame in enumerate(container.decode(video=0)):
            assert i<len(rel), 'More decoded frames than metadata'
            rgb=frame.to_ndarray(format='rgb24');h,w=rgb.shape[:2]
            assert frame.pts is not None
            pts.append(float(frame.pts*frame.time_base))
            frame_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            result=mesh.process(rgb)
            crop=np.zeros((36,36,3),np.uint8);trace=np.full((3,3),np.nan,np.float32)
            xy=np.full((len(vertices),2),np.nan,np.float32);area=np.zeros(3,np.int32);ok=False
            if result.multi_face_landmarks:
                lm=np.array([(p.x*w,p.y*h) for p in result.multi_face_landmarks[0].landmark],np.float32)
                xy=lm[vertices]
                x1,y1=np.floor(lm.min(0)).astype(int);x2,y2=np.ceil(lm.max(0)).astype(int)
                x1,y1=max(x1,0),max(y1,0);x2,y2=min(x2,w),min(y2,h)
                if x2>x1 and y2>y1:
                    crop=cv2.resize(rgb[y1:y2,x1:x2],(36,36),interpolation=cv2.INTER_AREA)
                    for j,poly in enumerate(POLYGONS.values()):
                        mask=np.zeros((h,w),np.uint8)
                        cv2.fillPoly(mask,[np.rint(lm[poly]).astype(np.int32)],1)
                        area[j]=int(mask.sum())
                        if area[j]>=16:trace[j]=rgb[mask.astype(bool)].mean(axis=0)
                    ok=bool(np.isfinite(trace).all())
            crops.append(crop);traces.append(trace);coords.append(xy);areas.append(area);valid.append(ok)
            if (i+1)%1000==0: print(json.dumps({'video':key,'frames':i+1}),flush=True)
    assert len(crops)==len(rel), 'Metadata/decode count mismatch'
    arrays={'crop_rgb_uint8':np.asarray(crops),'trace_rgb_float32':np.asarray(traces),
        'polygon_xy_float32':np.asarray(coords),'polygon_vertex_ids':np.asarray(vertices),
        'roi_pixel_count':np.asarray(areas),'valid':np.asarray(valid),
        'acquisition_seconds':rel,'container_pts_seconds':np.asarray(pts),
        'source_frame_index':np.asarray(indices)}
    path=OUT/(key+'.npz');assert not path.exists(), 'Preserve existing pilot'
    np.savez_compressed(path,**arrays)
    with np.load(path) as check:
        for k,a in arrays.items(): assert np.array_equal(a,check[k],equal_nan=True), k
    (OUT/(key+'.frame_sha256.json')).write_text(json.dumps(frame_hashes))
    (OUT/(key+'.metadata.txt')).write_text('\n'.join(lines)+'\n')
    record={'record':row,'source':str(source),'cache':str(path),'container':info,
        'frames':len(rel),'acquisition_duration':float(rel[-1]-rel[0]),
        'effective_acquisition_fps':(len(rel)-1)/(rel[-1]-rel[0]),
        'duplicate_acquisition_timestamps':int((np.diff(rel)==0).sum()),
        'valid_fraction':float(np.mean(valid)),'raw_bytes':source.stat().st_size,
        'cache_bytes':path.stat().st_size,'uncompressed_array_bytes':sum(a.nbytes for a in arrays.values()),
        'roi_polygons':POLYGONS,'versions':{'av':av.__version__,'opencv':cv2.__version__,'mediapipe':mp.__version__},
        'source_sha256':json.loads((ROOT/(row['video']+'.verified.json')).read_text())['sha256'],
        'source_preserved':True,'ppg_synchronization_validated':False,
        'note':'Pilot frontend, not a validated training cache; RGB remains gamma-encoded, no normalization. Cache cannot recreate full-resolution frontend or compression experiments.',
        'seconds':time.time()-t0}
    reports.append(record)
    (OUT/'mcd_pilot.json').write_text(json.dumps(reports,indent=2));print(json.dumps(record),flush=True)
# Exact metadata counts; projection assumes pilot cache bytes/frame transfers to other people.
total_frames=0
for row in rows:
    total_frames+=sum(bool(line.strip()) for line in (ROOT/row['meta']).open())
summary={'pilot_records':len(reports),'all_recording_metadata_frames':total_frames,
         'crop_uint8_uncompressed_bytes':total_frames*36*36*3,
         'pilot_cache_projection_bytes':total_frames*sum(r['cache_bytes'] for r in reports)/sum(r['frames'] for r in reports),
         'projection_is_estimate':True,'bulk_deletion_enabled':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
