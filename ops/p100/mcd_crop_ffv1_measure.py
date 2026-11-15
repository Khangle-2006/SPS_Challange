"""Pixel-exact storage experiment; never associates ambiguous acquisition rows."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import csv,json,time,hashlib,shutil
from pathlib import Path
from fractions import Fraction
import av,cv2,mediapipe as mp,numpy as np
ROOT=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
OUT=ROOT.parent.parent/'mapping_audit_20261030/crop_storage';OUT.mkdir(parents=True,exist_ok=True)
cv2.setNumThreads(1)
POLYS={'forehead':[109,10,338,337,151,108], 'left_cheek':[50,101,205,187,123], 'right_cheek':[280,330,425,411,352]}
vids=sorted(set(i for p in POLYS.values() for i in p))
reports=[]
for row in csv.DictReader((ROOT/'db.csv').open()):
    if row['patient_id']!='1020' or row['step']!='after':continue
    src=ROOT/row['video'];folder=OUT/src.stem;folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():reports.append(json.loads((folder/'result.json').read_text()));continue
    dst=folder/'crop36.ffv1.nut';assert not dst.exists()
    now=time.time(); hashes=[];pts=[];trace=[];areas=[];coords=[];valid=[];corrupt=[];crop_array=[]
    with av.open(str(src)) as inp, av.open(str(dst),'w',format='nut') as enc, mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,max_num_faces=1,refine_landmarks=False,min_detection_confidence=.5,min_tracking_confidence=.5) as mesh:
        ss=inp.streams.video[0];ds=enc.add_stream('ffv1',rate=ss.average_rate)
        ds.width=36;ds.height=36;ds.pix_fmt='bgr0';ds.time_base=ss.time_base
        ds.codec_context.time_base=ss.time_base;ds.codec_context.thread_count=1
        ds.options={'level':'3','coder':'1','context':'1','slicecrc':'1'}
        for i,f in enumerate(inp.decode(video=0)):
            rgb=f.to_ndarray(format='rgb24');h,w=rgb.shape[:2];result=mesh.process(rgb)
            cr=np.zeros((36,36,3),np.uint8);tr=np.full((3,3),np.nan,np.float32);ar=np.zeros(3,np.int32)
            xy=np.full((len(vids),2),np.nan,np.float32);ok=False
            if result.multi_face_landmarks:
                lm=np.asarray([(p.x*w,p.y*h) for p in result.multi_face_landmarks[0].landmark],np.float32)
                xy=lm[vids];low=np.floor(lm.min(0)).astype(int);high=np.ceil(lm.max(0)).astype(int)
                x1,y1=max(0,low[0]),max(0,low[1]);x2,y2=min(w,high[0]),min(h,high[1])
                if x2>x1 and y2>y1:
                    cr=cv2.resize(rgb[y1:y2,x1:x2],(36,36),interpolation=cv2.INTER_AREA)
                    for j,p in enumerate(POLYS.values()):
                        mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(lm[p]).astype(np.int32)],1)
                        ar[j]=int(mask.sum())
                        if ar[j]>=16:tr[j]=rgb[mask.astype(bool)].mean(axis=0)
                    ok=bool(np.isfinite(tr).all())
            is_corrupt=bool(f.is_corrupt)
            hashes.append(hashlib.sha256(cr.tobytes()).hexdigest());pts.append(str(f.pts*f.time_base))
            trace.append(tr);areas.append(ar);coords.append(xy);valid.append(ok and not is_corrupt);corrupt.append(is_corrupt);crop_array.append(cr)
            of=av.VideoFrame.from_ndarray(cr,format='rgb24');of.pts=f.pts;of.time_base=f.time_base
            for packet in ds.encode(of):enc.mux(packet)
            if (i+1)%1000==0:print(json.dumps({'video':src.stem,'frames':i+1}),flush=True)
        for packet in ds.encode():enc.mux(packet)
    with av.open(str(dst)) as check:
        count=0
        for i,f in enumerate(check.decode(video=0)):
            assert i<len(hashes)
            assert hashlib.sha256(f.to_ndarray(format='rgb24').tobytes()).hexdigest()==hashes[i],('RGB',i)
            assert f.pts*f.time_base==Fraction(pts[i]),('PTS',i)
            count+=1
    assert count==len(hashes)
    np.savez_compressed(folder/'crop36.npz',rgb=np.asarray(crop_array))
    with np.load(folder/'crop36.npz') as npz:
        assert np.array_equal(np.asarray(crop_array),npz['rgb'])
    arrays={'trace_rgb':np.asarray(trace),'roi_area':np.asarray(areas),'polygon_xy':np.asarray(coords),
        'vertex_ids':np.asarray(vids),'valid':np.asarray(valid),'decoder_corrupt':np.asarray(corrupt)}
    np.savez_compressed(folder/'traces_geometry.npz',**arrays)
    with np.load(folder/'traces_geometry.npz') as z:
        for k,v in arrays.items():assert np.array_equal(v,z[k],equal_nan=True)
    (folder/'frame_index_pts_checksums.json').write_text(json.dumps([{'decoded_index':i,'container_pts_rational':p,'crop_sha256':h} for i,(p,h) in enumerate(zip(pts,hashes))]))
    shutil.copy2(ROOT/row['meta'],folder/'unmapped_acquisition_metadata.txt')
    result={'video':row['video'],'camera':row['camera'],'frames':count,'crop_raw_bytes':count*36*36*3,
        'ffv1_bytes':dst.stat().st_size,'npz_bytes':(folder/'crop36.npz').stat().st_size,
        'traces_geometry_npz_bytes':(folder/'traces_geometry.npz').stat().st_size,
        'valid_fraction':float(np.mean(valid)),'corrupt_frames':sum(corrupt),'exact_rgb_and_pts':True,
        'all_crop_bytes_saved_before_validation_mask':True,'polygons':POLYS,'source_raw_preserved':True,
        'training_eligible':False,'note':'Storage-only; acquisition clock mapping unresolved, no PPG association. Pilot dynamic FaceMesh bbox; not frozen E3 frontend.',
        'versions':{'mediapipe':mp.__version__,'av':av.__version__,'cv2':cv2.__version__},'elapsed_s':time.time()-now}
    (folder/'result.json').write_text(json.dumps(result,indent=2));reports.append(result)
    (OUT/'results.json').write_text(json.dumps(reports,indent=2));print(json.dumps(result),flush=True)
total_frames=json.loads((ROOT.parent.parent/'storage_pilot_20261030/summary.json').read_text())['metadata_nonempty_total']
summary={'pilot_videos':len(reports),'bytes_ratio_raw_over_ffv1':sum(r['crop_raw_bytes'] for r in reports)/sum(r['ffv1_bytes'] for r in reports),
    'projected_crop_ffv1_bytes':total_frames*sum(r['ffv1_bytes'] for r in reports)/sum(r['frames'] for r in reports),
    'projected_trace_rgb_float32_only_bytes':total_frames*3*3*4,
    'projection_only':'One person, three cameras; not guaranteed dataset total. Metadata count differs from decoded counts.',
    'raw_deletion_enabled':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
