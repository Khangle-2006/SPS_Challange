"""Frozen-checkpoint CPU inference and MATLAB v7.3 artifact export; no training."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ops'/'deep_hr')]
from submission.mat_writer import write_mat,verify_mat,file_sha256
from submission.metrics import compute_metrics

CHECKPOINT_SHA256='a173fdacd0105216e2cbe14f30f97d6b24379ed50b4203592e7334c4e5620318'
PIPELINE_VERSION='round1-packaging-v1.1-20261030'


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def read_reference(path):
    data=np.genfromtxt(path,delimiter=',',names=True,encoding='utf-8')
    if not {'timestamps_s','ppg'}<=set(data.dtype.names or []):
        raise ValueError('reference_csv requires timestamps_s,ppg in the video clock')
    return np.atleast_1d(data['timestamps_s']),np.atleast_1d(data['ppg'])


def load_recording(a):
    from data_adapters import UBFC2,MCD,Recording,VideoFrames,Label,video_pts
    rt,rv=read_reference(a.reference_csv)
    if a.dataset in ('ubfc2','mcd'):
        if not a.root or not a.recording:raise ValueError('Dataset adapter requires root and recording')
        if a.dataset=='mcd' and not a.audit_dir:raise ValueError('MCD requires existing V3 receipts')
        backend=UBFC2(a.root) if a.dataset=='ubfc2' else MCD(a.root,a.audit_dir)
        rec=backend.load(a.recording); pp=rec.require_synchronized_ppg()
        if Path(rec.frames.path).resolve()!=Path(a.video).resolve():raise ValueError('Video/adapter identity mismatch')
        if rt.shape!=pp.timestamps.shape or rv.shape!=pp.values.shape or not np.allclose(rt,pp.timestamps,atol=1e-9,rtol=0) or not np.allclose(rv,pp.values,equal_nan=True,atol=1e-9,rtol=0):
            raise ValueError('CSV must be unchanged adapter PPG on original relative clock')
        return rec
    if not a.reference_clock_shared:raise ValueError('Generic input requires explicit --reference-clock-shared; no offset inferred')
    vt=np.load(a.video_timestamps) if a.video_timestamps else video_pts(a.video)
    return Recording(VideoFrames(Path(a.video),len(vt)),vt,{'ppg':Label(rv,rt,'a.u.')},
                     a.subject_id,Path(a.video).stem,{'dataset':'external','operational_alignment_accepted':True,
                     'clock':'explicit shared-clock declaration by caller; not independently verified'})


def roi_data(cache,t,boxes):
    if cache:
        source=Path(cache)/'trace.npz' if Path(cache).is_dir() else Path(cache)
        with np.load(source) as z:
            if len(z['timestamps_s'])!=len(t) or not np.allclose(z['timestamps_s'],t,atol=1e-8,rtol=0):
                raise ValueError('T3 cache timestamp/frame mismatch')
            p=z['polygons'].astype(np.float32)
            return {'roi_type':'landmark_polygons','forehead':p[:,0], 'cheek_left':p[:,1],
                    'cheek_right':p[:,2], 'periocular':p[:,3], 'polygon_lengths':z['polygon_lengths'],
                    'valid':z['region_valid'], 'timestamps_s':t,'coordinate_system':'pixel x,y; source frame',
                    'landmark_confidence':'NA; FaceMesh API does not expose per-landmark confidence',
                    'cache_sha256':file_sha256(source)}
    p=np.stack([boxes[:,[0,1]],boxes[:,[2,1]],boxes[:,[2,3]],boxes[:,[0,3]]],axis=1)
    return {'roi_type':'bbox_placeholder','forehead':p,'cheek_left':p,'cheek_right':p,
            'timestamps_s':t,'warning':'Same Haar inner bbox in all fields; NOT anatomical forehead/cheek polygons'}


def export_target(frames,path,fps,expected_n):
    """Lossless RGB MP4 derivative; explicit timestamps separately preserve sensor clock."""
    import av
    from fractions import Fraction
    digest=hashlib.sha256(); count=0; shape=None
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264rgb',rate=Fraction(str(fps)).limit_denominator(100000))
        stream.pix_fmt='rgb24';stream.options={'crf':'0','preset':'fast','threads':'1'}
        for rgb in frames:
            if shape is None:
                shape=rgb.shape;stream.height,stream.width=shape[:2]
            f=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            for packet in stream.encode(f):out.mux(packet)
            digest.update(rgb.tobytes());count+=1
        for packet in stream.encode():out.mux(packet)
    if count!=expected_n:raise ValueError('Target source frame count mismatch')
    back=hashlib.sha256();n=0
    with av.open(str(path)) as c:
        for f in c.decode(video=0):back.update(f.to_ndarray(format='rgb24').tobytes());n+=1
    if n!=count or back.digest()!=digest.digest():raise ValueError('target_video.mp4 RGB not bit-exact')
    return shape,{'frames':count,'decoded_rgb_sha256':digest.hexdigest(),'roundtrip_bit_exact':True,
                  'codec':'libx264rgb CRF0','clock_note':'CFR presentation; original timestamps in video_timestamps.csv and MAT params'}


def run(a):
    import cv2
    import torch
    import av
    from prepare_recording import prepare
    from video_roi_frontend import clip_tensor
    from EfficientPhys_fixed import EfficientPhys
    from hr_decoder import estimate_hr
    torch.set_num_threads(1);torch.manual_seed(2027);np.random.seed(2027);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(a.out_dir)
    if out.exists():raise FileExistsError('New output directory required: '+str(out))
    out.mkdir(parents=True)
    t0=time.perf_counter(); timings={}
    protocol=Path(a.protocol)
    if not protocol.is_file():raise ValueError('Frozen protocol required before real-data run')
    checkpoint=Path(a.checkpoint or Path(__file__).parent/'checkpoints'/'baseline_seed2027.pt')
    if file_sha256(checkpoint)!=CHECKPOINT_SHA256:raise ValueError('Checkpoint differs from frozen seed2027 baseline')
    constraints=[x for x in a.constraints.split(',') if x]
    if constraints and not a.constraint_manifest:
        raise ValueError('--constraints annotates verified T5 derivative; --constraint-manifest required; no implicit simulation')
    rec=load_recording(a)
    dump(out/'status.json',{'state':'running','stage':'frontend','recording':rec.recording_id})
    ppg=rec.require_synchronized_ppg(); t=np.asarray(rec.timestamps)
    start=time.perf_counter();prepare(rec,out/'haar36_cache');timings['frontend_s']=time.perf_counter()-start
    meta=np.load(out/'haar36_cache'/'meta.npz');raw=np.load(out/'haar36_cache'/'rgb.npy',mmap_mode='r')
    support=json.loads((out/'haar36_cache'/'windows.json').read_text())
    # Include support-rejected windows in output coverage denominator; no silent exclusion.
    rows=sorted(support['valid_windows']+support['rejected_windows'],key=lambda r:r['index'])
    model=EfficientPhys(frame_depth=10,img_size=36).eval()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
    waveforms=[];wave_times=[];wave_window=[];predicted=[];reference=[];abstain=[];results=[]
    start=time.perf_counter()
    for row in rows:
        ws=row['start_s'];hr=np.nan;ref=np.nan;reason='|'.join(row['reasons']);q=ws+np.arange(300)/30
        if not reason:
            clip,_=clip_tensor(raw,meta['t'],meta['valid'],ws)
            if clip is None:reason='invalid_frontend_clip'
            else:
                with torch.inference_mode():diff=model(torch.from_numpy(clip)[None]).numpy()[0]
                wave=np.r_[0.,np.cumsum(diff[:-1])]
                try:hr=estimate_hr(wave,30.)
                except ValueError:reason='invalid_model_waveform'
                if not reason:
                    waveforms.extend(wave);wave_times.extend(q);wave_window.extend([row['index']]*len(q))
            # Reference processing is downstream of all predictions and never changes their peak.
            try:ref=estimate_hr(np.interp(q,meta['ppg_t'],meta['ppg']),30.)
            except ValueError:reason=(reason+'|invalid_reference_waveform').strip('|')
        rejected=bool(reason)
        predicted.append(hr);reference.append(ref);abstain.append(rejected)
        results.append({'window_index':row['index'],'window_start_s':ws,'hr_bpm':float(hr) if np.isfinite(hr) else None,
                        'reference_hr_bpm':float(ref) if np.isfinite(ref) else None,'abstain':rejected,'reason':reason})
    timings['inference_and_reference_hr_s']=time.perf_counter()-start
    metrics=compute_metrics(predicted,reference,[rec.subject_id]*len(rows),abstain)
    rois=roi_data(a.roi_cache,t,meta['boxes'])
    with av.open(a.video) as c:
        stream=c.streams.video[0];nominal=float(stream.average_rate) if stream.average_rate else None
        codec=stream.codec_context.name
    effective=float((len(t)-1)/(t[-1]-t[0]))
    start=time.perf_counter();shape,target=export_target(rec.frames,out/'target_video.mp4',nominal or effective,len(t))
    timings['lossless_target_export_verify_s']=time.perf_counter()-start
    shutil.copyfile(a.reference_csv,out/'reference_signal.csv')
    np.savetxt(out/'video_timestamps.csv',np.column_stack([np.arange(len(t)),t]),delimiter=',',
               header='frame_index,timestamps_s',comments='',fmt=['%d','%.12f'])
    np.save(out/'video_timestamps.npy',t)
    with open(out/'processed_output.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]) if results else ['window_index','window_start_s','hr_bpm','reference_hr_bpm','abstain','reason']);w.writeheader();w.writerows(results)
    with av.open(str(out/'target_video.mp4')) as c:
        packaged_duration=float(c.duration)/av.time_base
    params={'resolution':np.asarray(shape[:2][::-1]),'fps_nominal':nominal if nominal else 'NA','fps_effective':effective,
            'lux':'NA','camera_distance_cm':'NA','fitzpatrick':'NA','age':'NA','ita_estimate':'NA',
            'constraints_applied':constraints or ['None'],'dataset':rec.metadata.get('dataset',a.dataset),
            'subject_id':rec.subject_id,'codec':'h264 RGB lossless packaging','source_codec':codec,
            'bitrate_kbps_measured':(out/'target_video.mp4').stat().st_size*8/packaged_duration*1e-3,
            'bitrate_basis':'packaged target_video.mp4 whole-file bits / its measured container duration; not C5 operating bitrate',
            'pipeline_version':PIPELINE_VERSION,'git_hash':'NA','checkpoint_sha256':CHECKPOINT_SHA256,
            'decoder':'argmax .7-3Hz; frozen existing decoder; no v2 adoption',
            'model_input':'Haar36 unchanged; T3 polygons exported alongside, not fed into checkpoint',
            'video_timestamps_s':t,'label_offset_s':0.,'protocol_sha256':file_sha256(protocol),
            'roi_type':rois['roi_type']}
    if a.constraint_manifest:params['constraint_manifest_json']=Path(a.constraint_manifest).read_text()
    processed={'rppg':np.asarray(waveforms),'timestamps_s':np.asarray(wave_times),'fs_hz':30.,
               'hr_bpm_per_window':np.asarray(predicted),'window_start_s':np.asarray([r['start_s'] for r in rows]),
               'abstain':np.asarray(abstain,bool),'window_index_per_sample':np.asarray(wave_window,dtype=np.int64),
               'waveform_note':'Each10s waveform starts at0; do not interpret seams as a continuous pulse recording'}
    fields={'roi_data':rois,'reference_signal':{'ppg':ppg.values,'timestamps_s':ppg.timestamps,
            'source':str(a.reference_csv),'fs_hz':float((len(ppg.timestamps)-1)/(ppg.timestamps[-1]-ppg.timestamps[0])),
            'valid':ppg.valid_mask},'processed_signal':processed,'metrics':metrics,'params':params}
    dump(out/'status.json',{'state':'running','stage':'mat_export','recording':rec.recording_id})
    mat=out/'Round1_Baseline_None.mat';start=time.perf_counter()
    if a.external_video_reference:
        receipt=write_mat(mat,fields,video_reference=out/'target_video.mp4');check=verify_mat(mat)
    else:
        matshape=(*shape,len(t));receipt=write_mat(mat,fields,frames=iter(rec.frames),shape=matshape)
        check=verify_mat(mat,matshape,receipt['decoded_rgb_sha256'])
        if receipt['decoded_rgb_sha256']!=target['decoded_rgb_sha256']:raise AssertionError('MAT/target RGB mismatch')
    timings['mat_write_full_readback_s']=time.perf_counter()-start
    dump(out/'metrics.json',metrics);dump(out/'window_results.json',results)
    dump(out/'mat_verification.json',dict(receipt,**check));dump(out/'target_verification.json',target)
    source_files=list((ROOT/'submission').glob('*.py'))+[ROOT/'submission'/'requirements.txt',ROOT/'submission'/'SCHEMA.md']+[ROOT/'ops'/'deep_hr'/name for name in
                 ['data_adapters.py','prepare_recording.py','window_support.py','video_roi_frontend.py','hr_decoder.py','EfficientPhys_fixed.py']]
    dump(out/'script_sha256.json',{str(p.relative_to(ROOT)):file_sha256(p) for p in source_files})
    # Each artifact carries its own executable source and frozen checkpoint.
    for src in source_files:
        dst=out/'code'/src.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    shutil.copyfile(Path(__file__).with_name('requirements.txt'),out/'requirements.txt')
    shutil.copyfile(Path(__file__).with_name('SCHEMA.md'),out/'SCHEMA.md')
    frozen=out/'code'/'reports'/'t4_submission_20261030'/'protocol_frozen.json'
    frozen.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(protocol,frozen)
    cp=out/'code'/'submission'/'checkpoints'/'baseline_seed2027.pt';cp.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(checkpoint,cp)
    (out/'process_round1.py').write_text("from pathlib import Path\nimport runpy\nrunpy.run_path(str(Path(__file__).parent/'code'/'submission'/'process_round1.py'),run_name='__main__')\n")
    if a.roi_cache:
        cached=Path(a.roi_cache);cached=cached/'trace.npz' if cached.is_dir() else cached
        shutil.copyfile(cached,out/'roi_trace.npz')
    dump(out/'reproduce_command.json',['python','process_round1.py','--video','target_video.mp4',
         '--reference_csv','reference_signal.csv','--out_dir','reproduced_output','--reference-clock-shared',
         '--video-timestamps','video_timestamps.npy','--subject-id',rec.subject_id]+(['--roi-cache','roi_trace.npz'] if a.roi_cache else []))
    timings['total_s']=time.perf_counter()-t0
    dump(out/'status.json',{'state':'complete','recording':rec.recording_id,'timings':timings,'metrics':metrics,
         'roi_type':rois['roi_type'],'mat_verification':check,'python':platform.python_version(),
         'checkpoint_sha256':CHECKPOINT_SHA256,'external_reference_requested':a.external_video_reference})
    print(json.dumps({'state':'complete','out_dir':str(out),'timings':timings}),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['video','reference_csv','out_dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--constraints',default='');p.add_argument('--constraint-manifest')
    p.add_argument('--dataset',choices=['external','ubfc2','mcd'],default='external')
    p.add_argument('--root');p.add_argument('--recording');p.add_argument('--audit-dir')
    p.add_argument('--subject-id',default='external:unknown');p.add_argument('--reference-clock-shared',action='store_true')
    p.add_argument('--video-timestamps');p.add_argument('--checkpoint');p.add_argument('--roi-cache')
    p.add_argument('--external-video-reference',action='store_true')
    p.add_argument('--protocol',default=str(ROOT/'reports'/'t4_submission_20261030'/'protocol_frozen.json'))
    return p


if __name__=='__main__':
    args=parser().parse_args()
    try:run(args)
    except Exception as error:
        if Path(args.out_dir).is_dir():dump(Path(args.out_dir)/'failure.json',{'error':repr(error),'partial_results_retained':True})
        raise
