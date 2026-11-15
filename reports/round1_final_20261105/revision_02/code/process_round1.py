"""Versioned oval36/canonical decoder package; historical submission remains unchanged."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,ast,csv,hashlib,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'vendor/v3'),str(HERE/'vendor/oval'),str(HERE/'vendor')]
from spectral import snr
from t8_plan import read,write,sha
from t8_evaluate import decode_difference,reference_row,csv_write
from polarity_loader import load_canonical_clip
from ppg_polarity import adapt_cached_meta
from score_round1 import score,reference_hr
from submission.mat_writer import write_mat,verify_mat
from submission.process_round1 import export_target,roi_data
from submission.metrics import compute_metrics

def run(a):
    import torch,cv2,av
    from EfficientPhys_fixed import EfficientPhys
    from data_adapters import MCD,UBFC2
    from prepare_recording import prepare
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    t0=time.perf_counter();out=a.out;assert not out.exists();out.mkdir(parents=True)
    timings={};write(out/'status.json',dict(state='running',stage='frontend'))
    idx=read(a.index);record=next(r for r in idx['records'] if r['dataset']==a.dataset and r['recording_id']==a.recording)
    pref='mcd' if a.dataset=='MCD' else 'ubfc';fold=next(f for f in idx['folds']['folds'] if record['subject_id'] in f[pref+'_val_subjects'])
    assert record['subject_id'] not in fold[pref+'_train_subjects']
    assert a.branch in ('ubfc_only','v3_joint','hybrid_ubfc_only','hybrid_v3_joint')
    expected=read(a.checkpoint_manifest)
    assert expected['seed']==2027 and expected['fold']==fold['fold']
    assert sha(a.checkpoint)==expected['checkpoint_sha256']
    backend=MCD(a.data_root,a.audit) if a.dataset=='MCD' else UBFC2(a.data_root)
    rec=backend.load(a.recording);start=time.perf_counter()
    prepare(rec,out/'oval36_cache')
    timings['frontend_s']=time.perf_counter()-start
    frozen=Path(idx['roots'][a.dataset])/a.recording/'native/crop'
    # Frozen index may contain P100 paths; the caller supplies the local original cache.
    if a.frozen_cache is not None:frozen=a.frozen_cache
    raw=np.load(out/'oval36_cache/rgb.npy',mmap_mode='r');old=np.load(frozen/'rgb.npy',mmap_mode='r')
    assert raw.shape==old.shape and np.array_equal(raw,old),'New frontend differs from frozen oval pixels'
    with np.load(out/'oval36_cache/meta.npz') as z:
        meta={k:z[k].copy() for k in z.files}
    with np.load(frozen/'meta.npz') as z:
        for k in ('t','valid','ppg_t','ppg','ppg_valid'):assert np.array_equal(meta[k],z[k],equal_nan=True),'Clock/cache changed '+k
    canonical,normalization=adapt_cached_meta(meta,a.dataset,source_convention='raw_device')
    write(out/'FRONTEND_PARITY.json',dict(all_pixels_exact=True,clocks_support_exact=True,raw_shape=list(raw.shape),normalization=normalization,source_cache_rgb_sha256=sha(frozen/'rgb.npy')))
    model=EfficientPhys(frame_depth=10,img_size=36).eval();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
    ns=dict(np=np,math=math);src=HERE/'vendor/hr_baseline.py';tree=ast.parse(src.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pos');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'),ns)
    refs=[];preds=[];waveforms=[];wave_times=[];wave_indices=[];chosen=[];starts=[];hrs=[];refhrs=[];abstain=[]
    start=time.perf_counter()
    for wi,w in enumerate(record['candidates']):
        ws=float(w['start_s']);q=ws+np.arange(300)/30
        ref=reference_row(record,w,canonical,'native');refs.append(ref)
        row=dict(subject_id=ref['subject_id'],recording_id=ref['recording_id'],window_start_s=ws,hr_bpm='',abstain=True)
        selected='abstain';wave=None;hr=None
        usable=ws in record['source_valid_starts'] and record['evaluation_usable']['native'].get(str(int(ws)) if ws.is_integer() else str(ws),False)
        # Authoritative frozen key format is determined by t8_plan.window_key.
        from t8_plan import window_key
        usable=window_key(w) in record['source_valid_starts'] and record['evaluation_usable']['native'][str(window_key(w))]
        if usable:
            x,_,_,_=load_canonical_clip(out/'oval36_cache',ws,a.dataset)
            with torch.inference_mode():difference=model(torch.from_numpy(x[None])).numpy()[0]
            wave=np.cumsum(difference.astype(float));hr=decode_difference(difference);selected='model'
            if a.branch.startswith('hybrid'):
                # Same spatial-mean raw RGB and source clock as OOF POS, no PPG in selector.
                j=np.searchsorted(meta['t'],q,side='right').clip(1,len(meta['t'])-1)
                weight=(q-meta['t'][j-1])/(meta['t'][j]-meta['t'][j-1])
                mean0=np.asarray(raw[j-1],float).mean(axis=(1,2));mean1=np.asarray(raw[j],float).mean(axis=(1,2))
                rgb=mean0*(1-weight[:,None])+mean1*weight[:,None]
                pw=ns['pos'](rgb,30);ms,ps=snr(wave),snr(pw)
                if ps is not None and (ms is None or ps>ms):
                    wave=pw;hr=reference_hr(dict(window_start_s=0.,ppg_timestamps_json=json.dumps((np.arange(301)/30).tolist()),ppg_values_json=json.dumps(np.r_[pw,pw[-1]].tolist())));selected='POS'
            row.update(hr_bpm=hr,abstain=False)
            waveforms.extend(wave);wave_times.extend(q);wave_indices.extend([wi]*300)
        preds.append(row);starts.append(ws);hrs.append(np.nan if hr is None else hr);chosen.append(selected)
        try:refhrs.append(reference_hr(ref))
        except ValueError:refhrs.append(np.nan)
        abstain.append(row['abstain'])
    timings['inference_and_reference_s']=time.perf_counter()-start
    csv_write(out/'reference_windows.csv',refs);csv_write(out/'processed_output.csv',preds)
    scored=score(out/'reference_windows.csv',out/'processed_output.csv');write(out/'canonical_metrics.json',scored)
    expected_windows=read(a.oof_windows)
    same=[r for r in expected_windows if r['branch']==a.branch and r['seed']==2027 and r['recording_id']==a.recording and r['condition']==('MCD native' if a.dataset=='MCD' else 'UBFC clean')]
    expected_by_start={r['window_start_s']:r for r in same};deltas=[]
    for r in scored['windows']:
        e=expected_by_start[r['window_start_s']]
        assert r['accepted']==e['accepted'] and r['reference_hr_bpm']==e['reference_hr_bpm']
        if r['accepted']:deltas.append(abs(r['prediction_hr_bpm']-e['prediction_hr_bpm']))
    assert not deltas or max(deltas)<=.5,'CPU deployment HR parity exceeds preregistered0.5bpm bin tolerance'
    write(out/'OOF_PARITY.json',dict(max_HR_delta_bpm=max(deltas) if deltas else None,tolerance_bpm=.5,coverage_exact=True,reference_exact=True))
    t=np.asarray(rec.timestamps);rois=roi_data(a.roi_trace,t,meta['boxes']);assert rois['roi_type']=='landmark_polygons'
    with av.open(str(rec.frames.path)) as c:
        stream=c.streams.video[0];nominal=float(stream.average_rate);source_codec=stream.codec_context.name
    start=time.perf_counter();shape,target=export_target(rec.frames,out/'target_video.mp4',nominal,len(t));timings['target_export_s']=time.perf_counter()-start
    fields=dict(roi_data=rois,reference_signal=dict(ppg=canonical['ppg'],timestamps_s=canonical['ppg_t'],valid=canonical['ppg_valid'],polarity_convention=normalization.get('convention','blood_volume_positive_v1'),hr_bpm_per_window=np.asarray(refhrs),window_start_s=np.asarray(starts)),processed_signal=dict(rppg=np.asarray(waveforms),timestamps_s=np.asarray(wave_times),window_index_per_sample=np.asarray(wave_indices),hr_bpm_per_window=np.asarray(hrs),window_start_s=np.asarray(starts),abstain=np.asarray(abstain),fs_hz=30.,waveform_note='Independent10s windows; joins are not continuous physiological waveform',selected_source=chosen),metrics=compute_metrics(hrs,refhrs,[rec.subject_id]*len(starts),abstain),params=dict(resolution=np.array(shape[:2][::-1]),fps_nominal=nominal,fps_effective=float((len(t)-1)/(t[-1]-t[0])),lux='NA',camera_distance_cm='NA',fitzpatrick='NA',age=str(rec.metadata.get('source_db_row',{}).get('age','NA')),constraints_applied=['None'],source_codec=source_codec,model_input='frozen unified36point oval tight bbox INTER_AREA36',roi_type='T3 landmark polygons',checkpoint_sha256=sha(a.checkpoint),branch=a.branch,seed=2027,fold=fold['fold'],video_timestamps_s=t,label_offset_s=0.,decoder='canonical0.7-3Hz lambda100 HannFFT4096',pipeline_version='round1-final-revision02-20261105'))
    start=time.perf_counter();mat=out/'Round1_Baseline_None.mat';receipt=write_mat(mat,fields,frames=iter(rec.frames),shape=(*shape,len(t)));check=verify_mat(mat,(*shape,len(t)),receipt['decoded_rgb_sha256']);assert receipt['decoded_rgb_sha256']==target['decoded_rgb_sha256'];timings['mat_write_readback_s']=time.perf_counter()-start
    np.savetxt(out/'reference_signal.csv',np.column_stack([canonical['ppg_t'],canonical['ppg']]),delimiter=',',header='timestamps_s,ppg',comments='')
    np.savetxt(out/'reference_hr.csv',np.column_stack([starts,refhrs]),delimiter=',',header='window_start_s,hr_bpm',comments='')
    np.save(out/'video_timestamps.npy',t);write(out/'mat_verification.json',dict(receipt,**check));write(out/'target_verification.json',target)
    timings['total_s']=time.perf_counter()-t0
    write(out/'FIELD_LAYOUT.json',dict(facial_video=dict(dtype='uint8',shape=[*shape,len(t)]),roi_data={k:dict(dtype=str(v.dtype),shape=list(v.shape)) for k,v in rois.items() if isinstance(v,np.ndarray)},reference_signal=dict(ppg_samples=len(canonical['ppg']),timestamps_samples=len(canonical['ppg_t'])),processed_signal=dict(samples=len(waveforms),windows=len(starts)),metrics=fields['metrics'],params=fields['params']|{'resolution':list(shape[:2][::-1]),'video_timestamps_s':'stored vector'}))
    write(out/'status.json',dict(state='complete',dataset=a.dataset,recording=a.recording,branch=a.branch,timings=timings,mat_verification=check,checkpoint_sha256=sha(a.checkpoint),oof_parity_max_bpm=max(deltas) if deltas else None))
    print(json.dumps(dict(state='complete',recording=a.recording,timings=timings)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('out','index','data-root','checkpoint','checkpoint-manifest','roi-trace','oof-windows'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--frozen-cache',type=Path);p.add_argument('--audit',type=Path);p.add_argument('--dataset',choices=['MCD','UBFC2'],required=True);p.add_argument('--recording',required=True);p.add_argument('--branch',required=True)
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        if a.out.exists():write(a.out/'FAILURE.json',dict(error=repr(exc),partial_preserved=True))
        raise
