"""Compressed-image detection vs native-image ROI; frozen HR predictors, no test."""
import hashlib,json,os,platform,subprocess,time
from pathlib import Path
import cv2
import numpy as np
import torch
from train_h264_ablation import ROOT,CLEAN,SEEDS,EfficientPhys,verify
from train_efficientphys import sha
from evaluate_spectral_ensemble import make_reports,forward_wave
from spectral_ensemble import normalized_psd,summarize_three
from video_roi_frontend import Tracker,process_video,clip_tensor
from verify_video_roi_frontend import run as verify_math
OUT=ROOT/'results/end_to_end_detection_20261019';D=ROOT/'datasets/UBFC-rPPG';INPUT=D/'end_to_end_detection_inputs_20261019';OLD=ROOT/'results/spectral_ensemble_20261019'
METHODS=[str(s) for s in SEEDS]+['hr_median','psd_mean'];CONDS=['h264_250k','h264_500k','h264_1000k']
def read(p):return json.loads(p.read_text())
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2))
def replay_clean():
    z=np.load(INPUT/'clean_detector_replay.npz');info=[]
    for s in read(CLEAN/'subject_split.json')['val']:
        tr=Tracker(float(z[s+'_fps']));indices=z[s+'_gray_indices'];images=z[s+'_gray_frames'];expected=z[s+'_boxes'];w,h=map(int,z[s+'_frame_dimensions']);k=0
        native=np.load(CLEAN/(s+'_meta.npz'));np.testing.assert_array_equal(expected,native['boxes'])
        for i,b in enumerate(expected):
            gray=None
            if tr.needs_detection(i):
                assert k<len(indices) and indices[k]==i
                gray=images[k];k+=1
            valid,actual,fresh=tr.step(i,w,h,gray);assert valid;np.testing.assert_array_equal(actual,b)
        assert k==len(indices);info.append({'subject':s,'frames':len(expected),'detector_attempts':k,'all_boxes_exact':True})
    return info
def agreement(old,new,valid):
    if not valid.any():return {'usable_frames':0,'mean_IoU_to_native':None,'p05_IoU_to_native':None,'different_boxes_fraction':None}
    a=old[valid];b=new[valid];inter=np.prod(np.maximum(0,np.minimum(a[:,2:],b[:,2:])-np.maximum(a[:,:2],b[:,:2])),axis=1)
    iou=inter/(np.prod(a[:,2:]-a[:,:2],axis=1)+np.prod(b[:,2:]-b[:,:2],axis=1)-inter)
    return {'usable_frames':int(valid.sum()),'mean_IoU_to_native':float(iou.mean()),'p05_IoU_to_native':float(np.percentile(iou,5)),
      'different_boxes_fraction':float(np.mean(np.any(a!=b,axis=1))),
      'max_coordinate_step_px':int(np.abs(np.diff(b,axis=0)).max()) if len(b)>1 else 0}
def native_pixel_control(path,times,boxes,cached):
    """Separate reference-only decode: excludes cross-host codec/resize differences."""
    cap=cv2.VideoCapture(str(path));assert cap.isOpened();crops=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert len(crops)<len(times);x1,y1,x2,y2=boxes[len(crops)]
        crops.append(cv2.cvtColor(cv2.resize(frame[y1:y2,x1:x2],(36,36),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
    cap.release();assert len(crops)==len(times)
    unique,inverse,counts=np.unique(times,return_inverse=True,return_counts=True)
    merged=np.zeros((len(unique),36,36,3),np.float32);np.add.at(merged,inverse,np.array(crops));merged/=counts[:,None,None,None]
    merged=merged.astype(np.float16);np.testing.assert_array_equal(merged,cached)
    return merged
def benchmark(path,times,rows,models,config,expected):
    """Direct wall clock: full-record decode/detect/crop followed by batch1 inference."""
    family,method=config;torch.cuda.synchronize();start=time.perf_counter();front=process_video(path,times);frontend_done=time.perf_counter();preds=[];invalid=0
    with torch.no_grad():
        for row in rows:
            arr,bad=clip_tensor(front['rgb'],front['t'],front['valid'],row['start_s'])
            if arr is None:preds.append(None);invalid+=1;continue
            x=torch.from_numpy(arr)[None].cuda()
            selected=[models[family][2029]] if method=='2029' else list(models[family].values())
            waves=np.stack([forward_wave(model,x)[0].cpu().numpy() for model in selected])
            try:
                if method=='2029':f,p=normalized_psd(waves[0]);hr=float(f[np.argmax(p)])
                else:hr=summarize_three(waves)['psd_mean_hr']
            except ValueError:hr=None;invalid+=1
            preds.append(hr)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    for hr,row in zip(preds,rows):assert hr==expected[(row['subject'],row['window'])]['pred_hr']
    return {'family':family,'method':method,'subject':rows[0]['subject'],'wall_s':elapsed,'frontend_wall_s':frontend_done-start,
        'resampling_forward_postprocess_s':elapsed-(frontend_done-start),'recorded_duration_s':float(times[-1]-times[0]),'windows':len(rows),'invalid_windows':invalid,
        'processed_faster_than_recorded_duration':bool(elapsed<(times[-1]-times[0])),'all_predictions_match_cache_pass':True}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():raise RuntimeError('Refusing overwrite of completed experiment')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;cv2.setNumThreads(1)
    source=read(INPUT/'COMPLETE.json');assert source['complete'] and source['videos']==21
    for n,m in source['file_inventory'].items():assert (INPUT/n).stat().st_size==m['bytes'] and sha(INPUT/n)==m['sha256']
    videos=read(INPUT/'videos.json')
    for v in videos:assert sha(INPUT/v['path'])==v['sha256']
    oldprotocol=read(OLD/'protocol.json');oldresults=read(OLD/'results.json');checkpoints=oldprotocol['checkpoints']
    assert sha(OLD/'protocol.json')==oldresults['protocol_sha256'] and oldresults['all_single_predictions_reproduced']
    assert sha(Path(__file__).parent/'spectral_ensemble.py')==oldprotocol['code_sha256']['spectral_ensemble.py']
    for group in checkpoints.values():
        for m in group.values():assert sha(Path(m['path']))==m['sha256']
    protocol={'checkpoints':checkpoints,'conditions':CONDS,'subjects':read(CLEAN/'subject_split.json')['val'],'windows_per_condition':42,
       'comparison':'Same encoded files; native-image ROI vs current compressed-image detection, both recomputed batch1. Native-ROI HRs must exactly reproduce all prior ensemble/single results. Same recorded timestamps and HR labels.',
       'frontend':'Original half-size Haar frontal detector scaleFactor1.1/minNeighbors5/minSize30; cadence max(1,round(input_container_fps/2)); same nearest-face tracking and inner crop; no EMA.',
       'invalid_policy':'No native-box fallback, pixel interpolation across unusable ROI, or dropped windows. Any invalid positive-weight temporal support rejects clip for every model, penalty135bpm. >4 scheduled misses invalid until reacquired.',
       'clean_control':'A100 must reproduce all7 clean native boxes and merged RGB; P100 must replay every clean detector input and reproduce every box. Separate native-ROI recrop of all21 compressed videos on P100 must reproduce old float16 pixels exactly.',
       'timing_configs':[['mse','2029'],['mse','psd_mean'],['spectral','psd_mean']],
       'timing':'Direct same-P100 full-record wall clock for ALL21 videos per config, one warm file-cache pass each; rotate config order across videos. Includes full decode, detect/crop, duplicate merge, temporal interpolation/normalization, H2D, batch1 forward and HR. Full-record offline, not streaming first-output latency. No network/encoding/checkpoint loading/camera acquisition.',
       'scope':'Frozen development evaluation on7 previously used validation subjects; no training, checkpoint selection, weighting/ROI tuning or test evaluation.',
       'opencv':cv2.__version__,'torch':torch.__version__,'python':platform.python_version(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
       'gpu':torch.cuda.get_device_name(),'torch_threads':torch.get_num_threads(),'cv_threads':cv2.getNumThreads(),
       'cascade_sha256':sha(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'),
       'input_protocol_sha256':sha(INPUT/'protocol.json'),'old_ensemble_protocol_sha256':sha(OLD/'protocol.json'),
       'code_sha256':{n:sha(Path(__file__).parent/n) for n in ['evaluate_end_to_end_detection.py','video_roi_frontend.py','verify_video_roi_frontend.py','spectral_ensemble.py','evaluate_spectral_ensemble.py','EfficientPhys_fixed.py','train_efficientphys.py']}}
    pp=OUT/'protocol.json'
    if pp.exists():assert read(pp)==protocol
    else:write('protocol.json',protocol)
    input_protocol=read(INPUT/'protocol.json')
    assert protocol['cascade_sha256']==input_protocol['cascade_sha256']
    assert sha(Path(__file__).parent/'video_roi_frontend.py')==input_protocol['code_sha256']['video_roi_frontend.py']
    unit=verify_math();write('frontend_tests.json',unit);write('clean_control_replay.json',replay_clean());print('P100 CLEAN REPLAY AND FAILURE POLICY VERIFIED',flush=True)
    oldrows=[r for r in read(CLEAN/'clips.json') if r['split']=='val'];assert len(oldrows)==42
    subjects=protocol['subjects'];allwaves={c:{f:np.full((3,42,300),np.nan,np.float32) for f in checkpoints} for c in CONDS}
    nativewaves={c:{f:np.full((3,42,300),np.nan,np.float32) for f in checkpoints} for c in CONDS}
    coverage={};front_reports={};pixel_controls=[];mapping={(r['subject'],r['window']):i for i,r in enumerate(oldrows)}
    models={}
    for family,group in checkpoints.items():
        models[family]={}
        for seed in SEEDS:
            model=EfficientPhys(frame_depth=10,img_size=36).cuda().eval();model.load_state_dict(torch.load(group[str(seed)]['path'],map_location='cuda',weights_only=False)['model']);models[family][seed]=model
    cache_root=D/'detected_on_h264_20261019';cache_root.mkdir(exist_ok=True)
    old_sources=read(OLD/'source_verification.json')
    for c in CONDS:
        rate=int(c.split('_')[1][:-1]);out=cache_root/c;out.mkdir(exist_ok=True);coverage[c]=[];front_reports[c]=[]
        native_cache=D/f'efficientphys36_val_h264_{rate}k_v1'
        assert sha(native_cache/'inventory.json')==old_sources['cache_inventory_sha256'][c];verify(native_cache)
        for s in subjects:
            m=np.load(CLEAN/(s+'_meta.npz'));path=INPUT/f'h264_{rate}k'/(s+'.mp4');front=process_video(path,m['original_t'])
            native_rgb=native_pixel_control(path,m['original_t'],m['boxes'],np.load(native_cache/(s+'_rgb.npy'),mmap_mode='r'))
            pixel_controls.append({'subject':s,'condition':c,'same_host_native_pixels_exact':True});write('native_pixel_controls.json',pixel_controls)
            np.testing.assert_array_equal(front['t'],m['t']);comp=agreement(m['boxes'],front['boxes'],front['original_valid'])
            front_reports[c].append({'subject':s,**front['report'],'ROI_agreement_diagnostic':comp})
            np.save(out/(s+'_rgb.npy'),front['rgb']);np.savez(out/(s+'_meta.npz'),t=m['t'],ppg=m['ppg'],original_t=m['original_t'],boxes=front['boxes'],valid=front['valid'],original_valid=front['original_valid'],fresh=front['fresh'])
            rows=[r for r in oldrows if r['subject']==s]
            with torch.no_grad():
                for row in rows:
                    arr,bad=clip_tensor(front['rgb'],front['t'],front['valid'],row['start_s']);ok=arr is not None;i=mapping[(s,row['window'])]
                    coverage[c].append({'subject':s,'window':row['window'],'start_s':row['start_s'],'usable':ok,'invalid_resampled_samples':bad,
                       'reason':None if ok else ('unusable_ROI_support' if bad else 'flat_or_nonfinite_input')})
                    native_arr,_=clip_tensor(native_rgb,m['t'],np.ones(len(m['t']),bool),row['start_s']);assert native_arr is not None
                    xn=torch.from_numpy(native_arr)[None].cuda()
                    for family,group in models.items():
                        for j,seed in enumerate(SEEDS):nativewaves[c][family][j,i]=forward_wave(group[seed],xn)[0].cpu().numpy()
                    if not ok:continue
                    x=torch.from_numpy(arr)[None].cuda()
                    for family,group in models.items():
                        for j,seed in enumerate(SEEDS):allwaves[c][family][j,i]=forward_wave(group[seed],x)[0].cpu().numpy()
            print('PREPARED AND INFERRED',c,s,flush=True)
        (out/'clips.json').write_text(json.dumps(oldrows,indent=2));(out/'subject_split.json').write_text(json.dumps(read(CLEAN/'subject_split.json'),indent=2))
        inventory={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.iterdir() if p.name!='inventory.json'}
        (out/'inventory.json').write_text(json.dumps(inventory,indent=2))
    results={};dest=OUT/'waveforms';dest.mkdir(exist_ok=True)
    for family in models:
        results[family]={}
        for c in CONDS:
            reports,powers=make_reports(allwaves[c][family],oldrows);native,npowers=make_reports(nativewaves[c][family],oldrows)
            for method in METHODS:
                prior=oldresults['groups'][family]['validation'][c][method]
                for row,old in zip(native[method]['records'],prior['records']):
                    assert row['subject']==old['subject'] and row['window']==old['window'] and row['gt_hr']==old['gt_hr'] and row['pred_hr']==old['pred_hr']
            results[family][c]={'native_ROI':native,'detected_ROI':reports}
            np.savez_compressed(dest/f'{family}_{c}.npz',waves=allwaves[c][family],normalized_psd=powers,native_waves=nativewaves[c][family],native_normalized_psd=npowers)
    write('results.json',{'groups':results,'protocol_sha256':sha(pp),'test_evaluated':False});write('coverage.json',coverage);write('frontend_reports.json',front_reports)
    timing=[];configs=[('mse','2029'),('mse','psd_mean'),('spectral','psd_mean')]
    # Actual full pipeline, separate fresh decode/detection per timing configuration.
    for n,v in enumerate(videos):
        s=v['subject'];c=f"h264_{v['rate_kbps']}k";times=np.load(CLEAN/(s+'_meta.npz'))['original_t'];rows=[r for r in oldrows if r['subject']==s]
        ordered=configs[n%3:]+configs[:n%3]
        for config in ordered:
            family,method=config;expected={(r['subject'],r['window']):r for r in results[family][c]['detected_ROI'][method]['records']}
            measured=benchmark(INPUT/v['path'],times,rows,models,config,expected);measured['condition']=c;timing.append(measured)
        write('timing.json',timing);print('END TO END TIMED',c,s,flush=True)
    write('COMPLETE.json',{'complete':True,'frozen_models':6,'methods_per_group':5,'conditions':3,'windows_per_condition':42,'timed_video_config_runs':63,'clean_control_subjects':7,'test_evaluated':False})
    write('environment.json',{'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu,driver_version','--format=csv'],text=True)})
    print('ALL END TO END DETECTION EVALUATIONS COMPLETE',flush=True)
if __name__=='__main__':main()
