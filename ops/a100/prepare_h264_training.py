"""Apply the existing validation H264 recipe to training subjects only."""
import hashlib,json,subprocess
from pathlib import Path
import cv2,imageio_ffmpeg
import numpy as np
R=Path('/data/khangle/facevitals')
BASE=R/'datasets/UBFC-rPPG/efficientphys36_trainval_v1'
OUT=R/'datasets/UBFC-rPPG/efficientphys36_train_h264_500k_v1'
VIDEOS=R/'datasets/UBFC-rPPG/training_h264_500k_v1'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);VIDEOS.mkdir(parents=True,exist_ok=True)
    split=json.loads((BASE/'subject_split.json').read_text())
    rows=[r for r in json.loads((BASE/'clips.json').read_text()) if r['split']=='train']
    assert len(rows)==161 and len(split['train'])==28
    (OUT/'subject_split.json').write_text(json.dumps(split,indent=2))
    (OUT/'clips.json').write_text(json.dumps(rows,indent=2))
    cv2.setNumThreads(1);encoder=imageio_ffmpeg.get_ffmpeg_exe();reports=[]
    version=subprocess.check_output([encoder,'-version'],text=True).splitlines()[0]
    for subject in split['train']:
        original=R/'datasets/UBFC-rPPG/DATASET_2'/subject/'vid.avi';video=VIDEOS/(subject+'.mp4')
        command=[encoder,'-nostdin','-y','-i',str(original),'-map','0:v:0','-an','-c:v','libx264',
                 '-preset','fast','-pix_fmt','yuv420p','-b:v','500k','-maxrate','500k','-bufsize','1000k',
                 '-threads','2','-fps_mode','passthrough',str(video)]
        with (VIDEOS/(subject+'_encoder.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=log,check=True)
        meta=np.load(BASE/(subject+'_meta.npz'));times=meta['original_t'];boxes=meta['boxes']
        cap=cv2.VideoCapture(str(video));assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))];crops=[]
        while True:
            ok,f=cap.read()
            if not ok:break
            x1,y1,x2,y2=boxes[len(crops)]
            crops.append(cv2.cvtColor(cv2.resize(f[y1:y2,x1:x2],(36,36),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
        cap.release();assert len(crops)==len(times)==declared
        unique,inverse,counts=np.unique(times,return_inverse=True,return_counts=True)
        np.testing.assert_array_equal(unique,meta['t'])
        merged=np.zeros((len(unique),36,36,3),np.float32);np.add.at(merged,inverse,np.asarray(crops))
        merged/=counts[:,None,None,None]
        np.save(OUT/(subject+'_rgb.npy'),merged.astype(np.float16))
        np.savez(OUT/(subject+'_meta.npz'),**{k:meta[k] for k in meta.files})
        report={'subject':subject,'frames':len(crops),'size':size,'container_fps':fps,
                'container_average_kbps':video.stat().st_size*8/(len(crops)/fps)/1000,
                'recorded_duration_s':float(times[-1]-times[0]),'video_sha256':sha(video),'command':command}
        reports.append(report);print(json.dumps(report),flush=True)
    (OUT/'encoding_report.json').write_text(json.dumps({'videos':reports,'encoder_version':version,
        'scope':'Training subjects only. Same codec settings, fixed original ROI boxes, recorded timestamps and duplicate merging as validation compression cache.'},indent=2))
    inventory={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(OUT.iterdir()) if f.name!='inventory.json'}
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2));print('COMPLETE',flush=True)
