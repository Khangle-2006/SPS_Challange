"""Prove clean frontend equivalence, bundle encoded videos and replay inputs."""
import hashlib,json,os,tarfile
from pathlib import Path
import cv2
import numpy as np
from video_roi_frontend import process_video
R=Path('/data/khangle/facevitals');D=R/'datasets/UBFC-rPPG';BASE=D/'efficientphys36_trainval_v1'
OUT=D/'end_to_end_detection_inputs_20261019'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():raise RuntimeError('Refusing overwrite')
    split=json.loads((BASE/'subject_split.json').read_text());subjects=split['val'];assert len(subjects)==7
    protocol={'subjects':subjects,'rates_kbps':[250,500,1000],'roi_source':'Only current video frame; old Haar parameters and half-fps update cadence.',
       'control':'All7 clean videos must reproduce old boxes and merged float16 RGB exactly; replay detector half-resolution grayscale on P100 before compressed inference.',
       'failure':'No initial face or >4 consecutive scheduled failures marks frame invalid; allow later reacquisition, no fallback to native ROI. Reject any clip using any invalid positive-weight interpolation support. Keep all42 windows and score invalid135bpm.',
       'new_training':False,'test_evaluated':False,'code_sha256':{n:sha(Path(__file__).parent/n) for n in ['prepare_end_to_end_inputs.py','video_roi_frontend.py']},
       'opencv':cv2.__version__,'cascade_sha256':sha(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml')}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));control=[];replay={}
    for s in subjects:
        assert s not in split['test'];m=np.load(BASE/(s+'_meta.npz'))
        result=process_video(D/'DATASET_2'/s/'vid.avi',m['original_t'],capture_grays=True)
        np.testing.assert_array_equal(result['boxes'],m['boxes'])
        np.testing.assert_array_equal(result['rgb'],np.load(BASE/(s+'_rgb.npy'),mmap_mode='r'))
        np.testing.assert_array_equal(result['t'],m['t']);assert result['valid'].all() and result['report']['decode_complete']
        for k in ['gray_frames','gray_indices','frame_dimensions','boxes','fps']:replay[s+'_'+k]=result[k]
        control.append({'subject':s,'boxes_exact':True,'merged_RGB_exact':True,**result['report']})
        print('CLEAN CONTROL VERIFIED',s,flush=True)
    np.savez_compressed(OUT/'clean_detector_replay.npz',**replay)
    (OUT/'clean_control.json').write_text(json.dumps(control,indent=2));videos=[]
    for rate in [250,500,1000]:
        records=json.loads((D/f'efficientphys36_val_h264_{rate}k_v1/encoding_report.json').read_text())['videos']
        target=OUT/f'h264_{rate}k';target.mkdir(exist_ok=True)
        for s in subjects:
            p=D/f'validation_h264_{rate}k_v1'/(s+'.mp4');expected=next(x for x in records if x['subject']==s)
            assert sha(p)==expected['video_sha256'];dest=target/p.name
            if not dest.exists():os.link(p,dest)
            assert sha(dest)==expected['video_sha256']
            videos.append({'rate_kbps':rate,'subject':s,'path':str(dest.relative_to(OUT)),'bytes':dest.stat().st_size,'sha256':expected['video_sha256']})
    (OUT/'videos.json').write_text(json.dumps(videos,indent=2))
    inventory={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
    (OUT/'COMPLETE.json').write_text(json.dumps({'complete':True,'clean_subjects':7,'videos':21,'file_inventory':inventory,'test_evaluated':False},indent=2))
    archive=D/'end_to_end_detection_inputs_20261019.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(OUT,arcname=str(OUT.relative_to(R)))
    manifest={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'complete':True}
    (R/'results/end_to_end_detection_inputs_20261019.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
