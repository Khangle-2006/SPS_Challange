"""Actual-detector secondary caches for old train subjects, without touching primary views."""
import json
import numpy as np
import cv2
from cv_common import DATA,CLEAN,WORK,OUT,read,write,sha,CVClips
from video_roi_frontend import process_video
def main():
    split=read(CLEAN/'subject_split.json');subjects=split['train'];pool=read(WORK/'folds.json')['pool_subjects']
    input_root=DATA/'cv_secondary_inputs_20261020';manifests={};coverage={};reports=[]
    for rate in [250,1000]:
        c=f'h264_{rate}k';cache=DATA/f'cv_detected_{rate}k_20261020';cache.mkdir(exist_ok=True)
        source_report=read(input_root/c/'encoding_report.json');video_hash={x['subject']:x['sha256'] for x in source_report['videos']}
        for s in subjects:
            m=np.load(CLEAN/(s+'_meta.npz'));path=input_root/c/(s+'.mp4');assert sha(path)==video_hash[s]
            front=process_video(path,m['original_t']);assert front['report']['decode_complete'];np.testing.assert_array_equal(front['t'],m['t'])
            np.save(cache/(s+'_rgb.npy'),front['rgb']);np.savez(cache/(s+'_meta.npz'),t=m['t'],ppg=m['ppg'],original_t=m['original_t'],boxes=front['boxes'],valid=front['valid'],original_valid=front['original_valid'],fresh=front['fresh'])
            duration=len(m['original_t'])/front['report']['container_fps']
            reports.append({'condition':c,'subject':s,'video_sha256':video_hash[s],'bytes':path.stat().st_size,'container_average_kbps':path.stat().st_size*8/duration/1000,'recorded_duration_average_kbps':path.stat().st_size*8/(m['original_t'][-1]-m['original_t'][0])/1000,**front['report']})
            print('PREPARED',rate,s,flush=True)
        view=WORK/'cache'/f'detected_{rate}k';view.mkdir(exist_ok=True);manifests[c]={}
        for s in pool:
            root=cache if s in subjects else DATA/f'detected_on_h264_20261019/h264_{rate}k';manifests[c][s]={}
            a=np.load(CLEAN/(s+'_meta.npz'));b=np.load(root/(s+'_meta.npz'))
            for key in ['t','ppg','original_t']:np.testing.assert_array_equal(a[key],b[key])
            for suffix in ['_rgb.npy','_meta.npz']:
                p=root/(s+suffix);dest=view/p.name
                if s not in subjects:
                    expected=read(root/'inventory.json')[p.name]
                    assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256']
                if dest.exists():assert dest.resolve()==p.resolve()
                else:dest.symlink_to(p)
                manifests[c][s][suffix]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
        rows=read(WORK/'folds.json')['rows'];ds=CVClips(view,rows);coverage[c]={'windows':len(ds),'unusable':len(ds.invalid)}
    write(WORK/'secondary_manifest.json',{'conditions':manifests,'coverage':coverage,'cascade_sha256':sha(__import__('pathlib').Path(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')),'frontend_code_sha256':sha(__import__('pathlib').Path(__file__).parent/'video_roi_frontend.py'),'test_opened':False})
    write(OUT/'secondary_frontend_reports.json',reports);write(OUT/'SECONDARY_COMPLETE.json',{'coverage':coverage,'secondary_manifest_sha256':sha(WORK/'secondary_manifest.json'),'test_opened':False})
    print(json.dumps(coverage),flush=True)
if __name__=='__main__':main()
