"""Image-only compressed ROI training cache; no test video access."""
import json
import numpy as np
from train_h264_ablation import DATA,CLEAN,verify
from train_efficientphys import sha
from video_roi_frontend import process_video,clip_tensor

OUT=DATA/'detected_on_h264_train500_20261019'
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        verify(OUT);print('PREPARATION ALREADY COMPLETE',flush=True);return
    rows=[r for r in json.loads((CLEAN/'clips.json').read_text()) if r['split']=='train']
    split=json.loads((CLEAN/'subject_split.json').read_text());coverage=[];reports=[]
    for s in split['train']:
        m=np.load(CLEAN/(s+'_meta.npz'));video=DATA/'training_h264_500k_v1'/(s+'.mp4')
        front=process_video(video,m['original_t']);np.testing.assert_array_equal(front['t'],m['t'])
        assert front['report']['decode_complete'],s
        np.save(OUT/(s+'_rgb.npy'),front['rgb'])
        np.savez(OUT/(s+'_meta.npz'),t=m['t'],ppg=m['ppg'],original_t=m['original_t'],boxes=front['boxes'],valid=front['valid'],original_valid=front['original_valid'],fresh=front['fresh'])
        reports.append({'subject':s,'video_sha256':sha(video),**front['report']})
        for r in [r for r in rows if r['subject']==s]:
            arr,bad=clip_tensor(front['rgb'],front['t'],front['valid'],r['start_s'])
            coverage.append({'subject':s,'window':r['window'],'usable':arr is not None,'invalid_samples':bad})
        print('PREPARED',s,flush=True)
    # Identical common support in both experimental arms, fixed before training.
    good={(r['subject'],r['window']) for r in coverage if r['usable']}
    common=[r for r in rows if (r['subject'],r['window']) in good]
    assert len(common)>=2 and {r['subject'] for r in common}==set(split['train'])
    for name,obj in [('clips.json',rows),('subject_split.json',split),('coverage.json',coverage),('frontend_reports.json',reports),('common_train_rows.json',common)]:
        (OUT/name).write_text(json.dumps(obj,indent=2))
    (OUT/'COMPLETE.json').write_text(json.dumps({'train_windows':len(rows),'common_usable':len(common),'excluded':len(rows)-len(common),'test_opened':False},indent=2))
    inventory={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.name!='inventory.json'}
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2));verify(OUT)
    print((OUT/'COMPLETE.json').read_text(),flush=True)
if __name__=='__main__':main()
