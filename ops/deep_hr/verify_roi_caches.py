"""Prove the ROI change preserves every label, timestamp, split and clip."""
import json
import numpy as np
import train_roi_ablation as roi
b=roi.base
pairs=[(b.CLEAN,b.DATA/'efficientphys36_trainval_v1'),(b.COMP_TRAIN,b.DATA/'efficientphys36_train_h264_500k_v1'),(b.COMP_VAL,b.DATA/'efficientphys36_val_h264_500k_v1')]
pairs.extend((b.DATA/f'roi_ema010_val{rate}_v1',b.DATA/f'efficientphys36_val_h264_{rate}k_v1') for rate in [250,1000])
reports=[]
for new,old in pairs:
    b.verify(new);b.verify(old)
    for name in ['clips.json','subject_split.json']:assert json.loads((new/name).read_text())==json.loads((old/name).read_text())
    subjects={r['subject'] for r in json.loads((new/'clips.json').read_text())}
    changed=0
    for s in subjects:
        a=np.load(new/(s+'_meta.npz'));o=np.load(old/(s+'_meta.npz'))
        assert a.files==o.files
        for key in a.files:
            if key!='boxes':np.testing.assert_array_equal(a[key],o[key])
        changed+=int(np.any(a['boxes']!=o['boxes']))
        assert np.load(new/(s+'_rgb.npy'),mmap_mode='r').shape==np.load(old/(s+'_rgb.npy'),mmap_mode='r').shape
    reports.append({'new_cache':str(new),'subjects':len(subjects),'subjects_with_changed_boxes':changed,'labels_timestamps_split_clips_exact':True})
b.OUT.mkdir(parents=True,exist_ok=True)
(b.OUT/'cache_pair_verification.json').write_text(json.dumps(reports,indent=2))
print('ROI CACHE PAIR VERIFICATION PASSED',flush=True)
