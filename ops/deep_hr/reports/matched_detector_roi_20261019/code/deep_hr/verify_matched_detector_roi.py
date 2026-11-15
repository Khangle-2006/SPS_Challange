"""Check matched sample support and exact frontend/training interpolation agreement."""
import json
import numpy as np
from train_matched_detector_roi import ValidClips,NEW,VAL,base,same_pair
from video_roi_frontend import clip_tensor
def main():
    report={'train':{},'validation':{},'test_opened':False}
    clean=ValidClips(base.CLEAN,'train')
    for name,root in [('native',base.COMP_TRAIN),('detected',NEW)]:
        ds=ValidClips(root,'train');same_pair(clean,ds)
        assert not ds.invalid
        for i,r in enumerate(ds.rows):
            x,y,_=ds[i];cx,cy,_=clean[i];assert np.isfinite(x.numpy()).all();np.testing.assert_array_equal(y.numpy(),cy.numpy())
            m=np.load(root/(r['subject']+'_meta.npz'));raw=np.load(root/(r['subject']+'_rgb.npy'),mmap_mode='r')
            valid=m['valid'] if 'valid' in m else np.ones(len(m['t']),bool)
            expected,bad=clip_tensor(raw,m['t'],valid,r['start_s']);assert expected is not None and bad==0
            np.testing.assert_array_equal(x.numpy(),expected)
        report['train'][name]={'windows':len(ds),'all_labels_paired':True,'all_interpolated_pixels_exact':True}
    vc=ValidClips(base.CLEAN,'val');vh=ValidClips(VAL,'val');same_pair(vc,vh);assert len(vc)==len(vh)==42
    report['validation']={'windows':42,'unusable_compressed':len(vh.invalid),'same_validation_for_both_arms':True}
    p=base.ROOT/'results/matched_detector_roi_20261019';p.mkdir(exist_ok=True)
    (p/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
