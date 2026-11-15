"""Paired MSE training, common real-detector validation and common train support."""
import argparse,json,sys
import numpy as np
import torch
import train_h264_ablation as base
from train_efficientphys import Clips as OriginalClips
from video_roi_frontend import clip_tensor

EXP=base.ROOT/'results/matched_detector_roi_20261019'
NEW=base.DATA/'detected_on_h264_train500_20261019'
VAL=base.DATA/'detected_on_h264_20261019/h264_500k'

class ValidClips(OriginalClips):
    def __init__(self,root,tag):
        super().__init__(root,tag)
        if tag=='train':
            common=json.loads((NEW/'common_train_rows.json').read_text())
            keep={(r['subject'],r['window']) for r in common}
            self.rows=[r for r in self.rows if (r['subject'],r['window']) in keep]
            assert self.rows==common
        self.invalid=set()
        for i,r in enumerate(self.rows):
            m=np.load(root/(r['subject']+'_meta.npz'))
            if 'valid' in m:
                t=m['t'];q=r['start_s']+np.arange(301)/30
                j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1)
                w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
                if not np.all(((w>=1)|m['valid'][j-1])&((w<=0)|m['valid'][j])):self.invalid.add(i)
        if tag=='train':assert not self.invalid
    def __getitem__(self,i):
        if i in self.invalid:
            return torch.zeros(301,3,36,36),torch.zeros(300),i
        return super().__getitem__(i)

def same_pair(a,b):
    assert a.rows==b.rows
    for s in {r['subject'] for r in a.rows}:
        x=np.load(a.root/(s+'_meta.npz'));y=np.load(b.root/(s+'_meta.npz'))
        for k in ('t','ppg','original_t'):np.testing.assert_array_equal(x[k],y[k])

original_evaluate=base.evaluate
def evaluate(model,loader,device):
    result=original_evaluate(model,loader,device)
    for i,r in enumerate(result['records']):
        if i in loader.dataset.invalid:r.update(pred_hr=None,abs_error=135.,invalid=True)
    records=result['records'];e=np.array([r['abs_error'] for r in records])
    result.update(subject_macro_MAE=float(np.mean([np.mean([r['abs_error'] for r in records if r['subject']==s]) for s in sorted({r['subject'] for r in records})])),
        window_MAE=float(e.mean()),window_RMSE=float(np.sqrt((e**2).mean())),invalid=sum(r.get('invalid',False) for r in records),
        P95=float(np.percentile(e,95)),errors_over20=int((e>20).sum()))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['native','detected'],required=True);p.add_argument('--seed',type=int,choices=base.SEEDS,required=True);a=p.parse_args()
    EXP.mkdir(exist_ok=True)
    protocol={'seeds':base.SEEDS,'arms':['native','detected'],'epochs':30,'model':'EfficientPhys36','loss':'MSE',
        'train_mix':'50% clean / 50% H264500, paired Bernoulli and shuffle RNG',
        'difference':'H264 training ROI: native image detector vs compressed image detector; clean unchanged',
        'validation':'same clean and actual compressed detector H264500 for BOTH arms; equal macro MAE; earliest minimum',
        'common_train_support':json.loads((NEW/'COMPLETE.json').read_text()),
        'common_train_rows_sha256':base.sha(NEW/'common_train_rows.json'),
        'actual_train_inventory_sha256':base.sha(NEW/'inventory.json'),
        'actual_validation_inventory_sha256':base.sha(VAL/'inventory.json'),
        'keep_gate':{'mean_selection_gain_min':.10,'clean_mean_increase_max':.25,'h264_improved_seeds_min':2,'h264_mean_P95_increase_max':1.,'h264_over20_increase_max':0},
        'secondary':'250/1000 seen development conditions; single seed, HR median, equal normalized PSD fixed; no per-rate decoder selection',
        'test_opened':False,'code_sha256':{n:base.sha(base.Path(__file__).parent/n) for n in ['train_matched_detector_roi.py','prepare_matched_roi_training.py','video_roi_frontend.py','train_h264_ablation.py']}}
    pp=EXP/'matched_protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,indent=2))
    base.Clips=ValidClips;base.same_pair=same_pair;base.evaluate=evaluate
    base.COMP_TRAIN=base.COMP_TRAIN if a.arm=='native' else NEW
    base.COMP_VAL=VAL;base.OUT=EXP/a.arm
    base.OUT.mkdir(exist_ok=True)
    sys.argv=[sys.argv[0],'--mode','augmentation','--seed',str(a.seed)]
    base.main()
if __name__=='__main__':main()
