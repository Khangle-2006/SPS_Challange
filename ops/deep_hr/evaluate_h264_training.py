"""Post-training diagnostics only: never selects checkpoints or opens test."""
import json
import torch
from torch.utils.data import DataLoader
from train_h264_ablation import ROOT,OUT,CLEAN,COMP_TRAIN,Clips,same_pair,verify,EfficientPhys,SEEDS
from train_efficientphys import evaluate,sha

torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
verify(CLEAN);verify(COMP_TRAIN)
datasets={'clean':Clips(CLEAN,'train'),'h264_500k':Clips(COMP_TRAIN,'train')}
same_pair(datasets['clean'],datasets['h264_500k'])
loaders={c:DataLoader(d,batch_size=2,num_workers=2) for c,d in datasets.items()}
for mode in ['control','augmentation']:
    for seed in SEEDS:
        p=OUT/f'{mode}_{seed}';status=json.loads((p/'status.json').read_text())
        assert status['state']=='complete' and sha(p/'best.pt')==status['checkpoint_sha256']
        model=EfficientPhys(frame_depth=10,img_size=36).cuda()
        model.load_state_dict(torch.load(p/'best.pt',map_location='cuda',weights_only=False)['model'])
        result={'scope':'Train-only diagnostics after validation-based checkpoint selection; no selection changes.',
                'test_opened':False,'checkpoint_sha256':status['checkpoint_sha256'],
                'conditions':{c:evaluate(model,dl,'cuda') for c,dl in loaders.items()}}
        (p/'train_diagnostics.json').write_text(json.dumps(result,indent=2))
        print(mode,seed,{c:round(r['subject_macro_MAE'],4) for c,r in result['conditions'].items()},flush=True)
        del model
