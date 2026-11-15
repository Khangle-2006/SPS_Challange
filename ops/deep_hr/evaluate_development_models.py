"""Evaluate preselected models on fixed development conditions; never opens test."""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_efficientphys import Clips,evaluate,sha
from stress_ubfc import ROOT,StressClips,CONDITIONS,RUNS
from EfficientPhys_fixed import EfficientPhys

if __name__=='__main__':
    torch.set_num_threads(2)
    out=ROOT/'results/development_comparison_v1';out.mkdir(exist_ok=True,parents=True)
    h264=ROOT/'datasets/UBFC-rPPG/efficientphys36_val_h264_500k_v1'
    for name,m in json.loads((h264/'inventory.json').read_text()).items():
        assert (h264/name).stat().st_size==m['bytes'] and sha(h264/name)==m['sha256']
    val=Clips(h264,'val');assert len(val)==42
    models={f'original_{s}':ROOT/'results'/run for s,run in RUNS.items()}
    lock=json.loads((ROOT/'results/efficientphys36_three_seeds/frozen_protocol.json').read_text())
    for e in lock['checkpoints']:
        assert sha(models[f"original_{e['seed']}"]/'best.pt')==e['checkpoint_sha256']
    ablation=ROOT/'results/compression_ablation_v1'
    for folder in sorted(ablation.glob('*_*')):
        if folder.is_dir() and (folder/'status.json').exists():
            assert json.loads((folder/'status.json').read_text())['state']=='complete'
            assert sha(folder/'best.pt')==json.loads((folder/'status.json').read_text())['checkpoint_sha256']
            models[folder.name]=folder
    summaries=[]
    for label,folder in models.items():
        conditions=['h264_500k'] if label.startswith('original') else CONDITIONS+['h264_500k']
        model=EfficientPhys(frame_depth=10,img_size=36).cuda()
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=False)['model'])
        for c in conditions:
            target=out/f'{label}_{c}.json'
            if target.exists():r=json.loads(target.read_text())
            else:
                ds=val if c=='h264_500k' else StressClips('val',c)
                r=evaluate(model,DataLoader(ds,batch_size=2,num_workers=2),'cuda')
                if not label.startswith('original') and c in ('clean','jpeg90','jpeg50'):
                    expected=json.loads((folder/'best_validation.json').read_text())['conditions'][c]['subject_macro_MAE']
                    assert abs(r['subject_macro_MAE']-expected)<1e-8
                r.update(model=label,condition=c,checkpoint_sha256=sha(folder/'best.pt'))
                r['subjects']={s:sum(x['abs_error'] for x in r['records'] if x['subject']==s)/sum(x['subject']==s for x in r['records']) for s in sorted({x['subject'] for x in r['records']})}
                tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2));tmp.replace(target)
            compact={k:v for k,v in r.items() if k not in ('records','subjects')}
            summaries.append(compact);print(json.dumps(compact),flush=True)
        del model
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    (out/'provenance.json').write_text(json.dumps({'evaluation_code_sha256':sha(Path(__file__)),
        'h264_inventory_sha256':sha(h264/'inventory.json'),'test_opened':False,
        'note':'All results are on development subjects. H264 was not used in augmentation training or checkpoint selection.'},indent=2))
    print('COMPLETE',len(summaries),flush=True)
