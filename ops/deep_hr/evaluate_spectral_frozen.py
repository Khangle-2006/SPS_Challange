"""Freeze twelve checkpoints before evaluating unseen bitrates; no test access."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_h264_ablation import ROOT,CLEAN,COMP_TRAIN,COMP_VAL,SEEDS,Clips,verify,same_pair,EfficientPhys
from train_efficientphys import evaluate,sha
OUT=ROOT/'results/spectral_ablation_v1'
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
frozen={};audit={}
for mode in ['control','augmentation']:
 for seed in SEEDS:
    old=ROOT/'results/h264_ablation_v1'/f'{mode}_{seed}';new=OUT/f'{mode}_{seed}'
    po=json.loads((old/'provenance.json').read_text());pn=json.loads((new/'provenance.json').read_text())
    assert po['initial_model_sha256']==pn['initial_model_sha256']
    co=dict(po['config']);cn=dict(pn['config']);co.pop('loss');cn.pop('loss');assert co==cn
    for name in ['train_efficientphys.py','EfficientPhys_fixed.py','hr_baseline.py']:assert po['code_sha256'][name]==pn['code_sha256'][name]==sha(Path(__file__).parent/name)
    ho=json.loads((old/'history.json').read_text());hn=json.loads((new/'history.json').read_text());assert len(ho)==len(hn)==30
    assert all(x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips'] for x,y in zip(ho,hn))
    audit[f'{mode}_{seed}']={'initialization_equal':True,'all30_sample_orders_equal':True,'all30_augmentation_counts_equal':True,'all_nonloss_config_equal':True,'common_source_equal':True}
    for family,p in [('mse',old),('spectral',new)]:
        status=json.loads((p/'status.json').read_text());assert status['state']=='complete' and status['reload_verified']
        v=json.loads((p/'best_validation.json').read_text());h=json.loads((p/'history.json').read_text())
        assert v['epoch']==min(h,key=lambda r:r['score'])['epoch']
        assert sha(p/'best.pt')==status['checkpoint_sha256']
        frozen[f'{family}_{mode}_{seed}']={'path':str(p/'best.pt'),'sha256':status['checkpoint_sha256'],'best_epoch':v['epoch'],'selection_score':v['score']}
freeze={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checkpoints':frozen,'pairing_audit':audit,
        'protocol_sha256':sha(OUT/'protocol.json'),'rule':'All checkpoints frozen before reading/evaluating 250k or1000k caches; no selection from those rates.'}
fp=OUT/'frozen_checkpoints.json'
if fp.exists():
    oldfreeze=json.loads(fp.read_text());assert oldfreeze['checkpoints']==frozen;freeze=oldfreeze
else:fp.write_text(json.dumps(freeze,indent=2))
roots={'clean':CLEAN,'h264_500k':COMP_VAL,'h264_250k':ROOT/'datasets/UBFC-rPPG/efficientphys36_val_h264_250k_v1','h264_1000k':ROOT/'datasets/UBFC-rPPG/efficientphys36_val_h264_1000k_v1'}
datasets={}
for c,p in roots.items():verify(p);datasets[c]=Clips(p,'val')
for c,d in datasets.items():same_pair(datasets['clean'],d)
loaders={c:DataLoader(d,batch_size=2,num_workers=2) for c,d in datasets.items()}
trainsets={'clean':Clips(CLEAN,'train'),'h264_500k':Clips(COMP_TRAIN,'train')}
verify(COMP_TRAIN);same_pair(trainsets['clean'],trainsets['h264_500k'])
trainloaders={c:DataLoader(d,batch_size=2,num_workers=2) for c,d in trainsets.items()}
def enrich(r):
    e=np.array([x['abs_error'] for x in r['records']]);r.update({'P95':float(np.percentile(e,95,method='linear')),'max_error':float(e.max()),'errors_over10':int((e>10).sum()),'errors_over20':int((e>20).sum())})
    r['by_subject']={s:{'MAE':float(np.mean([x['abs_error'] for x in r['records'] if x['subject']==s])),
        'over10':sum(x['abs_error']>10 for x in r['records'] if x['subject']==s),'over20':sum(x['abs_error']>20 for x in r['records'] if x['subject']==s)} for s in sorted({x['subject'] for x in r['records']})}
    return r
dest=OUT/'frozen_evaluation';dest.mkdir(exist_ok=True)
for name,m in frozen.items():
    p=Path(m['path']);assert sha(p)==m['sha256']
    model=EfficientPhys(frame_depth=10,img_size=36).cuda();model.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model'])
    result={'checkpoint':m,'conditions':{},'train_conditions':{},'test_opened':False,'freeze_sha256':sha(fp)}
    prior=json.loads((p.parent/'best_validation.json').read_text())
    for c,dl in loaders.items():
        report=enrich(evaluate(model,dl,'cuda'));result['conditions'][c]=report
        if c in prior['conditions']:assert abs(report['subject_macro_MAE']-prior['conditions'][c]['subject_macro_MAE'])<1e-8
    if name.startswith('mse_'):
        tr=json.loads((p.parent/'train_diagnostics.json').read_text());assert tr['checkpoint_sha256']==m['sha256']
        result['train_conditions']={c:enrich(r) for c,r in tr['conditions'].items()}
    else:result['train_conditions']={c:enrich(evaluate(model,dl,'cuda')) for c,dl in trainloaders.items()}
    (dest/(name+'.json')).write_text(json.dumps(result,indent=2));print('EVALUATED',name,{c:round(r['subject_macro_MAE'],4) for c,r in result['conditions'].items()},flush=True)
    del model
(OUT/'evaluation_complete.json').write_text(json.dumps({'complete':True,'models':12,'validation_conditions':4,'n_validation_windows':42,'n_train_windows':161,'test_opened':False,'freeze_sha256':sha(fp)},indent=2))
print('ALL FROZEN EVALUATIONS COMPLETE',flush=True)
