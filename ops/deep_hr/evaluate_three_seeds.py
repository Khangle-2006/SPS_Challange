"""Freeze three validation-selected checkpoints, then evaluate an untouched split."""
import argparse, datetime, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_efficientphys import Clips, evaluate, sha
from EfficientPhys_fixed import EfficientPhys
ROOT=Path('/data_pool/khangle/facevitals')
OUT=ROOT/'results/efficientphys36_three_seeds'
RUNS={2027:'efficientphys36_ubfc_v1',2028:'efficientphys36_ubfc_seed2028',2029:'efficientphys36_ubfc_seed2029'}

class Heldout(Clips):
    def __init__(self,root):
        self.root=root;self.cache={}
        split=json.loads((root/'subject_split.json').read_text())
        self.rows=json.loads((root/'clips.json').read_text())
        assert len(self.rows)==40
        assert {r['subject'] for r in self.rows}==set(split['test'])
        assert all(r['split']=='test' and r['subject'] not in split['train']+split['val'] for r in self.rows)

def freeze():
    path=OUT/'frozen_protocol.json'
    entries=[];base_cfg=None;source_hashes=None;data_hash=None
    for seed,name in RUNS.items():
        d=ROOT/'results'/name
        status=json.loads((d/'status.json').read_text())
        assert status['state']=='complete' and status['completed_epochs']==30
        prov=json.loads((d/'provenance.json').read_text())
        cfg={k:v for k,v in prov['config'].items() if k not in ('seed','run_name')}
        if base_cfg is None:base_cfg=cfg;source_hashes=prov['code_sha256'];data_hash=prov['data_inventory_sha256']
        assert cfg==base_cfg and prov['code_sha256']==source_hashes and prov['data_inventory_sha256']==data_hash
        best=json.loads((d/'best_validation.json').read_text())
        entries.append({'seed':seed,'run':name,'best_epoch':best['epoch'],'checkpoint_sha256':sha(d/'best.pt'),
                        'validation_MAE':best['subject_macro_MAE']})
    for f,digest in source_hashes.items():assert sha(Path(__file__).parent/f)==digest
    lock={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_without_seed':base_cfg,
          'training_code_sha256':source_hashes,'evaluation_code_sha256':sha(Path(__file__)),
          'trainval_inventory_sha256':data_hash,'checkpoints':entries,
          'selection':'best validation subject-macro HR MAE independently for each seed',
          'test_rule':'evaluate all three; no test-based seed selection, ensemble, or parameter change'}
    if path.exists():
        previous=json.loads(path.read_text());assert previous['checkpoints']==entries;return previous
    path.write_text(json.dumps(lock,indent=2));return lock

def summarize(rows):
    errors=np.array([r['abs_error'] for r in rows]); subjects=sorted({r['subject'] for r in rows})
    return {'window_MAE':float(errors.mean()),'window_RMSE':float(np.sqrt(np.mean(errors**2))),
            'median_absolute_error':float(np.median(errors)),
            'subject_macro_MAE':float(np.mean([np.mean([r['abs_error'] for r in rows if r['subject']==s]) for s in subjects])),
            'n':len(rows),'subjects':len(subjects),'invalid':sum(r.get('invalid',False) for r in rows)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],required=True);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/(a.split+'_results.json')
    if target.exists():raise RuntimeError('Results already exist; refusing to overwrite evaluation')
    if a.split=='val': lock=freeze()
    else:
        assert (OUT/'val_results.json').exists(), 'Validate/freeze before test'
        lock=json.loads((OUT/'frozen_protocol.json').read_text())
        assert sha(Path(__file__))==lock['evaluation_code_sha256'], 'Evaluation code changed after freeze'
        (OUT/'test_evaluation_started.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'frozen_protocol_sha256':sha(OUT/'frozen_protocol.json')},indent=2))
    data_root=ROOT/'datasets/UBFC-rPPG'/('efficientphys36_trainval_v1' if a.split=='val' else 'efficientphys36_test_v1')
    for filename,meta in json.loads((data_root/'inventory.json').read_text()).items():
        assert (data_root/filename).stat().st_size==meta['bytes'] and sha(data_root/filename)==meta['sha256'],filename
    data=Clips(data_root,'val') if a.split=='val' else Heldout(data_root)
    torch.set_num_threads(2)
    records=[]
    for entry in lock['checkpoints']:
        path=ROOT/'results'/entry['run']/'best.pt'
        assert sha(path)==entry['checkpoint_sha256']
        ck=torch.load(path,map_location='cuda',weights_only=False)
        assert ck['epoch']+1==entry['best_epoch']
        model=EfficientPhys(frame_depth=10,img_size=36).cuda();model.load_state_dict(ck['model'],strict=True)
        result=evaluate(model,DataLoader(data,batch_size=2,num_workers=2),'cuda')
        if a.split=='val':assert abs(result['subject_macro_MAE']-entry['validation_MAE'])<1e-8
        report={'seed':entry['seed'],'best_epoch':entry['best_epoch'],**summarize(result['records']), 'records':result['records']}
        records.append(report);print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
        del model,ck
    pos_rows=[dict(r,abs_error=abs(r['pos_hr']-r['gt_hr']),invalid=False) for r in records[0]['records']]
    aggregate={}
    for metric in ('window_MAE','window_RMSE','subject_macro_MAE'):
        values=[r[metric] for r in records]
        aggregate[metric]={'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)),'min':min(values),'max':max(values)}
    summary={'split':a.split,'seeds':records,'pos':summarize(pos_rows),'aggregate':aggregate,
             'note':'Seed spread measures training variability on this one subject split, not a population confidence interval.',
             'frozen_protocol_sha256':sha(OUT/'frozen_protocol.json'),'input_inventory_sha256':sha(data_root/'inventory.json')}
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(summary,indent=2));tmp.replace(target)
    print(json.dumps({'split':a.split,'pos':summary['pos'],'aggregate':aggregate},indent=2),flush=True)
