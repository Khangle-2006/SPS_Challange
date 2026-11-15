"""Frozen paired evaluation, all fixed decoders; train/validation only."""
import json
from pathlib import Path
import numpy as np
import torch
from train_matched_detector_roi import EXP,NEW,VAL,ValidClips,base
from evaluate_spectral_ensemble import make_reports,forward_wave
from video_roi_frontend import clip_tensor

def read(p):return json.loads(p.read_text())
def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    dest=EXP/'evaluation';dest.mkdir(exist_ok=True)
    frozen={};paired={}
    for seed in base.SEEDS:
        paired[str(seed)]={}
        for arm in ('native','detected'):
            p=EXP/arm/f'augmentation_{seed}';st=read(p/'status.json');assert st['state']=='complete' and st['reload_verified']
            assert base.sha(p/'best.pt')==st['checkpoint_sha256']
            paired[str(seed)][arm]={'provenance':read(p/'provenance.json'),'history':read(p/'history.json')}
            frozen[f'{arm}_{seed}']={'path':str(p/'best.pt'),'sha256':st['checkpoint_sha256'],'epoch':st['best_epoch']}
        a,b=[paired[str(seed)][arm] for arm in ('native','detected')]
        assert a['provenance']['initial_model_sha256']==b['provenance']['initial_model_sha256']
        assert a['provenance']['config']==b['provenance']['config']
        assert len(a['history'])==len(b['history'])==30
        for x,y in zip(a['history'],b['history']):
            assert x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips']
    (EXP/'frozen_checkpoints.json').write_text(json.dumps(frozen,indent=2))
    roots={'clean':base.CLEAN,**{f'h264_{r}k':base.DATA/f'detected_on_h264_20261019/h264_{r}k' for r in (500,250,1000)}}
    for p in roots.values():base.verify(p)
    rows=ValidClips(base.CLEAN,'val').rows;results={}
    for arm in ('native','detected'):
        models=[]
        for seed in base.SEEDS:
            m=base.EfficientPhys(frame_depth=10,img_size=36).cuda().eval()
            m.load_state_dict(torch.load(frozen[f'{arm}_{seed}']['path'],map_location='cuda',weights_only=False)['model']);models.append(m)
        results[arm]={}
        for c,root in roots.items():
            ds=ValidClips(root,'val');assert ds.rows==rows
            waves=np.full((3,len(rows),300),np.nan,np.float32)
            with torch.no_grad():
                for i in range(len(ds)):
                    if i in ds.invalid:continue
                    x=ds[i][0][None].cuda()
                    for j,m in enumerate(models):waves[j,i]=forward_wave(m,x)[0].cpu().numpy()
            reports,_=make_reports(waves,rows);results[arm][c]=reports
            np.savez(dest/f'{arm}_{c}_waves.npz',waves=waves)
            for seed in base.SEEDS:
                if c in ('clean','h264_500k'):
                    selected=read(EXP/arm/f'augmentation_{seed}/best_validation.json')['conditions'][c]
                    assert reports[str(seed)]['subject_macro_MAE']==selected['subject_macro_MAE']
            print('EVALUATED',arm,c,flush=True)
        del models;torch.cuda.empty_cache()
    summary={}
    for arm in results:
        summary[arm]={c:{'MAE_mean':float(np.mean([rr[str(s)]['subject_macro_MAE'] for s in base.SEEDS])),
            'MAE_sd':float(np.std([rr[str(s)]['subject_macro_MAE'] for s in base.SEEDS],ddof=1)),
            'P95_mean':float(np.mean([rr[str(s)]['P95'] for s in base.SEEDS])),
            'over20_sum':sum(rr[str(s)]['errors_over20'] for s in base.SEEDS)} for c,rr in results[arm].items()}
    a,b=summary['native'],summary['detected'];score=lambda d:(d['clean']['MAE_mean']+d['h264_500k']['MAE_mean'])/2
    checks={'score_gain_at_least_10pct':score(b)<=.9*score(a),'clean_increase_at_most_025':b['clean']['MAE_mean']<=a['clean']['MAE_mean']+.25,
        'h264_improved_at_least_2_seeds':sum(results['detected']['h264_500k'][str(s)]['subject_macro_MAE']<results['native']['h264_500k'][str(s)]['subject_macro_MAE'] for s in base.SEEDS)>=2,
        'h264_P95_increase_at_most_1':b['h264_500k']['P95_mean']<=a['h264_500k']['P95_mean']+1,
        'h264_over20_not_increased':b['h264_500k']['over20_sum']<=a['h264_500k']['over20_sum']}
    (EXP/'results.json').write_text(json.dumps(results,indent=2))
    (EXP/'comparison.json').write_text(json.dumps({'summary':summary,'checks':checks,'keep_gate_passed':all(checks.values()),'paired_initialization_config_order_augmentation_verified':True,'test_opened':False},indent=2))
    changes={}
    for c in roots:
        changes[c]={}
        for method in [str(s) for s in base.SEEDS]+['hr_median','psd_mean']:
            pairs=[]
            for x,y in zip(results['native'][c][method]['records'],results['detected'][c][method]['records']):
                assert (x['subject'],x['window'])==(y['subject'],y['window'])
                pairs.append({'subject':x['subject'],'window':x['window'],'gt_hr':x['gt_hr'],'native':x,'detected':y,'error_delta':y['abs_error']-x['abs_error']})
            changes[c][method]=sorted(pairs,key=lambda r:r['error_delta'],reverse=True)
    (EXP/'window_changes.json').write_text(json.dumps(changes,indent=2))
    (EXP/'COMPLETE.json').write_text(json.dumps({'runs':6,'epochs_each':30,'paired_verified':True,'keep_gate_passed':all(checks.values()),'test_opened':False},indent=2))
    print(json.dumps({'summary':summary,'checks':checks},indent=2),flush=True)
if __name__=='__main__':main()
