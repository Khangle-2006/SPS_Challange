"""Frozen loss-only comparison against matched MSE, no test access."""
import json
import numpy as np
import torch
from train_matched_spectral import ROOT,DATA,CLEAN,OUT,REFERENCE,SEEDS,verify,EfficientPhys,Clips,sha
from evaluate_spectral_ensemble import make_reports,forward_wave

def read(p):return json.loads(p.read_text())
def folder(family):return REFERENCE if family=='mse' else OUT
def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    dest=OUT/'evaluation';dest.mkdir(exist_ok=True);frozen={}
    for seed in SEEDS:
        pair=[]
        for family in ['mse','spectral']:
            p=folder(family)/f'augmentation_{seed}';st=read(p/'status.json');assert st['state']=='complete' and st['completed_epochs']==30 and st['reload_verified']
            assert sha(p/'best.pt')==st['checkpoint_sha256']
            pair.append((read(p/'provenance.json'),read(p/'history.json')))
            frozen[f'{family}_{seed}']={'path':str(p/'best.pt'),'sha256':st['checkpoint_sha256'],'epoch':st['best_epoch']}
        (a,ha),(b,hb)=pair
        assert a['initial_model_sha256']==b['initial_model_sha256']
        assert {k:v for k,v in a['config'].items() if k!='loss'}=={k:v for k,v in b['config'].items() if k!='loss'}
        assert a['train_h264_inventory']==b['train_h264_inventory']
        assert len(ha)==len(hb)==30
        for x,y in zip(ha,hb):assert x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips']
    (OUT/'frozen_checkpoints.json').write_text(json.dumps(frozen,indent=2))
    roots={'clean':CLEAN,**{f'h264_{r}k':DATA/f'detected_on_h264_20261019/h264_{r}k' for r in [500,250,1000]}}
    for root in roots.values():verify(root)
    rows=Clips(CLEAN,'val').rows;assert len(rows)==42
    prior=read(REFERENCE.parent/'results.json')['native'];results={}
    for family in ['mse','spectral']:
        models=[]
        for seed in SEEDS:
            model=EfficientPhys(frame_depth=10,img_size=36).cuda().eval();model.load_state_dict(torch.load(frozen[f'{family}_{seed}']['path'],map_location='cuda',weights_only=False)['model']);models.append(model)
        results[family]={}
        for c,root in roots.items():
            ds=Clips(root,'val');assert ds.rows==rows
            waves=np.full((3,len(rows),300),np.nan,np.float32)
            with torch.no_grad():
                for i in range(len(ds)):
                    if i in ds.invalid:continue
                    x=ds[i][0][None].cuda()
                    for j,model in enumerate(models):waves[j,i]=forward_wave(model,x)[0].cpu().numpy()
            reports,powers=make_reports(waves,rows);results[family][c]=reports
            np.savez(dest/f'{family}_{c}_waves.npz',waves=waves,normalized_psd=powers)
            for method in reports:
                if family=='mse':
                    for x,y in zip(reports[method]['records'],prior[c][method]['records']):assert x['pred_hr']==y['pred_hr'] and x['abs_error']==y['abs_error']
                if method.isdigit() and c in ['clean','h264_500k']:
                    selected=read(folder(family)/f'augmentation_{method}/best_validation.json')['conditions'][c]
                    assert reports[method]['subject_macro_MAE']==selected['subject_macro_MAE']
            print('EVALUATED',family,c,flush=True)
        del models;torch.cuda.empty_cache()
    summary={family:{c:{'MAE_mean':float(np.mean([rr[str(s)]['subject_macro_MAE'] for s in SEEDS])),
        'MAE_sd':float(np.std([rr[str(s)]['subject_macro_MAE'] for s in SEEDS],ddof=1)),
        'P95_mean':float(np.mean([rr[str(s)]['P95'] for s in SEEDS])),
        'over20_sum':sum(rr[str(s)]['errors_over20'] for s in SEEDS)} for c,rr in group.items()} for family,group in results.items()}
    a,b=summary['mse'],summary['spectral'];score=lambda d:(d['clean']['MAE_mean']+d['h264_500k']['MAE_mean'])/2
    checks={'score_gain_at_least_10pct':score(b)<=.9*score(a),'clean_increase_at_most_025':b['clean']['MAE_mean']<=a['clean']['MAE_mean']+.25,
        'h264_improved_at_least_2_seeds':sum(results['spectral']['h264_500k'][str(s)]['subject_macro_MAE']<results['mse']['h264_500k'][str(s)]['subject_macro_MAE'] for s in SEEDS)>=2,
        'h264_P95_increase_at_most_1':b['h264_500k']['P95_mean']<=a['h264_500k']['P95_mean']+1,
        'h264_over20_not_increased':b['h264_500k']['over20_sum']<=a['h264_500k']['over20_sum']}
    (OUT/'results.json').write_text(json.dumps(results,indent=2));(OUT/'comparison.json').write_text(json.dumps({'summary':summary,'checks':checks,'keep_gate_passed':all(checks.values()),'paired_verified':True,'mse_reference_reproduced':True,'test_opened':False},indent=2))
    (OUT/'COMPLETE.json').write_text(json.dumps({'fresh_runs':3,'epochs_each':30,'paired_verified':True,'mse_reference_reproduced':True,'keep_gate_passed':all(checks.values()),'test_opened':False},indent=2))
    print(json.dumps({'summary':summary,'checks':checks},indent=2),flush=True)
if __name__=='__main__':main()
