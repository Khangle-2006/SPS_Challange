"""Paired fixed-protocol ROI evaluation, never test."""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import train_roi_ablation as roi
from train_efficientphys import evaluate,sha
b=roi.base;out=b.OUT;torch.set_num_threads(2)
def enrich(r):
    e=np.array([x['abs_error'] for x in r['records']])
    r.update(P95=float(np.percentile(e,95)),errors_over20=int((e>20).sum()),errors_over10=int((e>10).sum()))
    return r
frozen={}
for seed in b.SEEDS:
    old=b.ROOT/'results/h264_ablation_v1'/f'augmentation_{seed}';new=out/f'augmentation_{seed}'
    po=json.loads((old/'provenance.json').read_text());pn=json.loads((new/'provenance.json').read_text())
    assert po['config']==pn['config'] and po['initial_model_sha256']==pn['initial_model_sha256']
    assert po['code_sha256']==pn['code_sha256']
    ho=json.loads((old/'history.json').read_text());hn=json.loads((new/'history.json').read_text())
    assert len(ho)==len(hn)==30
    assert all(x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips'] for x,y in zip(ho,hn))
    status=json.loads((new/'status.json').read_text());assert status['state']=='complete' and status['reload_verified']
    assert sha(new/'best.pt')==status['checkpoint_sha256']
    frozen[str(seed)]={'sha256':sha(new/'best.pt'),'best_epoch':status['best_epoch']}
(out/'frozen_checkpoints.json').write_text(json.dumps(frozen,indent=2))
roots={'clean':b.CLEAN,'h264_500k':b.COMP_VAL,'h264_250k':b.DATA/'roi_ema010_val250_v1','h264_1000k':b.DATA/'roi_ema010_val1000_v1'}
datasets={c:b.Clips(p,'val') for c,p in roots.items()}
for c,p in roots.items():b.verify(p);b.same_pair(datasets['clean'],datasets[c])
loaders={c:DataLoader(d,batch_size=2,num_workers=2) for c,d in datasets.items()}
trainloaders={c:DataLoader(b.Clips(p,'train'),batch_size=2,num_workers=2) for c,p in [('clean',b.CLEAN),('h264_500k',b.COMP_TRAIN)]}
results={};dest=out/'evaluation';dest.mkdir(exist_ok=True)
for seed in b.SEEDS:
    p=out/f'augmentation_{seed}'/'best.pt';model=b.EfficientPhys(frame_depth=10,img_size=36).cuda()
    model.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model'])
    result={'checkpoint':frozen[str(seed)],'conditions':{c:enrich(evaluate(model,dl,'cuda')) for c,dl in loaders.items()},'train_conditions':{c:enrich(evaluate(model,dl,'cuda')) for c,dl in trainloaders.items()},'test_opened':False}
    selected=json.loads((p.parent/'best_validation.json').read_text())
    for c in ['clean','h264_500k']:assert result['conditions'][c]['subject_macro_MAE']==selected['conditions'][c]['subject_macro_MAE']
    results[str(seed)]=result;(dest/f'{seed}.json').write_text(json.dumps(result,indent=2));print('EVALUATED',seed,flush=True)
old={str(s):json.loads((b.ROOT/'results/spectral_ablation_v1/frozen_evaluation'/f'mse_augmentation_{s}.json').read_text()) for s in b.SEEDS}
summary={}
for family,data in [('reference',old),('smoothed_roi',results)]:
    summary[family]={c:{'MAE_mean':float(np.mean([r['conditions'][c]['subject_macro_MAE'] for r in data.values()])),
       'MAE_sd':float(np.std([r['conditions'][c]['subject_macro_MAE'] for r in data.values()],ddof=1)),
       'P95_mean':float(np.mean([r['conditions'][c]['P95'] for r in data.values()])),
       'over20_sum':sum(r['conditions'][c]['errors_over20'] for r in data.values())} for c in roots}
a=summary['reference'];n=summary['smoothed_roi'];score=lambda d:np.mean([d[c]['MAE_mean'] for c in ['clean','h264_500k']])
checks={'score_improvement_10pct':bool(score(n)<=.9*score(a)),'clean_increase_le025':bool(n['clean']['MAE_mean']<=a['clean']['MAE_mean']+.25),
 'h264_improved_2of3':sum(results[s]['conditions']['h264_500k']['subject_macro_MAE']<old[s]['conditions']['h264_500k']['subject_macro_MAE'] for s in results)>=2,
 'h264_p95_increase_le1':n['h264_500k']['P95_mean']<=a['h264_500k']['P95_mean']+1,
 'h264_over20_not_increased':n['h264_500k']['over20_sum']<=a['h264_500k']['over20_sum']}
(out/'comparison.json').write_text(json.dumps({'summary':summary,'checks':checks,'keep_gate_passed':all(checks.values()),'test_opened':False,'paired_init_config_code_orders_verified':True},indent=2))
rankings={}
for partition in ['conditions','train_conditions']:
    rankings[partition]={}
    for c in results['2027'][partition]:
        paired={}
        for seed in results:
            ref={(r['subject'],r['window']):r for r in old[seed][partition][c]['records']}
            for r in results[seed][partition][c]['records']:
                key=(r['subject'],r['window']);o=ref[key]
                assert r['gt_hr']==o['gt_hr'] and r['start_s']==o['start_s']
                row=paired.setdefault(key,{'subject':key[0],'window':key[1],'gt_hr':r['gt_hr'],'start_s':r['start_s'],'seeds':{}})
                row['seeds'][seed]={'old_hr':o['pred_hr'],'new_hr':r['pred_hr'],'old_error':o['abs_error'],'new_error':r['abs_error']}
        rows=list(paired.values())
        for row in rows:
            row['mean_error_change']=float(np.mean([v['new_error']-v['old_error'] for v in row['seeds'].values()]))
            row['new_max_error']=max(v['new_error'] for v in row['seeds'].values())
        rankings[partition][c]={'by_new_worst_error':sorted(rows,key=lambda x:x['new_max_error'],reverse=True),
            'by_mean_change':sorted(rows,key=lambda x:x['mean_error_change'],reverse=True)}
(out/'window_error_rankings.json').write_text(json.dumps(rankings,indent=2))
lines=['ROI EMA .10s with IoU<.50 reset: controlled 3-seed development experiment',
       'Reference: MSE + H264 augmentation, same initialization, sample order, labels, optimizer and selection.',
       'No test evaluation. Validation and all bitrates have already been used for development.',
       'The unrestricted EMA failed train geometric checks; reset protection was added before any new training.',
       'Geometry agreement with detector is not anatomical tracking ground truth.',
       'Keep gate passed: '+str(all(checks.values())),json.dumps(checks),
       'Condition | Reference MAE mean +/- sample SD | New ROI MAE mean +/- sample SD']
for c in roots:lines.append(f"{c} | {a[c]['MAE_mean']:.6f} +/- {a[c]['MAE_sd']:.6f} | {n[c]['MAE_mean']:.6f} +/- {n[c]['MAE_sd']:.6f}")
lines+=['Hard training case subject11 window5 H264500:']
for r in rankings['train_conditions']['h264_500k']['by_new_worst_error']:
    if r['subject']=='subject11' and r['window']==5:lines.append(json.dumps(r))
(out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
(out/'COMPLETE.json').write_text(json.dumps({'complete':True,'runs':3,'epochs_each':30,'test_opened':False}))
print(json.dumps(summary),flush=True);print('ROI EXPERIMENT COMPLETE',checks,flush=True)
