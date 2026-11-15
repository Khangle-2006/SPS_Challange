"""Independently recompute ROI experiment metrics from saved predictions."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/roi_experiment_20261019'
NEW=OUT/'results/roi_ema010_ablation_v1'
OLD=HERE/'reports/h264_experiment_20261015/results/h264_ablation_v1'
EVOLD=HERE/'reports/spectral_experiment_20261015/results/spectral_ablation_v1/frozen_evaluation'
SEEDS=[2027,2028,2029];CONDS=['clean','h264_500k','h264_250k','h264_1000k']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def audit(r,n):
    rows=r['records'];assert len(rows)==r['n']==n
    e=np.array([abs(x['pred_hr']-x['gt_hr']) if x['pred_hr'] is not None else 135. for x in rows])
    assert all(abs(x['abs_error']-v)<1e-9 for x,v in zip(rows,e))
    metrics={'window_MAE':e.mean(),'window_RMSE':np.sqrt((e**2).mean()),'P95':np.percentile(e,95),
             'errors_over10':int((e>10).sum()),'errors_over20':int((e>20).sum()),
             'invalid':sum(x.get('invalid',False) for x in rows),
             'subject_macro_MAE':statistics.mean(statistics.mean(x['abs_error'] for x in rows if x['subject']==s) for s in {x['subject'] for x in rows})}
    for k,v in metrics.items():assert abs(r[k]-v)<1e-8,(k,r[k],v)
    return {k:float(v) if k not in ['errors_over10','errors_over20','invalid'] else int(v) for k,v in metrics.items()}
def main():
    assert read(NEW/'COMPLETE.json')=={'complete':True,'runs':3,'epochs_each':30,'test_opened':False}
    frozen=read(NEW/'frozen_checkpoints.json');comp=read(NEW/'comparison.json')
    values={'reference':{},'smoothed_roi':{}};metrics={};runs=[];pairs={}
    for s in SEEDS:
        oldp=OLD/f'augmentation_{s}';newp=NEW/f'augmentation_{s}'
        po=read(oldp/'provenance.json');pn=read(newp/'provenance.json')
        assert po['config']==pn['config'] and po['initial_model_sha256']==pn['initial_model_sha256'] and po['code_sha256']==pn['code_sha256']
        ho=read(oldp/'history.json');hn=read(newp/'history.json');status=read(newp/'status.json');best=read(newp/'best_validation.json')
        assert len(ho)==len(hn)==30 and status['state']=='complete' and status['reload_verified']
        assert all(a['sample_order_sha256']==b['sample_order_sha256'] and a['compressed_train_clips']==b['compressed_train_clips'] for a,b in zip(ho,hn))
        assert best['epoch']==min(hn,key=lambda r:r['score'])['epoch']==status['best_epoch']==frozen[str(s)]['best_epoch']
        assert sha(newp/'best.pt')==status['checkpoint_sha256']==frozen[str(s)]['sha256']
        values['reference'][s]=read(EVOLD/f'mse_augmentation_{s}.json')
        values['smoothed_roi'][s]=read(NEW/'evaluation'/f'{s}.json')
        assert values['smoothed_roi'][s]['checkpoint']==frozen[str(s)]
        for family in values:
            ev=values[family][s];assert not ev['test_opened']
            for partition in ['conditions','train_conditions']:
                for c,r in ev[partition].items():audit(r,42 if partition=='conditions' else 161)
            for c in ['clean','h264_500k']:
                if family=='smoothed_roi':assert ev['conditions'][c]['subject_macro_MAE']==best['conditions'][c]['subject_macro_MAE']
        row={'seed':s,'old_best_epoch':read(oldp/'status.json')['best_epoch'],'new_best_epoch':status['best_epoch']}
        for family in values:
            row[family]={c:values[family][s]['conditions'][c]['subject_macro_MAE'] for c in CONDS}
            row[family]['selection_score']=statistics.mean(row[family][c] for c in ['clean','h264_500k'])
        runs.append(row)
    for family,evs in values.items():
        metrics[family]={}
        for partition in ['conditions','train_conditions']:
            metrics[family][partition]={}
            for c in evs[2027][partition]:
                rs=[e[partition][c] for e in evs.values()];maes=[r['subject_macro_MAE'] for r in rs]
                metrics[family][partition][c]={'MAE_mean':statistics.mean(maes),'MAE_sd':statistics.stdev(maes),
                    'P95_mean':statistics.mean(r['P95'] for r in rs),'over20_sum':sum(r['errors_over20'] for r in rs),
                    'over10_sum':sum(r['errors_over10'] for r in rs),'invalid_sum':sum(r['invalid'] for r in rs)}
                if partition=='conditions':
                    for k,v in comp['summary'][family][c].items():assert abs(metrics[family][partition][c][k]-v)<1e-8
    for partition in ['conditions','train_conditions']:
        pairs[partition]={}
        for c in values['reference'][2027][partition]:
            joined={}
            for s in SEEDS:
                ref={(r['subject'],r['window']):r for r in values['reference'][s][partition][c]['records']}
                for r in values['smoothed_roi'][s][partition][c]['records']:
                    key=(r['subject'],r['window']);o=ref[key]
                    for k in ['subject','window','split','start_s','gt_hr']:assert r[k]==o[k]
                    row=joined.setdefault(key,{'subject':r['subject'],'window':r['window'],'start_s':r['start_s'],'gt_hr':r['gt_hr'],'seeds':{}})
                    row['seeds'][str(s)]={'old_pred':o['pred_hr'],'new_pred':r['pred_hr'],'old_error':o['abs_error'],'new_error':r['abs_error']}
            rows=list(joined.values())
            for r in rows:r['mean_error_change']=statistics.mean(v['new_error']-v['old_error'] for v in r['seeds'].values())
            pairs[partition][c]=sorted(rows,key=lambda r:r['mean_error_change'],reverse=True)
    old=metrics['reference']['conditions'];new=metrics['smoothed_roi']['conditions']
    scores={f:statistics.mean(metrics[f]['conditions'][c]['MAE_mean'] for c in ['clean','h264_500k']) for f in metrics}
    deterioration={c:100*(new[c]['MAE_mean']/old[c]['MAE_mean']-1) for c in CONDS}
    keep={'score_improvement_10pct':scores['smoothed_roi']<=.9*scores['reference'],
        'clean_increase_le025':new['clean']['MAE_mean']<=old['clean']['MAE_mean']+.25,
        'h264_improved_2of3':sum(r['smoothed_roi']['h264_500k']<r['reference']['h264_500k'] for r in runs)>=2,
        'h264_p95_increase_le1':new['h264_500k']['P95_mean']<=old['h264_500k']['P95_mean']+1,
        'h264_over20_not_increased':new['h264_500k']['over20_sum']<=old['h264_500k']['over20_sum']}
    assert keep==comp['checks'] and all(keep.values())==comp['keep_gate_passed']
    write('verified_summary.json',{'metrics':metrics,'per_seed':runs,'selection_scores':scores,'MAE_increase_percent':deterioration,'gate':keep,'all_audits_passed':True})
    write('verified_window_changes.json',pairs)
    x=np.arange(4);w=.36;fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for family,label,color,offset in [('reference','Old ROI','#4477aa',-w/2),('smoothed_roi','Smoothed ROI','#cc6677',w/2)]:
        d=metrics[family]['conditions'];axes[0].bar(x+offset,[d[c]['MAE_mean'] for c in CONDS],w,yerr=[d[c]['MAE_sd'] for c in CONDS],capsize=4,label=label,color=color)
        axes[1].bar(x+offset,[d[c]['P95_mean'] for c in CONDS],w,label=label,color=color)
    for ax in axes:
        ax.set_xticks(x,['Clean','H264 500k','H264 250k','H264 1000k']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend()
    axes[0].set_ylabel('HR MAE (bpm), mean +/- sample SD of 3 seeds');axes[1].set_ylabel('Mean per-seed 95th-percentile error (bpm)')
    fig.suptitle('Fixed MSE + H264 training: old vs causal smoothed ROI (validation)');fig.savefig(OUT/'roi_comparison.png',dpi=160);plt.close(fig)
    lines=['KET QUA ROI - DA KIEM TRA LAI TU TUNG DU DOAN',
       '3 seed 2027/2028/2029, moi seed 30 epoch; checkpoint da reload va kiem tra hash.',
       'Ket luan: ROI EMA 0.10s + reset IoU<0.50 KHONG dat tieu chi giu; giu baseline ROI cu.',
       'Cung khoi tao, cau hinh, ma trainer, sample order va lua chon augmentation tung epoch.',
       'Khong danh gia test. Val gom 7 nguoi, 42 cua so; da duoc su dung cho phat trien.',
       'MAE la subject-macro; +/- la sample SD giua 3 seed, khong phai khoang tin cay.',
       'P95 la trung binh percentile95 cua moi seed. So loi gop 42x3=126 du doan, khong phai 126 mau doc lap.',
       'Condition | Old MAE +/- SD | New MAE +/- SD | Increase | P95 old->new | >20 old->new']
    for c in CONDS:lines.append(f"{c}: {old[c]['MAE_mean']:.6f} +/- {old[c]['MAE_sd']:.6f} | {new[c]['MAE_mean']:.6f} +/- {new[c]['MAE_sd']:.6f} | {deterioration[c]:.2f}% | {old[c]['P95_mean']:.6f}->{new[c]['P95_mean']:.6f} | {old[c]['over20_sum']}->{new[c]['over20_sum']}")
    lines+=['PER SEED (old->new):']
    for r in runs:lines.append(json.dumps(r))
    lines+=['Selection score old/new: '+json.dumps(scores),'Keep gate: '+json.dumps(keep),
        'Subject11 window5 train H264500:']
    lines.extend(json.dumps(r) for r in pairs['train_conditions']['h264_500k'] if r['subject']=='subject11' and r['window']==5)
    lines+=['Top validation regressions H264500:']+[json.dumps(r) for r in pairs['conditions']['h264_500k'][:5]]
    lines+=['ROI smoother giam buoc nhay hinh hoc khong dong nghia cai thien tin hieu sinh ly. Chua co thi nghiem tach rieng de quy nguyen nhan cho do tre hay viec doi vung da.',
       'Ket qua nay bac bo cau hinh ROI cu the da thu, khong bac bo moi cach tracking/on dinh ROI.']
    (OUT/'REPORT_VI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'per_seed':runs,'scores':scores,'deterioration_pct':deterioration,'train_metrics':{f:metrics[f]['train_conditions'] for f in metrics},'hardcase':[r for r in pairs['train_conditions']['h264_500k'] if r['subject']=='subject11' and r['window']==5],'val500_worst':pairs['conditions']['h264_500k'][:3]},indent=2))
if __name__=='__main__':main()
