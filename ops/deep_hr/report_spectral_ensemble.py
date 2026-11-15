"""Independent batched linear-algebra PSD audit and ensemble comparison report."""
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
from scipy import signal,linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'reports/spectral_ensemble_20261019';BASE=OUT/'results/spectral_ensemble_20261019'
SEEDS=[2027,2028,2029];METHODS=[str(s) for s in SEEDS]+['hr_median','psd_mean'];CONDS=['clean','h264_500k','h264_250k','h264_1000k']
LABELS={'mse':'MSE + H264','spectral':'Spectral + H264'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def independent_psd(waves):
    # Dense SPD solve rather than per-wave sparse detrend from inference.
    n=waves.shape[-1];eye=np.eye(n);d=np.diff(eye,n=2,axis=0)
    h=eye-linalg.solve(eye+100**2*(d.T@d),eye,assume_a='pos')
    detrended=np.asarray(waves,dtype=float)@h.T
    filt=signal.sosfiltfilt(signal.butter(1,[.75,3.],fs=30,btype='bandpass',output='sos'),detrended,axis=-1)
    f,p=signal.periodogram(filt,fs=30,window=signal.windows.hann(n,sym=False),nfft=4096,detrend=False,axis=-1)
    mask=(f>=.75)&(f<=3);p=p[...,mask];return f[mask]*60,p/p.sum(axis=-1,keepdims=True)
def audit_report(r):
    rows=r['records'];assert len(rows)==r['n'];e=np.array([abs(x['pred_hr']-x['gt_hr']) if x['pred_hr'] is not None else 135. for x in rows])
    assert all(abs(x['abs_error']-v)<1e-9 for x,v in zip(rows,e))
    expected={'subject_macro_MAE':statistics.mean(statistics.mean(x['abs_error'] for x in rows if x['subject']==s) for s in {x['subject'] for x in rows}),
       'window_MAE':float(e.mean()),'window_RMSE':float(np.sqrt((e**2).mean())),'P95':float(np.percentile(e,95)),
       'errors_over10':int((e>10).sum()),'errors_over20':int((e>20).sum()),'max_error':float(e.max()),'invalid':sum(x.get('invalid',False) for x in rows)}
    for k,v in expected.items():assert abs(r[k]-v)<1e-8,(k,r[k],v)
def main():
    done=read(BASE/'COMPLETE.json');assert done['complete'] and done['frozen_models']==6 and not done['test_evaluated']
    result=read(BASE/'results.json');protocol=read(BASE/'protocol.json');timing=read(BASE/'timing.json')
    assert result['protocol_sha256']==sha(BASE/'protocol.json') and result['all_single_predictions_reproduced']
    assert read(BASE/'math_verification.json')['passed']
    for name,h in protocol['code_sha256'].items():assert sha(OUT/'code/deep_hr'/name)==h
    for family in ['mse','spectral']:
        oldroot=HERE/'reports'/('h264_experiment_20261015/results/h264_ablation_v1' if family=='mse' else 'spectral_experiment_20261015/results/spectral_ablation_v1')
        for seed in SEEDS:assert sha(oldroot/f'augmentation_{seed}/best.pt')==protocol['checkpoints'][family][str(seed)]['sha256']
    groups=result['groups'];max_psd_delta=0.;summary={};flat=[];window_changes={};hard=[]
    for family,data in groups.items():
        summary[family]={};window_changes[family]={}
        for part,conditions in [('validation',CONDS),('train',['clean','h264_500k'])]:
            assert set(data[part])==set(conditions);summary[family][part]={};window_changes[family][part]={}
            for c in conditions:
                reports=data[part][c];assert set(reports)==set(METHODS)
                z=np.load(BASE/f'waveforms/{family}_{part}_{c}.npz');waves=z['waves'];f,p=independent_psd(waves)
                delta=float(np.max(np.abs(p.transpose(1,0,2)-z['normalized_psd'])));assert delta<1e-7,delta;max_psd_delta=max(max_psd_delta,delta)
                assert waves.shape==(3,42 if part=='validation' else 161,300)
                single=f[np.argmax(p,axis=-1)]
                preds={str(s):single[j] for j,s in enumerate(SEEDS)};preds['hr_median']=np.median(single,axis=0);preds['psd_mean']=f[np.argmax(p.mean(axis=0),axis=-1)]
                meta=reports['2027']['records'];assert all(r['split']==('val' if part=='validation' else 'train') for r in meta)
                summary[family][part][c]={}
                for m,r in reports.items():
                    audit_report(r)
                    assert len(r['records'])==waves.shape[1]
                    for i,rec in enumerate(r['records']):
                        for k in ['subject','window','start_s','gt_hr']:assert rec[k]==meta[i][k]
                        assert rec['pred_hr']==float(preds[m][i]),(family,part,c,m,i,rec['pred_hr'],float(preds[m][i]))
                    summary[family][part][c][m]={k:v for k,v in r.items() if k!='records'}
                    flat.append({'family':family,'partition':part,'condition':c,'method':m,**summary[family][part][c][m]})
                changes=[]
                for i,row in enumerate(meta):
                    old=reports['hr_median']['records'][i];new=reports['psd_mean']['records'][i]
                    changes.append({'subject':row['subject'],'window':row['window'],'start_s':row['start_s'],'gt_hr':row['gt_hr'],
                       'single_hr':[reports[str(s)]['records'][i]['pred_hr'] for s in SEEDS],
                       'median_hr':old['pred_hr'],'psd_mean_hr':new['pred_hr'],'median_error':old['abs_error'],'psd_mean_error':new['abs_error'],
                       'error_change_vs_median':new['abs_error']-old['abs_error']})
                    if (part=='train' and (row['subject'],row['window']) in [('subject11',5),('subject10',4)]) or (part=='validation' and (row['subject'],row['window']) in [('subject32',0),('subject9',3),('subject26',0)]):
                        hard.append({'family':family,'partition':part,'condition':c,**changes[-1]})
                window_changes[family][part][c]=sorted(changes,key=lambda r:r['error_change_vs_median'],reverse=True)
        for m,t in timing[family]['methods'].items():
            a=np.array(t['samples_ms']);assert len(a)==42 and np.isfinite(a).all() and (a>0).all()
            assert abs(float(np.median(a))-t['end_to_end_ms_median'])<1e-8
            assert abs(float(np.percentile(a,95))-t['end_to_end_ms_p95'])<1e-8
    gates={}
    for family,data in summary.items():
        val=data['validation'];score=lambda m:statistics.mean(val[c][m]['subject_macro_MAE'] for c in ['clean','h264_500k'])
        mean_score=statistics.mean(score(str(s)) for s in SEEDS);med_score=score('hr_median');new_score=score('psd_mean')
        mean_clean=statistics.mean(val['clean'][str(s)]['subject_macro_MAE'] for s in SEEDS)
        mean_p95=statistics.mean(val['h264_500k'][str(s)]['P95'] for s in SEEDS)
        mean_rate=statistics.mean(val['h264_500k'][str(s)]['errors_over20']/42 for s in SEEDS)
        checks={'score_improve_10pct_vs_mean_single':new_score<=.9*mean_score,'score_improve_10pct_vs_HR_median':new_score<=.9*med_score,
           'clean_increase_le025_vs_both':val['clean']['psd_mean']['subject_macro_MAE']<=min(mean_clean,val['clean']['hr_median']['subject_macro_MAE'])+.25,
           'h264500_P95_increase_le1_vs_both':val['h264_500k']['psd_mean']['P95']<=min(mean_p95,val['h264_500k']['hr_median']['P95'])+1,
           'h264500_over20_rate_not_increased_vs_both':val['h264_500k']['psd_mean']['errors_over20']/42<=min(mean_rate,val['h264_500k']['hr_median']['errors_over20']/42)}
        gates[family]={'checks':checks,'passed':all(checks.values()),'score_mean_single':mean_score,'score_HR_median':med_score,'score_PSD_mean':new_score,
            'PSD_improvement_pct_vs_mean_single':100*(1-new_score/mean_score),'PSD_improvement_pct_vs_HR_median':100*(1-new_score/med_score)}
    change_stats={}
    for family in ['mse','spectral']:
        change_stats[family]={}
        for c in CONDS:
            rs=window_changes[family]['validation'][c];d=np.array([r['error_change_vs_median'] for r in rs]);best=min(rs,key=lambda r:r['error_change_vs_median'])
            change_stats[family][c]={'improved_windows':int((d<0).sum()),'worsened_windows':int((d>0).sum()),'tied_windows':int((d==0).sum()),
              'total_abs_error_reduction':float(-d.sum()),'largest_single_window_error_reduction':float(-best['error_change_vs_median']),
              'largest_improvement_window':{'subject':best['subject'],'window':best['window']},
              'largest_window_share_of_net_error_reduction':float(-best['error_change_vs_median']/(-d.sum())) if d.sum()<0 else None}
    audited={'summary':summary,'gates':gates,'change_vs_median':change_stats,'all_single_and_ensemble_predictions_independently_reproduced':True,'max_normalized_psd_difference':max_psd_delta,
       'timing':timing,'test_evaluated':False,'ensemble_uncertainty_note':'One deterministic ensemble per family; no SD of independent ensembles. Single-seed SD describes training seed spread only.'}
    write('verified_comparison.json',audited);write('window_changes.json',window_changes);write('hard_cases.json',hard)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['THÍ NGHIỆM ENSEMBLE PHỔ 3 SEED — 2026-10-19',
       'Hai nhóm MSE+H264 / spectral+H264, ROI gốc,6 checkpoint đóng băng. Không huấn luyện hay chọn lại checkpoint.',
       'Đã kiểm tra lại hash checkpoint/code, toàn bộ dự đoán đơn lẻ và ensemble. Audit PSD dùng dense SPD solve thay cho sparse detrend từng waveform.',
       f'Sai khác PSD chuẩn hóa lớn nhất giữa hai triển khai: {max_psd_delta:.3g}.',
       'Chuẩn hóa L1 công suất mỗi seed trong45-180bpm, trung bình trọng số1/3, lấy đỉnh. Đối chứng trung vị HR dùng cùng3 seed.',
       'Validation:7 người,42 cửa sổ. Mỗi ensemble là một predictor; không có3 ensemble độc lập để tính SD. Các bitrate đã dùng phát triển; không phải test độc lập.',
       'Số lỗi của mỗi predictor là trên42 cửa sổ val hoặc161 cửa sổ train, không gộp3 seed để so với1 ensemble.',
       'Thời gian là batch1 trên P100 với cache36px và weights nạp sẵn; bao gồm nội suy/normalize, chuyển GPU, forward và hậu xử lý. Không gồm decode/detector/thu nhận10s.',
       'Mỗi dòng: method | clean | H264500 | H264250 | H2641000 (subject-macro MAE bpm)']
    for family in ['mse','spectral']:
        lines.append(LABELS[family])
        for m in METHODS:lines.append(m+' | '+' | '.join(f"{summary[family]['validation'][c][m]['subject_macro_MAE']:.6f}" for c in CONDS))
        lines.append('Tiêu chí giữ đã đặt trước: '+json.dumps(gates[family],ensure_ascii=False))
        for m,t in timing[family]['methods'].items():lines.append(f"Timing {m}: median={t['end_to_end_ms_median']:.3f}ms, P95={t['end_to_end_ms_p95']:.3f}ms/clip10s")
    lines+=['LỖI ĐUÔI VALIDATION',json.dumps({f:{c:{m:{k:summary[f]['validation'][c][m][k] for k in ['P95','errors_over10','errors_over20','invalid']} for m in METHODS} for c in CONDS} for f in ['mse','spectral']},ensure_ascii=False,indent=2),
       'MỨC CẢI THIỆN TẬP TRUNG Ở CÁC CỬA SỔ NÀO (PHÂN TÍCH SAU KHI CÓ KẾT QUẢ)',json.dumps(change_stats,ensure_ascii=False,indent=2),
       'Ở H264250, phần lớn mức giảm tổng lỗi so với HR-median đến từ một cửa sổ mỗi nhóm. Không suy rộng thành mức cải thiện đồng đều hay bằng chứng tổng quát hóa.',
       'CÁC ĐOẠN KHÓ ĐÃ BIẾT TỪ VÒNG TRƯỚC',json.dumps(hard,ensure_ascii=False,indent=2),
       'Không tự thay baseline hay kết luận khả năng vào top3 từ validation UBFC. Lưu kết quả âm/dương và giữ protocol cố định.']
    (OUT/'REPORT_VI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained',sharey=True);x=np.arange(4);w=.25
    for ax,family in zip(axes,['mse','spectral']):
        val=summary[family]['validation'];mean=[statistics.mean(val[c][str(s)]['subject_macro_MAE'] for s in SEEDS) for c in CONDS]
        sd=[statistics.stdev(val[c][str(s)]['subject_macro_MAE'] for s in SEEDS) for c in CONDS]
        ax.bar(x-w,mean,w,yerr=sd,capsize=3,label='Single seeds: mean +/- SD',color='#999999')
        ax.bar(x,[val[c]['hr_median']['subject_macro_MAE'] for c in CONDS],w,label='Median of 3 HRs',color='#4477aa')
        ax.bar(x+w,[val[c]['psd_mean']['subject_macro_MAE'] for c in CONDS],w,label='Mean of 3 normalized PSDs',color='#228833')
        ax.set_xticks(x,['Clean','H264500k','H264250k','H2641000k']);ax.set(title=LABELS[family],ylabel='Validation subject-macro MAE (bpm)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(OUT/'ensemble_comparison.png',dpi=160);plt.close(fig)
    # Plot fixed prior difficult training cases; sign/phase alignment is unnecessary for PSD fusion.
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,(subject,window) in enumerate([('subject11',5),('subject10',4)]):
        for col,family in enumerate(['mse','spectral']):
            records=groups[family]['train']['h264_500k']['2027']['records'];i=next(i for i,r in enumerate(records) if r['subject']==subject and r['window']==window)
            z=np.load(BASE/f'waveforms/{family}_train_h264_500k.npz');f,p=independent_psd(z['waves'][:,i,:]);ax=axes[row,col]
            for j,s in enumerate(SEEDS):ax.plot(f,p[j],lw=.8,alpha=.6,label=str(s))
            ax.plot(f,p.mean(axis=0),color='black',lw=1.8,label='PSD mean');ax.axvline(records[i]['gt_hr'],ls='--',color='#cc6677',label='Reference HR')
            ax.set(title=f'{subject} window{window} / {LABELS[family]}',xlabel='bpm',ylabel='Power / total band power');ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.savefig(OUT/'hard_case_spectra.png',dpi=160);plt.close(fig)
    print(json.dumps({'gates':gates,'validation_MAE':{f:{m:[round(summary[f]['validation'][c][m]['subject_macro_MAE'],6) for c in CONDS] for m in METHODS} for f in ['mse','spectral']},
       'validation_tails':{f:{c:{m:[summary[f]['validation'][c][m]['P95'],summary[f]['validation'][c][m]['errors_over20']] for m in METHODS} for c in CONDS} for f in ['mse','spectral']},
       'timing_ms':{f:{m:round(v['end_to_end_ms_median'],3) for m,v in timing[f]['methods'].items()} for f in timing}},indent=2))
if __name__=='__main__':main()
