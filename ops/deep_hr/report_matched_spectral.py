"""Independent spectral audit, matched-loss comparison and subject32 report."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_spectral_ensemble import independent_psd,audit_report
HERE=Path(__file__).resolve().parent;OUT=HERE/'reports/matched_spectral_20261019';BASE=OUT/'results/matched_spectral_20261019'
REF=HERE/'reports/matched_detector_roi_20261019/results/matched_detector_roi_20261019/native'
SEEDS=[2027,2028,2029];CONDS=['clean','h264_500k','h264_250k','h264_1000k'];METHODS=[str(s) for s in SEEDS]+['hr_median','psd_mean']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    done=read(BASE/'COMPLETE.json');assert done['fresh_runs']==3 and done['epochs_each']==30 and done['paired_verified'] and done['mse_reference_reproduced'] and not done['test_opened']
    results=read(BASE/'results.json');comparison=read(BASE/'comparison.json');frozen=read(BASE/'frozen_checkpoints.json');best={};duration={}
    assert sha(OUT/'code/deep_hr/spectral_objective.py')==sha(HERE/'reports/spectral_experiment_20261015/code/deep_hr/spectral_objective.py')
    recheck=read(BASE/'checkpoint_recheck.json');assert len(recheck)==6
    for n,item in frozen.items():assert recheck[n]['verified'] and recheck[n]['sha256']==item['sha256']
    for seed in SEEDS:
        a=read(REF/f'augmentation_{seed}/provenance.json');b=read(BASE/f'augmentation_{seed}/provenance.json')
        assert a['initial_model_sha256']==b['initial_model_sha256'] and a['train_h264_inventory']==b['train_h264_inventory']
        assert a['torch']==b['torch'] and a['gpu']==b['gpu']
        assert {k:v for k,v in a['config'].items() if k!='loss'}=={k:v for k,v in b['config'].items() if k!='loss'}
        assert sha(REF/f'augmentation_{seed}/provenance.json')==b['reference_provenance_sha256'] and b['paired_preflight_passed']
        for n,h in b['code_sha256'].items():assert sha(OUT/'code/deep_hr'/n)==h
        assert sha(OUT/'code/deep_hr/MATCHED_SPECTRAL_PROTOCOL.json')==b['protocol_sha256']
        assert read(BASE/'protocol.json')==read(OUT/'code/deep_hr/MATCHED_SPECTRAL_PROTOCOL.json')
        ha=read(REF/f'augmentation_{seed}/history.json');hb=read(BASE/f'augmentation_{seed}/history.json');assert len(ha)==len(hb)==30
        for x,y in zip(ha,hb):assert x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips']
        st=read(BASE/f'augmentation_{seed}/status.json');bs=read(BASE/f'augmentation_{seed}/best_validation.json')
        ix=min(range(30),key=lambda i:hb[i]['score']);assert st['state']=='complete' and st['reload_verified'] and st['best_epoch']==bs['epoch']==ix+1 and bs['score']==hb[ix]['score']
        assert st['checkpoint_sha256']==frozen[f'spectral_{seed}']['sha256']
        best[str(seed)]={'mse':read(REF/f'augmentation_{seed}/status.json')['best_epoch'],'spectral':st['best_epoch']}
        duration[str(seed)]=sum(x['elapsed_s'] for x in hb)
    flat=[];max_psd_delta=0.;refrows=results['mse']['clean']['2027']['records'];assert len(refrows)==42
    for family in ['mse','spectral']:
        for c in CONDS:
            z=np.load(BASE/f'evaluation/{family}_{c}_waves.npz');w=z['waves'];assert w.shape==(3,42,300) and np.isfinite(w).all()
            f,p=independent_psd(w);delta=float(np.max(np.abs(p.transpose(1,0,2)-z['normalized_psd'])));assert delta<1e-7;max_psd_delta=max(max_psd_delta,delta)
            single=f[np.argmax(p,axis=-1)];pred={str(s):single[j] for j,s in enumerate(SEEDS)}
            pred.update(hr_median=np.median(single,axis=0),psd_mean=f[np.argmax(p.mean(axis=0),axis=-1)])
            for method,r in results[family][c].items():
                audit_report(r);assert r['n']==42 and r['invalid']==0
                for i,row in enumerate(r['records']):
                    assert row['split']=='val' and all(row[k]==refrows[i][k] for k in ['subject','window','start_s','gt_hr'])
                    assert row['pred_hr']==float(pred[method][i])
                flat.append({'family':family,'condition':c,'method':method,**{k:v for k,v in r.items() if k!='records'}})
                if method.isdigit() and c in ['clean','h264_500k']:
                    root=REF if family=='mse' else BASE
                    selected=read(root/f'augmentation_{method}/best_validation.json')['conditions'][c]
                    assert r['subject_macro_MAE']==selected['subject_macro_MAE']
    summary={fam:{c:{'MAE_mean':float(np.mean([results[fam][c][str(s)]['subject_macro_MAE'] for s in SEEDS])),
        'MAE_sd':float(np.std([results[fam][c][str(s)]['subject_macro_MAE'] for s in SEEDS],ddof=1)),
        'P95_mean':float(np.mean([results[fam][c][str(s)]['P95'] for s in SEEDS])),
        'over20_sum':sum(results[fam][c][str(s)]['errors_over20'] for s in SEEDS)} for c in CONDS} for fam in ['mse','spectral']}
    for fam in summary:
        for c in CONDS:
            for k,v in summary[fam][c].items():assert abs(v-comparison['summary'][fam][c][k])<1e-9
    a,b=summary['mse'],summary['spectral'];score=lambda d:(d['clean']['MAE_mean']+d['h264_500k']['MAE_mean'])/2
    checks={'score_gain_at_least_10pct':bool(score(b)<=.9*score(a)),'clean_increase_at_most_025':bool(b['clean']['MAE_mean']<=a['clean']['MAE_mean']+.25),
        'h264_improved_at_least_2_seeds':sum(results['spectral']['h264_500k'][str(s)]['subject_macro_MAE']<results['mse']['h264_500k'][str(s)]['subject_macro_MAE'] for s in SEEDS)>=2,
        'h264_P95_increase_at_most_1':bool(b['h264_500k']['P95_mean']<=a['h264_500k']['P95_mean']+1),
        'h264_over20_not_increased':b['h264_500k']['over20_sum']<=a['h264_500k']['over20_sum']}
    assert checks==comparison['checks'] and all(checks.values())==comparison['keep_gate_passed']==done['keep_gate_passed']
    diag=read(BASE/'subject32/diagnosis.json');tr=np.load(BASE/'subject32/diagnosis_traces.npz');assert diag['raw_ppg_timestamps_match_training_cache'] and not diag['test_opened']
    for row in diag['records']:
        pre=f"w{row['window']}_"
        for n,st in row['reference'].items():
            f,p=independent_psd(tr[pre+n]);assert st['hr']==float(f[np.argmax(p)])
            assert abs(st['power_within_reference_pm6']-p[np.abs(f-row['gt_hr'])<=6].sum())<1e-8
        assert row['reference']['ppg']['hr']==row['reference']['legacy_reference']['hr']==row['reference']['training_target_reconstructed']['hr']==row['gt_hr']
    changes={}
    for c in CONDS:
        changes[c]=[]
        for i,row in enumerate(refrows):
            old=[results['mse'][c][str(s)]['records'][i]['abs_error'] for s in SEEDS];new=[results['spectral'][c][str(s)]['records'][i]['abs_error'] for s in SEEDS]
            changes[c].append({'subject':row['subject'],'window':row['window'],'gt_hr':row['gt_hr'],'mse_errors':old,'spectral_errors':new,'mean_error_delta':float(np.mean(new)-np.mean(old))})
        changes[c].sort(key=lambda x:x['mean_error_delta'],reverse=True)
    save('verified_comparison.json',{'three_fresh_runs_complete':True,'paired_verified':True,'all_predictions_independently_reproduced':True,'max_PSD_delta_dense_vs_sparse':max_psd_delta,
        'summary':summary,'best_epochs':best,'epoch_total_seconds':duration,'score_mse':score(a),'score_spectral':score(b),'score_relative_change_pct':100*(score(b)/score(a)-1),'checks':checks,'keep_gate_passed':all(checks.values()),'test_opened':False})
    save('window_changes_audited.json',changes)
    subjects={}
    for subject in sorted({r['subject'] for r in refrows}):
        subjects[subject]={}
        for family in ['mse','spectral']:
            subjects[subject][family]=float(np.mean([np.mean([r['abs_error'] for r in results[family][c][str(s)]['records'] if r['subject']==subject]) for s in SEEDS for c in ['clean','h264_500k']]))
        subjects[subject]['score_delta']=subjects[subject]['spectral']-subjects[subject]['mse']
    assert abs(np.mean([r['score_delta'] for r in subjects.values()])-(score(b)-score(a)))<1e-9
    save('primary_subject_contributions.json',subjects)
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['THÍ NGHIỆM MSE + 0,1 SPECTRAL KL VỚI VALIDATION DETECTOR THỰC TẾ','',
        '3 seed mới hoàn tất đủ 30 epoch, so với 3 đối chứng MSE matched_detector_roi_20261019/native.',
        'Cùng EfficientPhys36, ROI train lấy từ ảnh gốc, 50% clean/50% H264500, batch2, AdamW, OneCycleLR max0.009, seed2027/2028/2029.',
        'Khởi tạo, config ngoài loss, shuffle, lượt augmentation và dữ liệu được kiểm tra khớp. Cùng chọn checkpoint bằng clean + actual-detector H264500; earliest minimum.',
        'KL(target PSD || predicted PSD), weight0.1 và spectral_objective.py giữ nguyên. Không dò trọng số theo validation.',
        'Đã tái tính độc lập toàn bộ HR/chỉ số từ waveform bằng dense SPD detrend; đối chứng MSE tái hiện đúng kết quả cũ. Không mở tập test.',
        '', 'MAE subject-macro bpm, trung bình ± SD mẫu 3 seed:', 'Điều kiện | MSE | MSE+KL | thay đổi']
    for c in CONDS:lines.append(f"{c} | {a[c]['MAE_mean']:.6f} ± {a[c]['MAE_sd']:.6f} | {b[c]['MAE_mean']:.6f} ± {b[c]['MAE_sd']:.6f} | {100*(b[c]['MAE_mean']/a[c]['MAE_mean']-1):+.2f}%")
    lines+=['','Từng seed: seed | best epoch MSE/KL | clean MSE/KL | H264500 MSE/KL | score MSE/KL']
    for s in map(str,SEEDS):
        v=[results[fam][c][s]['subject_macro_MAE'] for c in ['clean','h264_500k'] for fam in ['mse','spectral']]
        lines.append(f"{s} | {best[s]['mse']}/{best[s]['spectral']} | {v[0]:.6f}/{v[1]:.6f} | {v[2]:.6f}/{v[3]:.6f} | {np.mean(v[::2]):.6f}/{np.mean(v[1::2]):.6f}")
    lines+=['',f'Điểm chính: {score(a):.6f} → {score(b):.6f}; thay đổi {100*(score(b)/score(a)-1):+.2f}% (thấp hơn tốt hơn).',f'Cổng định trước: {all(checks.values())}.']
    lines += [f'{k}: {v}' for k,v in checks.items()]
    lines+=['','Điểm clean/500 theo subject, trung bình 3 seed (mô tả hậu nghiệm):']
    for s,r in sorted(subjects.items(),key=lambda x:x[1]['score_delta'],reverse=True):lines.append(f"{s}: {r['mse']:.6f} → {r['spectral']:.6f} ({r['score_delta']:+.6f})")
    lines+=['','P95 trung bình và số lỗi >20 bpm gộp qua 3 seed (126 dự đoán trên 42 cửa sổ, không phải 126 mẫu độc lập):']
    for c in CONDS:lines.append(f"{c}: P95 {a[c]['P95_mean']:.6f} → {b[c]['P95_mean']:.6f}; >20: {a[c]['over20_sum']} → {b[c]['over20_sum']}/126")
    lines+=['','Ensemble cố định: điều kiện | decoder | MSE | MSE+KL']
    for c in CONDS:
        for m in ['hr_median','psd_mean']:lines.append(f"{c} | {m} | {results['mse'][c][m]['subject_macro_MAE']:.6f} | {results['spectral'][c][m]['subject_macro_MAE']:.6f}")
    lines+=['','PHÂN TÍCH SUBJECT32: mô tả hậu nghiệm, không sửa nhãn hoặc chọn đỉnh bằng GT.']
    for r in diag['records']:
        target=r['reference']['ppg'];sensor=r['sensor_HR'];lines+=['',f"Window{r['window']}, start={r['start_s']:.6f}s, GT={r['gt_hr']:.6f} bpm.",
            f"PPG gốc, tham chiếu legacy và target training tái dựng có cùng HR={target['hr']:.6f}: không thấy lỗi đổi HR do hai bước resample/forward-difference trong cửa sổ này.",
            f"Hai đỉnh PPG cao nhất: {target['peaks'][:2]}",f"Sensor HR median={sensor['median']:.3f}, range={sensor['min']:.3f}–{sensor['max']:.3f}; max timestamp gap={r['timing']['max_gap_s']:.3f}s.",
            f"Đỉnh phổ PPG hai nửa 5s: {target['first_half_hr']:.3f}/{target['second_half_hr']:.3f}. Cửa sổ ngắn có độ phân giải thấp, không đủ để kết luận HR sinh lý thay đổi."]
        for c in CONDS:
            inp=r['inputs'][c]['pos'];lines.append(f"{c}: POS HR={inp['hr']:.3f}, power trong GT±6={inp['power_within_reference_pm6']:.4f}; ROI max coordinate step={r['inputs'][c]['roi_max_coordinate_step_px']}px.")
            for fam in ['mse','spectral']:
                lines.append(f"  {fam}: HR={[r['models'][fam][c][str(s)]['hr'] for s in SEEDS]}; lỗi={[r['models'][fam][c][str(s)]['abs_error'] for s in SEEDS]}")
            for decoder in ['hr_median','psd_mean']:
                errors=[]
                for fam in ['mse','spectral']:
                    rec=next(x for x in results[fam][c][decoder]['records'] if x['subject']=='subject32' and x['window']==r['window'])
                    errors.append(rec['abs_error'])
                lines.append(f'  {decoder}: lỗi MSE/KL={errors}')
        if r['window']==0:lines.append('Diễn giải: tham chiếu PPG có đỉnh cạnh tranh gần99.3 bên cạnh đỉnh88.3; sensor HR cũng khác đỉnh FFT. Tuy nhiên phần lớn dự đoán clean/500 nằm khoảng77–80, không phải đỉnh99.3: không thể quy toàn bộ lỗi mô hình cho tham chiếu nhiều đỉnh. KL chưa giải quyết ca này. Chưa có cơ sở kết luận GT sai: thuật toán/smoothing/độ trễ sensor HR chưa được xác nhận. Giữ nguyên nhãn và tính đủ lỗi.')
        else:
            lines.append('Diễn giải: PPG tham chiếu có đỉnh108.98 rõ; ở250k, POS vẫn có đỉnh gần108 nhưng bị đỉnh85.25 vượt qua. Đây là dấu hiệu giảm độ nổi bật của thành phần gầnGT trong tín hiệu POS, không chứng minh đỉnh đó là xung thật hoặc mô hình chắc chắn có thể khôi phục. Các dự đoán quanh72 không phải lỗi nửa HR (72/109 ≈0.66). KL chỉ sửa được1/3 seed ở250k; HR median vẫn sai37.35, nhưng PSD ensemble giảm lỗi xuống0.879.')
            old=next(x for x in results['mse']['h264_250k']['psd_mean']['records'] if x['subject']=='subject32' and x['window']==2)
            new=next(x for x in results['spectral']['h264_250k']['psd_mean']['records'] if x['subject']=='subject32' and x['window']==2)
            net=sum(x['abs_error']-y['abs_error'] for x,y in zip(results['mse']['h264_250k']['psd_mean']['records'],results['spectral']['h264_250k']['psd_mean']['records']))
            share=(old['abs_error']-new['abs_error'])/net
            lines.append(f'Ca window2 đóng góp {share*100:.2f}% mức giảm lỗi ròng của spectral PSD so với MSE PSD trên250k; không coi mức giảm ensemble tổng thể là cải thiện trải rộng mọi cửa sổ.')
    lines+=['','Giới hạn: 7 validation subject đã được dùng nhiều lần; 250/1000 cũng là điều kiện development đã biết. Không chọn decoder/bitrate/loss weight theo kết quả hậu nghiệm. Không suy ra thứ hạng cuộc thi.',
        'Quyết định theo cổng định trước: '+('giữ spectral KL làm ứng viên cho xác nhận trên dữ liệu ngoài, chưa khẳng định tổng quát hóa.' if all(checks.values()) else 'chưa thay baseline MSE bằng spectral KL; báo cáo mọi kết quả kể cả không cải thiện.')]
    (OUT/'REPORT_VI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained');x=np.arange(4)
    for shift,fam,label,color in [(-.17,'mse','MSE','#3b6fb6'),(.17,'spectral','MSE + 0.1 spectral KL','#d77b32')]:
        d=summary[fam];axes[0].bar(x+shift,[d[c]['MAE_mean'] for c in CONDS],width=.32,yerr=[d[c]['MAE_sd'] for c in CONDS],capsize=4,color=color,label=label)
    axes[0].set(xticks=x,xticklabels=['Clean','H264 500k','H264 250k','H264 1000k'],ylabel='Subject-macro MAE (bpm)',title='Mean and sample SD across 3 seeds');axes[0].legend(fontsize=8)
    for s in map(str,SEEDS):axes[1].plot([0,1],[np.mean([results[fam][c][s]['subject_macro_MAE'] for c in ['clean','h264_500k']]) for fam in ['mse','spectral']],'o-',label=s)
    axes[1].set(xticks=[0,1],xticklabels=['MSE','MSE + spectral KL'],ylabel='Clean/H264500 mean MAE (bpm)',title='Same actual-detector checkpoint selection');axes[1].legend(title='Seed')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7.2),layout='constrained');f=tr['bpm']
    for k,r in enumerate(diag['records']):
        pre=f"w{r['window']}_";p=tr[pre+'ppg_psd'];axes[k,0].plot(f,p/p.max(),label='Contact PPG');axes[k,0].axvline(r['sensor_HR']['median'],color='gray',ls=':',label='Sensor HR median')
        for c,col in [('clean','#3b6fb6'),('h264_500k','#569d63'),('h264_250k','#d77b32')]:
            p=tr[pre+c+'_pos_psd'];axes[k,1].plot(f,p/p.max(),label=c,color=col)
        for fam,col in [('mse','#3b6fb6'),('spectral','#d77b32')]:
            for j,s in enumerate(SEEDS):
                p=tr[pre+fam+'_h264_250k_'+str(s)+'_psd'];axes[k,2].plot(f,p/p.max(),color=col,alpha=.65,label=fam if j==0 else None)
        for ax in axes[k]:ax.axvline(r['gt_hr'],color='black',ls='--',lw=1,label='Reference FFT HR');ax.set(xlim=(45,180),xlabel='Frequency (bpm)',ylabel='Power / maximum');ax.grid(alpha=.15);ax.legend(fontsize=7)
        axes[k,0].set_title(f"subject32 window{r['window']}: reference")
        axes[k,1].set_title('POS from actual detector ROI')
        axes[k,2].set_title('Model spectra on H264 250k, all 3 seeds')
    fig.savefig(OUT/'subject32_spectra.png',dpi=170);plt.close(fig)
    print('\n'.join(lines[:23]));print('Independent PSD maximum difference:',max_psd_delta)
if __name__=='__main__':main()
