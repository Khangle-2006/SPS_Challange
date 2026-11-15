"""Independent audit and Vietnamese report for matched compressed-ROI training."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_spectral_ensemble import independent_psd,audit_report
HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/matched_detector_roi_20261019'
BASE=OUT/'results/matched_detector_roi_20261019'
SEEDS=[2027,2028,2029];CONDS=['clean','h264_500k','h264_250k','h264_1000k']
METHODS=[str(s) for s in SEEDS]+['hr_median','psd_mean']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    complete=read(BASE/'COMPLETE.json');assert complete['runs']==6 and complete['epochs_each']==30 and complete['paired_verified'] and not complete['test_opened']
    protocol=read(BASE/'matched_protocol.json');preflight=read(BASE/'preflight.json')
    for n,h in protocol['code_sha256'].items():assert sha(OUT/'code/deep_hr'/n)==h
    assert preflight['validation']['windows']==42 and preflight['validation']['unusable_compressed']==0
    assert all(preflight['train'][a]['windows']==161 and preflight['train'][a]['all_interpolated_pixels_exact'] for a in ['native','detected'])
    results=read(BASE/'results.json');comparison=read(BASE/'comparison.json');frozen=read(BASE/'frozen_checkpoints.json')
    recheck=read(BASE/'checkpoint_recheck.json');assert len(recheck)==6
    for name,item in frozen.items():assert recheck[name]['matches_frozen'] and recheck[name]['sha256']==item['sha256']
    statuses={};histories={};provenance={};best={};durations={}
    for arm in ['native','detected']:
        statuses[arm]={};histories[arm]={};provenance[arm]={};best[arm]={};durations[arm]={}
        for seed in SEEDS:
            p=BASE/arm/f'augmentation_{seed}';s=read(p/'status.json');h=read(p/'history.json');prov=read(p/'provenance.json');b=read(p/'best_validation.json')
            assert s['state']=='complete' and s['completed_epochs']==30 and s['reload_verified'] and len(h)==30
            assert s['checkpoint_sha256']==frozen[f'{arm}_{seed}']['sha256']
            ix=min(range(30),key=lambda i:h[i]['score']);assert b['epoch']==s['best_epoch']==ix+1 and b['score']==h[ix]['score']==s['best']
            assert prov['protocol_sha256']==sha(p.parent/'protocol.json') and not prov['config']['test_opened']
            for item in h:assert abs(item['score']-np.mean([item['conditions'][c]['subject_macro_MAE'] for c in ['clean','h264_500k']]))<1e-10
            statuses[arm][str(seed)]=s;histories[arm][str(seed)]=h;provenance[arm][str(seed)]=prov;best[arm][str(seed)]=b['epoch'];durations[arm][str(seed)]=sum(x['elapsed_s'] for x in h)
    for seed in map(str,SEEDS):
        a,b=[provenance[x][seed] for x in ['native','detected']]
        assert a['initial_model_sha256']==b['initial_model_sha256'] and a['config']==b['config'] and a['code_sha256']==b['code_sha256']
        for x,y in zip(histories['native'][seed],histories['detected'][seed]):
            assert x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips']
    flat=[];summary={};maximum=0.
    reference=results['native']['clean']['2027']['records']
    for arm in ['native','detected']:
        summary[arm]={}
        for c in CONDS:
            waves=np.load(BASE/f'evaluation/{arm}_{c}_waves.npz')['waves'];assert waves.shape==(3,42,300) and np.isfinite(waves).all()
            f,p=independent_psd(waves);single=f[np.argmax(p,axis=-1)]
            predictions={str(seed):single[j] for j,seed in enumerate(SEEDS)}
            predictions.update(hr_median=np.median(single,axis=0),psd_mean=f[np.argmax(p.mean(axis=0),axis=-1)])
            summary[arm][c]={}
            for method in METHODS:
                r=results[arm][c][method];audit_report(r);assert r['n']==42 and r['invalid']==0
                for i,row in enumerate(r['records']):
                    assert all(row[k]==reference[i][k] for k in ['subject','window','start_s','gt_hr']) and row['split']=='val'
                    assert row['pred_hr']==float(predictions[method][i])
                if method.isdigit() and c in ['clean','h264_500k']:
                    selected=read(BASE/arm/f'augmentation_{method}/best_validation.json')['conditions'][c]
                    assert r['subject_macro_MAE']==selected['subject_macro_MAE']
                summary[arm][c][method]={k:v for k,v in r.items() if k!='records'}
                flat.append({'arm':arm,'condition':c,'method':method,**summary[arm][c][method]})
            for metric,field,agg in [('MAE_mean','subject_macro_MAE',np.mean),('MAE_sd','subject_macro_MAE',lambda x:np.std(x,ddof=1)),('P95_mean','P95',np.mean),('over20_sum','errors_over20',np.sum)]:
                v=agg([summary[arm][c][str(s)][field] for s in SEEDS]);assert abs(v-comparison['summary'][arm][c][metric])<1e-9
    a,b=comparison['summary']['native'],comparison['summary']['detected']
    score=lambda d:(d['clean']['MAE_mean']+d['h264_500k']['MAE_mean'])/2
    checks={'score_gain_at_least_10pct':bool(score(b)<=.9*score(a)),
        'clean_increase_at_most_025':bool(b['clean']['MAE_mean']<=a['clean']['MAE_mean']+.25),
        'h264_improved_at_least_2_seeds':sum(results['detected']['h264_500k'][str(s)]['subject_macro_MAE']<results['native']['h264_500k'][str(s)]['subject_macro_MAE'] for s in SEEDS)>=2,
        'h264_P95_increase_at_most_1':bool(b['h264_500k']['P95_mean']<=a['h264_500k']['P95_mean']+1),
        'h264_over20_not_increased':b['h264_500k']['over20_sum']<=a['h264_500k']['over20_sum']}
    assert checks==comparison['checks'] and all(checks.values())==comparison['keep_gate_passed']==complete['keep_gate_passed']
    changes={}
    for c in CONDS:
        pairs=[]
        for i,row in enumerate(reference):
            old=[results['native'][c][str(s)]['records'][i]['abs_error'] for s in SEEDS]
            new=[results['detected'][c][str(s)]['records'][i]['abs_error'] for s in SEEDS]
            pairs.append({'subject':row['subject'],'window':row['window'],'gt_hr':row['gt_hr'],'native_errors':old,'detected_errors':new,'mean_error_delta':float(np.mean(new)-np.mean(old))})
        changes[c]=sorted(pairs,key=lambda r:r['mean_error_delta'],reverse=True)
    save('verified_comparison.json',{'all_6_runs_complete':True,'paired_initialization_config_shuffle_augmentation_verified':True,'all_predictions_independently_reproduced':True,
        'best_epochs':best,'epoch_total_seconds':durations,'summary':summary,'three_seed_summary':comparison['summary'],'score_native':score(a),'score_detected':score(b),
        'score_relative_change_pct':100*(score(b)/score(a)-1),'checks':checks,'keep_gate_passed':all(checks.values()),'test_opened':False})
    save('window_changes_audited.json',changes)
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['KẾT QUẢ THÍ NGHIỆM HUẤN LUYỆN ROI H.264 CÓ ĐỐI CHỨNG','',
        'Hoàn tất 6/6 lượt, mỗi lượt đủ 30 epoch. Cùng mô hình EfficientPhys36 và MSE; 50% clean / 50% H.264 500 kbps.',
        'ROI cũ (native): phát hiện mặt trên video gốc rồi áp bbox lên H.264. ROI mới (detected): phát hiện trực tiếp trên H.264.',
        'Cả hai nhánh được train mới và chọn checkpoint bằng CÙNG clean + detector thực tế H.264 500 kbps. Không tái sử dụng checkpoint đối chứng được chọn theo ROI cũ.',
        '161/161 cửa sổ train và 42/42 validation hợp lệ; không loại mẫu. Kiểm tra khởi tạo, shuffle, lượt augmentation và nhãn khớp theo từng seed.',
        'Tất cả HR và chỉ số được tái tính độc lập từ waveform bằng phép giải detrend SPD dạng dense. Không dùng tập test.',
        '', 'MAE subject-macro (bpm), trung bình ± độ lệch chuẩn mẫu của 3 seed:', 'Điều kiện | ROI cũ | ROI mới | thay đổi']
    for c in CONDS:lines.append(f"{c} | {a[c]['MAE_mean']:.6f} ± {a[c]['MAE_sd']:.6f} | {b[c]['MAE_mean']:.6f} ± {b[c]['MAE_sd']:.6f} | {100*(b[c]['MAE_mean']/a[c]['MAE_mean']-1):+.2f}%")
    lines+=['','Chi tiết từng seed: seed | epoch tốt cũ/mới | clean cũ/mới | H264500 cũ/mới | score cũ/mới']
    for s in map(str,SEEDS):
        vals=[results[arm][c][s]['subject_macro_MAE'] for c in ['clean','h264_500k'] for arm in ['native','detected']]
        lines.append(f"{s} | {best['native'][s]}/{best['detected'][s]} | {vals[0]:.6f}/{vals[1]:.6f} | {vals[2]:.6f}/{vals[3]:.6f} | {np.mean(vals[::2]):.6f}/{np.mean(vals[1::2]):.6f}")
    lines+=['',f'Điểm chính clean/H264500: {score(a):.6f} → {score(b):.6f} ({100*(score(b)/score(a)-1):+.2f}%, thấp hơn là tốt hơn).',
        f"H264500 P95 trung bình: {a['h264_500k']['P95_mean']:.6f} → {b['h264_500k']['P95_mean']:.6f}; >20 bpm: {a['h264_500k']['over20_sum']} → {b['h264_500k']['over20_sum']}/126 dự đoán seed-cửa sổ (không phải 126 mẫu độc lập).",
        '', 'Tiêu chí định trước:']
    lines += [f'{k}: {v}' for k,v in checks.items()]
    lines += ['', 'Decoder cố định, MAE bpm; không chọn decoder theo bitrate sau khi nhìn kết quả:', 'Điều kiện | phương pháp | cũ | mới']
    for c in CONDS:
        for m in ['hr_median','psd_mean']:lines.append(f"{c} | {m} | {results['native'][c][m]['subject_macro_MAE']:.6f} | {results['detected'][c][m]['subject_macro_MAE']:.6f}")
    lines+=['','Các cửa sổ thay đổi nhiều nhất ở H264500 (lỗi trung bình 3 seed):']
    for title,rs in [('Tăng lỗi',changes['h264_500k'][:3]),('Giảm lỗi',changes['h264_500k'][-3:][::-1])]:
        for r in rs:lines.append(f"{title}: {r['subject']}/window{r['window']}, GT {r['gt_hr']:.3f}, lỗi {np.mean(r['native_errors']):.6f} → {np.mean(r['detected_errors']):.6f} ({r['mean_error_delta']:+.6f}).")
    lines+=['','Ca từng gây lỗi lớn subject32/window2:']
    for c in CONDS:
        r=next(r for r in changes[c] if r['subject']=='subject32' and r['window']==2)
        lines.append(f"{c}: lỗi 3 seed cũ={r['native_errors']}; mới={r['detected_errors']}")
    lines+=['','Kết luận: không thay baseline bằng cấu hình ROI mới này; không đạt cổng cải thiện định trước.',
        'ROI mới giảm lỗi tại 1000 kbps nhưng làm xấu điều kiện chính 500 kbps; đây là kết quả development, chưa phải bằng chứng tổng quát hóa.',
        'Chỉ có 7 subject validation, được sử dụng để chọn epoch và đã xem nhiều lần. 250/1000 kbps cũng là điều kiện development đã biết; không diễn giải như test độc lập.',
        'Không suy ra mọi detector/tracker mới đều thất bại; thí nghiệm chỉ kiểm tra thay nguồn ROI augmentation với Haar và cấu hình cố định này.',
        'Giữ detector trên video nén khi đánh giá thực tế. Cải thiện do đổi tiêu chí chọn checkpoint không được gán cho thay đổi ROI train.',
        'Bước hợp lý tiếp theo: giữ đối chứng hiện tại, ưu tiên dữ liệu ngoài/ban tổ chức khi được cấp; tránh tiếp tục chọn biến thể theo 7 subject validation.']
    (OUT/'REPORT_VI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained');x=np.arange(4)
    for shift,arm,label,color in [(-.17,'native','Native-derived training ROI','#3b6fb6'),(.17,'detected','H264-detected training ROI','#d77b32')]:
        d=comparison['summary'][arm];axes[0].bar(x+shift,[d[c]['MAE_mean'] for c in CONDS],width=.32,yerr=[d[c]['MAE_sd'] for c in CONDS],capsize=4,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=['Clean','H264 500k','H264 250k','H264 1000k'],ylabel='Subject-macro MAE (bpm)',title='Mean and sample SD across 3 seeds');axes[0].legend(fontsize=8)
    for s in map(str,SEEDS):
        scores=[np.mean([results[arm][c][s]['subject_macro_MAE'] for c in ['clean','h264_500k']]) for arm in ['native','detected']]
        axes[1].plot([0,1],scores,'o-',label=s)
    axes[1].set(xticks=[0,1],xticklabels=['Native training ROI','H264 training ROI'],ylabel='Mean clean / H264500 MAE (bpm)',title='Same actual-detector validation for both arms');axes[1].legend(title='Seed')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    print('\n'.join(lines[:25]))
if __name__=='__main__':main()
