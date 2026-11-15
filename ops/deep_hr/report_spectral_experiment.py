"""Audit all frozen evaluations and produce the spectral-loss experiment report."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/spectral_experiment_20261015'
BASE=OUT/'results/spectral_ablation_v1'
OLD=HERE/'reports/h264_experiment_20261015/results/h264_ablation_v1'
SEEDS=[2027,2028,2029]
GROUPS=['mse_control','mse_augmentation','spectral_control','spectral_augmentation']
LABELS=['MSE / clean train','MSE / H264 aug.','MSE+PSD / clean train','MSE+PSD / H264 aug.']
COLORS=['#4477aa','#66ccee','#aa3377','#228833']
CONDITIONS=['clean','h264_500k','h264_250k','h264_1000k']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(x):return {'mean':statistics.mean(x),'sd':statistics.stdev(x),'values':x}
def item(r,subject,window):return next(x for x in r['records'] if x['subject']==subject and x['window']==window)
def audit_metrics(r):
    rows=r['records'];assert len(rows)==r['n']
    errors=np.array([abs(x['pred_hr']-x['gt_hr']) if x['pred_hr'] is not None else 135. for x in rows])
    assert all(abs(x['abs_error']-e)<1e-9 for x,e in zip(rows,errors))
    expected={'window_MAE':float(errors.mean()),'window_RMSE':float(np.sqrt(np.mean(errors**2))),
              'P95':float(np.percentile(errors,95,method='linear')),'max_error':float(errors.max()),
              'errors_over10':int((errors>10).sum()),'errors_over20':int((errors>20).sum())}
    for k,v in expected.items():assert abs(r[k]-v)<1e-8,(k,r[k],v)
    macro=statistics.mean(statistics.mean(x['abs_error'] for x in rows if x['subject']==s) for s in {x['subject'] for x in rows})
    assert abs(macro-r['subject_macro_MAE'])<1e-8
    assert r['invalid']==sum(x.get('invalid',False) for x in rows)

def diagnostic_plots(after=False):
    oldpath=BASE/'diagnostics_before';newpath=BASE/'diagnostics_after'
    chosen=[('subject11',5),('subject10',4),('subject32',0)]
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for row,(subject,window) in enumerate(chosen):
        z=np.load(oldpath/f'{subject}_w{window}.npz');f=z['bpm']
        for col,mode in enumerate(['control','augmentation']):
            ax=axes[row,col];ax.plot(f,z['target_psd'],color='black',ls='--',label='Reference PPG',lw=1.8)
            if after:
                nz=np.load(newpath/f'{subject}_w{window}.npz')
                for j,s in enumerate(SEEDS):
                    ax.plot(f,z[f'{mode}_{s}_h264_500k_psd'],color=['#4477aa','#ee6677','#aa3377'][j],alpha=.5,lw=1,ls=':')
                    ax.plot(f,nz[f'{mode}_{s}_h264_500k_psd'],color=['#4477aa','#ee6677','#aa3377'][j],label=f'PSD loss seed{s}',lw=1.3)
            else:
                for j,s in enumerate(SEEDS):ax.plot(f,z[f'{mode}_{s}_h264_500k_psd'],label=f'MSE seed{s}',lw=1.3)
            ax.set(title=f'{subject} window{window} | {mode}',xlabel='Frequency (bpm)',ylabel='Power / maximum',xlim=(45,180));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Frozen-model H264 spectra: solid = PSD loss, dotted = MSE' if after else 'Prior difficult windows: H264 spectra of the MSE models')
    fig.savefig(OUT/('spectra_before_after.png' if after else 'spectra_before.png'),dpi=160);plt.close(fig)

def main():
    done=read(BASE/'evaluation_complete.json');assert done['complete'] and done['models']==12 and not done['test_opened']
    frozen=read(BASE/'frozen_checkpoints.json');assert done['freeze_sha256']==sha(BASE/'frozen_checkpoints.json')
    evaluations={};runs={}
    for group in GROUPS:
        for seed in SEEDS:
            name=f'{group}_{seed}';ev=read(BASE/'frozen_evaluation'/(name+'.json'))
            assert ev['checkpoint']==frozen['checkpoints'][name] and ev['freeze_sha256']==done['freeze_sha256'] and not ev['test_opened']
            for r in list(ev['conditions'].values())+list(ev['train_conditions'].values()):audit_metrics(r)
            family,mode=group.split('_',1);p=(OLD if family=='mse' else BASE)/f'{mode}_{seed}'
            status=read(p/'status.json');prov=read(p/'provenance.json');hist=read(p/'history.json');v=read(p/'best_validation.json')
            assert status['state']=='complete' and status['reload_verified'] and len(hist)==30
            assert sha(p/'best.pt')==ev['checkpoint']['sha256']==status['checkpoint_sha256']
            assert v['epoch']==min(hist,key=lambda h:h['score'])['epoch']==ev['checkpoint']['best_epoch']
            for c in ['clean','h264_500k']:assert abs(ev['conditions'][c]['subject_macro_MAE']-v['conditions'][c]['subject_macro_MAE'])<1e-8
            evaluations[name]=ev;runs[name]={'status':status,'provenance':prov,'history':hist,'best':v}
    for mode in ['control','augmentation']:
        for seed in SEEDS:
            a=runs[f'mse_{mode}_{seed}'];b=runs[f'spectral_{mode}_{seed}']
            assert a['provenance']['initial_model_sha256']==b['provenance']['initial_model_sha256']
            assert all(x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips'] for x,y in zip(a['history'],b['history']))
            assert b['provenance']['protocol_sha256']==sha(OUT/'code/deep_hr/SPECTRAL_PROTOCOL.json')
            assert read(OUT/'code/deep_hr/SPECTRAL_PROTOCOL.json')==read(BASE/'protocol.json')
    aggregate={};training={}
    for group in GROUPS:
        aggregate[group]={};training[group]={}
        for cond in CONDITIONS:
            rr=[evaluations[f'{group}_{s}']['conditions'][cond] for s in SEEDS]
            aggregate[group][cond]={k:stats([r[k] for r in rr]) for k in ['subject_macro_MAE','window_RMSE','P95','max_error']}
            aggregate[group][cond].update({'pooled_over10':sum(r['errors_over10'] for r in rr),'pooled_over20':sum(r['errors_over20'] for r in rr),'pooled_invalid':sum(r['invalid'] for r in rr),'pooled_predictions':126})
        aggregate[group]['selection_score']=stats([runs[f'{group}_{s}']['best']['score'] for s in SEEDS])
        for cond in ['clean','h264_500k']:
            rr=[evaluations[f'{group}_{s}']['train_conditions'][cond] for s in SEEDS]
            training[group][cond]={'MAE':stats([r['subject_macro_MAE'] for r in rr]),'P95':stats([r['P95'] for r in rr]),
                 'pooled_over10':sum(r['errors_over10'] for r in rr),'pooled_over20':sum(r['errors_over20'] for r in rr),'pooled_predictions':483}
    decisions={}
    for mode in ['control','augmentation']:
        old=aggregate['mse_'+mode];new=aggregate['spectral_'+mode]
        gain=1-new['selection_score']['mean']/old['selection_score']['mean']
        clean_delta=new['clean']['subject_macro_MAE']['mean']-old['clean']['subject_macro_MAE']['mean']
        improved=sum(n<o for n,o in zip(new['h264_500k']['subject_macro_MAE']['values'],old['h264_500k']['subject_macro_MAE']['values']))
        p95delta=new['h264_500k']['P95']['mean']-old['h264_500k']['P95']['mean']
        countdelta=new['h264_500k']['pooled_over20']-old['h264_500k']['pooled_over20']
        decisions[mode]={'selection_score_improvement':gain,'clean_MAE_delta':clean_delta,'H264_improved_seeds':improved,'H264_P95_delta':p95delta,'H264_over20_count_delta':countdelta,
            'passes_development_gate':gain>=.10 and clean_delta<=.25 and improved>=2 and p95delta<=1 and countdelta<=0}
    diag_before=read(BASE/'diagnostics_before/results.json');diag_after=read(BASE/'diagnostics_after/results.json')
    for family,d in [('mse',diag_before),('spectral',diag_after)]:
        for record in d['records']:
            for name,m in record['models'].items():
                mode,seed,condition=name.split('_',2)
                ev=evaluations[f'{family}_{mode}_{seed}'];domain='train_conditions' if record['split']=='train' else 'conditions'
                r=item(ev[domain][condition],record['subject'],record['window'])
                assert abs(r['pred_hr']-m['hr'])<1e-8
    hard={}
    for name,ev in evaluations.items():hard[name]={c:item(r,'subject11',5) for c,r in ev['train_conditions'].items()}
    worst={}
    for group in GROUPS:
        for cond in CONDITIONS:
            base=evaluations[f'{group}_2027']['conditions'][cond]['records'];rows=[]
            for r in base:
                errs=[item(evaluations[f'{group}_{s}']['conditions'][cond],r['subject'],r['window'])['abs_error'] for s in SEEDS]
                rows.append({'subject':r['subject'],'window':r['window'],'start_s':r['start_s'],'gt_hr':r['gt_hr'],'mean_error':statistics.mean(errs),'seed_errors':errs})
            worst[group+'_'+cond]=sorted(rows,key=lambda r:r['mean_error'],reverse=True)
    write('window_error_rankings.json',worst)
    summary={'aggregate':aggregate,'training':training,'decisions':decisions,'hard_train_window':hard,'pairing_audit':frozen['pairing_audit'],
       'scope':'Development results on7 repeatedly used validation subjects;250k/1000k are unseen rates of the same people, not independent-subject test results.',
       'test_evaluated':False,'new_training_runs':6,'epochs_per_run':30,'protocol_sha256':sha(BASE/'protocol.json'),
       'checks':{'metrics_recomputed_from_all_records':True,'12_checkpoint_hashes_verified':True,'all_selected_checkpoints_reloaded':True,'paired_initialization_and_orders_verified':True,'waveform_diagnostics_match_saved_HR':True}}
    write('comparison.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for j,conds in enumerate([['clean','h264_500k'],['h264_250k','h264_1000k']]):
        for k,g in enumerate(GROUPS):
            values=[aggregate[g][c]['subject_macro_MAE']['mean'] for c in conds];errors=[aggregate[g][c]['subject_macro_MAE']['sd'] for c in conds]
            axes[j].bar(np.arange(2)+(k-1.5)*.19,values,width=.19,yerr=errors,label=LABELS[k],color=COLORS[k],capsize=2)
        axes[j].set(xticks=range(2),xticklabels=conds,ylabel='Subject-macro MAE (bpm)',title='Used for checkpoint selection' if j==0 else 'Unseen bitrate; frozen checkpoints');axes[j].grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Three-seed means; error bars are sample SD, not confidence intervals')
    fig.savefig(OUT/'factorial_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,metric,title in zip(axes,['P95','pooled_over20'],['95th percentile absolute error','Errors >20 bpm across 3 seeds']):
        for k,g in enumerate(GROUPS):
            vals=[aggregate[g][c][metric]['mean'] if metric=='P95' else aggregate[g][c][metric] for c in CONDITIONS]
            ax.plot(range(4),vals,'o-',label=LABELS[k],color=COLORS[k])
        ax.set(xticks=range(4),xticklabels=['Clean','500k','250k','1000k'],ylabel='bpm' if metric=='P95' else 'Count /126 predictions',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(OUT/'tail_errors.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j,c in enumerate(['clean','h264_500k']):
        axes[j].axhline(126.5625,color='black',ls='--',label='Reference126.56')
        for k,g in enumerate(GROUPS):axes[j].scatter(np.full(3,k)+np.linspace(-.1,.1,3),[hard[f'{g}_{s}'][c]['pred_hr'] for s in SEEDS],color=COLORS[k],s=45)
        axes[j].set(xticks=range(4),xticklabels=['MSE clean','MSE aug.','PSD clean','PSD aug.'],ylabel='Predicted HR (bpm)',title=c);axes[j].grid(axis='y',alpha=.15)
    axes[0].legend();fig.suptitle('Previously difficult training window: subject11,51.64–61.64s')
    fig.savefig(OUT/'hard_window.png',dpi=170);plt.close(fig)
    diagnostic_plots(after=True)
    aug_old=aggregate['mse_augmentation'];aug_new=aggregate['spectral_augmentation']
    rate_gain={c:1-aug_new[c]['subject_macro_MAE']['mean']/aug_old[c]['subject_macro_MAE']['mean'] for c in ['h264_250k','h264_1000k']}
    lines=['THÍ NGHIỆM LOSS PHỔ CÓ ĐỐI CHỨNG — 15/10/2026','',
      'KẾT LUẬN: Cả hai nhánh loss phổ chưa đạt tiêu chí phát triển đã chốt; chưa thay cấu hình chính. Nhánh kết hợp loss phổ và H.264 đáng giữ như ứng viên để kiểm chứng độc lập vì kết quả ở mức nén mới tốt hơn về trung bình.',
      f"So với MSE + H.264: MAE ở 250 kbps giảm {rate_gain['h264_250k']*100:.2f}%, ở 1 Mbps giảm {rate_gain['h264_1000k']*100:.2f}%. Tại 250 kbps, lỗi >20 bpm giảm từ {aug_old['h264_250k']['pooled_over20']} xuống {aug_new['h264_250k']['pooled_over20']} trên 126 dự đoán (42 đoạn × 3 seed). Không đổi checkpoint hoặc tiêu chí sau khi thấy các kết quả này.",
      'Đoạn train subject11 / 51,64–61,64 s được cải thiện mạnh ở 2/3 seed khi kết hợp loss phổ và augmentation, nhưng seed còn lại vẫn sai khoảng 58 bpm. Đây chưa phải giải pháp ổn định cho lỗi lớn.',
      '',
      'Đã hoàn tất6 lượt train mới ×30epoch, đối chiếu6 baseline MSE đã có; đánh giá12 checkpoint ở4 điều kiện validation. Không đánh giá test.',
      'Cùng kiến trúc EfficientPhys36, split28 train/7 validation, batch2, AdamW/OneCycleLR0.009, seed2027/2028/2029. Nhánh augmentation vẫn là xác suất50% clean/50% H264500k.',
      'Loss mới: MSE sai phân PPG +0.1 KL(phổ PPG tham chiếu || phổ dự đoán), trọng số cố định trước train. Tích phân đầu ra sai phân trước FFT; detrend lambda100, Butterworth0.75–3Hz, Hann, FFT4096. Không sửa estimator/nhãn/ROI.',
      'Tham khảo giám sát miền tần số của PhysFormer (CVPR2022): https://github.com/ZitongYu/PhysFormer/blob/main/TorchLossComputer.py . Đây là biến thể thí nghiệm dùng PSD của PPG thật, không phải tái lập Gaussian HR/softmax/curriculum của PhysFormer.',
      'Checkpoint: epoch sớm nhất đạt MAE trung bình sạch/H264500k thấp nhất. Hai rate250k và1000k chỉ đánh giá sau khi khóa hash cả12 checkpoint; không dùng chọn epoch/loss.',
      '', 'KẾT QUẢ VALIDATION: subject-macro MAE bpm, mean ± sampleSD qua3 seed (không phải confidence interval).']
    for g in GROUPS:lines.append(g+': '+ '; '.join(f"{c} {aggregate[g][c]['subject_macro_MAE']['mean']:.4f} ± {aggregate[g][c]['subject_macro_MAE']['sd']:.4f}" for c in CONDITIONS)+f"; selection {aggregate[g]['selection_score']['mean']:.4f}")
    lines+=['','QUYẾT ĐỊNH THEO GATE ĐÃ CHỐT (mỗi nhánh so với MSE cùng cách augmentation):']
    for mode,d in decisions.items():lines.append(f"{mode}: giảm selection score {d['selection_score_improvement']*100:.2f}%; clean delta {d['clean_MAE_delta']:+.4f}; H264 tốt hơn {d['H264_improved_seeds']}/3seed; P95 delta {d['H264_P95_delta']:+.4f}; >20bpm count delta {d['H264_over20_count_delta']:+d}; gate {'ĐẠT' if d['passes_development_gate'] else 'CHƯA ĐẠT'}.")
    lines+=['Gate: score giảm>=10%, clean MAE tăng<=0.25bpm, H264500k tốt hơn>=2/3seed; P95 H264 tăng không quá1bpm và số lỗi>20bpm không tăng. Đây là ngưỡng thực dụng đã định trước, không phải kiểm định ý nghĩa thống kê.','', 'LỖI ĐUÔI LỚN TRÊN VALIDATION:']
    for g in GROUPS:
        for c in CONDITIONS:
            a=aggregate[g][c];lines.append(f"{g}/{c}: P95 mean {a['P95']['mean']:.3f}; >10bpm {a['pooled_over10']}/126; >20bpm {a['pooled_over20']}/126; invalid {a['pooled_invalid']}/126.")
    lines+=['Các126 dự đoán là42 đoạn ×3seed, không phải126 người hay126 quan sát độc lập.','', 'TRAIN — chỉ chẩn đoán, không dùng chọn checkpoint:']
    for g in GROUPS:
        for c in ['clean','h264_500k']:
            a=training[g][c];lines.append(f"{g}/{c}: MAE {a['MAE']['mean']:.3f} ± {a['MAE']['sd']:.3f}; P95 {a['P95']['mean']:.3f}; >20bpm {a['pooled_over20']}/483.")
    lines+=['','CHẨN ĐOÁN ĐOẠN SUBJECT11,51.64–61.64s, GT126.56bpm:']
    for g in GROUPS:lines.append(g+': clean HR '+', '.join(f"{hard[f'{g}_{s}']['clean']['pred_hr']:.2f}" for s in SEEDS)+'; H264 HR '+', '.join(f"{hard[f'{g}_{s}']['h264_500k']['pred_hr']:.2f}" for s in SEEDS))
    lines+=['Nhãn tái tạo từ target huấn luyện lệch tham chiếu đánh giá tối đa0.43945bpm ở7 đoạn được kiểm tra; không đủ giải thích lỗi50–70bpm. Tương quan khi cho dịch pha±0.5s cũng không phục hồi toàn bộ waveform của baseline nén; dịch pha chỉ là phép chẩn đoán, không dùng sửa dự đoán.',
      'Kiểm tra timestamps/nhãn ở đây bảo đảm nhất quán pipeline và so sánh sạch-nén; chưa xác nhận đồng bộ vật lý tuyệt đối giữa camera và cảm biến, hay độ trễ truyền mạch. Cực đại tương quan có thể mơ hồ do tính tuần hoàn, nên không dùng nó làm ước lượng độ trễ thiết bị.',
      'ROI subject11 nhảy41pixel tại58.097s và26pixel tại59.115s. Các ảnh trước/sau cho thấy nghiêng đầu và cập nhật crop đột ngột; crop vẫn chứa mặt. Đây là yếu tố có thể tương tác với nén, chưa chứng minh nguyên nhân riêng lẻ. ROI giữ nguyên trong thí nghiệm để không trộn tác động với loss.',
      '', 'CÁC ĐOẠN VALIDATION KHÓ NHẤT SAU KHI KHÓA CHECKPOINT:']
    for g in GROUPS:
        for c in ['h264_500k','h264_250k','h264_1000k']:
            for r in worst[g+'_'+c][:2]:lines.append(f"{g}/{c}: {r['subject']} {r['start_s']:.2f}s, GT{r['gt_hr']:.2f}, mean error{r['mean_error']:.3f}, seed errors {', '.join(f'{e:.2f}' for e in r['seed_errors'])}.")
    lines+=['','KIỂM TRA VÀ GIỚI HẠN:',
      'Đã kiểm hash cache, timestamps/PPG/ROI/clip rows sạch-nén, khởi tạo và thứ tự mẫu/augmentation ở đủ30epoch từng cặp.12checkpoint hash khớp; tái nạp và tái lập metric sạch/500k. MAE/RMSE/P95 và số lỗi được tái tính từ mọi dự đoán; không bỏ invalid.',
      'Chỉ7 người validation đã dùng nhiều lần để phát triển.250k/1000k là bitrate mới trên cùng người/video, không phải test độc lập hay cross-dataset. Crop lấy từ video gốc nên chưa đánh giá detector trên video nén.',
      'Bước lưới FFT4096 ở30Hz là0.43945bpm, không phải độ phân giải sinh lý hay đảm bảo độ chính xác lâm sàng. Tham chiếu HR ở đây được suy từ PPG theo cùng bộ ước lượng đã cố định.',
      'Kết quả không đủ để suy ra thứ hạng challenge hay khả năng top3. Không điều chỉnh loss, checkpoint hay nhãn dựa vào kết quả holdout rate.',
      'Kết luận về loss chỉ áp dụng cho công thức PSD-KL và trọng số0.1 đã thử; không chứng minh mọi phương pháp giám sát tần số đều có hoặc không có ích. Cải thiện một đoạn train từng thấy nhãn không phải bằng chứng tổng quát hóa.',
      'Tệp: comparison.json, window_error_rankings.json, factorial_comparison.png, tail_errors.png, hard_window.png, spectra_before_after.png, roi_jump_contact_sheet.png; checkpoint/log/config trong results/spectral_ablation_v1.',
      'P100: /data_pool/khangle/facevitals/results/spectral_ablation_v1. A100 giữ video nén250k/1000k trong datasets/UBFC-rPPG.']
    body='\n'.join(lines)+'\n'
    for a,b in {'hoàn tất6':'hoàn tất 6','×30epoch':'× 30 epoch','đối chiếu6':'đối chiếu 6','có; đánh giá12':'có; đánh giá 12','ở4':'ở 4','trúc EfficientPhys36':'trúc EfficientPhys36','split28':'split 28','train/7':'train / 7','batch2':'batch 2','seed2027/2028/2029':'seed 2027/2028/2029','suất50%':'suất 50%','clean/50%':'clean / 50%','H264500k':'H.264 500k','+0.1':'+ 0.1','lambda100':'lambda 100','Butterworth0.75':'Butterworth 0.75','FFT4096':'FFT 4096','qua3':'qua 3','cả12':'cả 12','3seed':'3 seed','đủ30epoch':'đủ 30 epoch','12checkpoint':'12 checkpoint','Chỉ7':'Chỉ 7','tối đa0.43945':'tối đa 0.43945','ở7':'ở 7','lỗi50–70':'lỗi 50–70','nhảy41pixel':'nhảy 41 pixel','và26pixel':'và 26 pixel','tại58.097s':'tại 58.097 s','tại59.115s':'tại 59.115 s','qua3seed':'qua 3 seed','Các126':'Các 126','là42':'là 42','phải126':'phải 126','trên7':'trên 7','bằng42':'bằng 42','ở30Hz':'ở 30 Hz','là0.43945bpm':'là 0.43945 bpm','trọng số0.1':'trọng số 0.1','2/3seed':'2/3 seed','4điều':'4 điều'}.items():body=body.replace(a,b)
    (OUT/'REPORT_VI.txt').write_text(body,encoding='utf-8')
    print(json.dumps({'aggregate':{g:{c:aggregate[g][c]['subject_macro_MAE'] for c in CONDITIONS} for g in GROUPS},'decisions':decisions},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before-only',action='store_true');a=p.parse_args()
    if a.before_only:diagnostic_plots()
    else:main()
