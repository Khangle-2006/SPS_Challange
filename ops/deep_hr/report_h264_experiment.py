"""Recompute metrics, audit paired runs, and render the H264 development report."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal,sparse
from scipy.sparse.linalg import spsolve

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/h264_experiment_20261015'
SEEDS=[2027,2028,2029]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def key(r):return (r['subject'],r['window'])
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(x):return {'mean':statistics.mean(x),'sample_sd':statistics.stdev(x),'values':x}
def spectrum(x):
    n=len(x);d=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    y=x-spsolve(sparse.eye(n,format='csc')+100**2*(d.T@d),x)
    y=signal.sosfiltfilt(signal.butter(1,[.75,3],btype='bandpass',fs=30,output='sos'),y)
    f,p=signal.periodogram(y,fs=30,window='hann',nfft=4096,detrend=False)
    b=(f>=.75)&(f<=3);return f[b]*60,p[b]/max(p[b].max(),1e-30)
def pos_analysis():
    pos=read(OUT/'pos_h264_results.json');rows=pos['records'];joined=[]
    originals={s:{key(r):r for r in read(HERE/f'reports/robustness_20261015/results/development_comparison_v1/original_{s}_h264_500k.json')['records']} for s in SEEDS}
    for r in rows:
        d=dict(r);d['original_learned_h264']={str(s):{k:originals[s][key(r)][k] for k in ('pred_hr','abs_error')} for s in SEEDS};joined.append(d)
    for condition,col in [('clean','clean_pos_error'),('h264','h264_pos_error')]:
        assert abs(statistics.mean(r[col] for r in rows)-pos[condition]['MAE'])<1e-10
    summary={c:{'MAE':pos[c]['MAE'],'RMSE':pos[c]['RMSE'],
        'signed_bias_bpm':statistics.mean(r[c+'_pos_hr']-r['gt_hr'] for r in rows),
        'large_underestimates':sum(r[c+'_pos_hr']<r['gt_hr']-10 for r in rows),
        'errors_over_10bpm':sum(r[c+'_pos_error']>10 for r in rows),
        'errors_over_20bpm':sum(r[c+'_pos_error']>20 for r in rows)} for c in ('clean','h264')}
    summary['new_errors_over_10bpm']=sum(r['clean_pos_error']<=10<r['h264_pos_error'] for r in rows)
    summary['n']=42;summary['subjects']=7
    save('matched_error_analysis.json',{'pos_summary':summary,'records':joined})
    chosen=[('subject9',3),('subject18',1),('subject32',0),('subject18',5)]
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    colors=['#30353b','#238b45','#c63d3d']
    for ax,k in zip(axes.flat,chosen):
        r=next(r for r in rows if key(r)==k);z=np.load(OUT/(r['subject']+'_traces.npz'))
        q=r['start_s']+np.arange(300)/30
        for name,label,color in zip(['ppg','clean_pos','h264_pos'],['Reference PPG','Clean POS','H264 POS'],colors):
            f,p=spectrum(np.interp(q,z['t'],z[name]));ax.plot(f,p,label=label,color=color,lw=1.5)
            expected=r['gt_hr' if name=='ppg' else name+'_hr'];assert abs(f[np.argmax(p)]-expected)<1e-8
        ax.axvline(r['gt_hr'],color='black',ls=':',alpha=.5)
        ax.set(title=f"{r['subject']} | {r['start_s']:.2f}–{r['start_s']+10:.2f} s",xlabel='Frequency (bpm)',ylabel='Power / within-signal maximum',xlim=(45,180))
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('H264 changes the dominant POS peak in matched validation windows',fontsize=14)
    fig.savefig(OUT/'pos_spectral_failures.png',dpi=170);plt.close(fig)
    return pos,joined,summary

def ablation(pos,joined,pos_summary):
    root=OUT/'results/h264_ablation_v1';protocol=read(root/'protocol.json');runs={};audit={}
    for mode in ['control','augmentation']:
        for seed in SEEDS:
            name=f'{mode}_{seed}';p=root/name;v=read(p/'best_validation.json');h=read(p/'history.json');status=read(p/'status.json');prov=read(p/'provenance.json')
            assert status['state']=='complete' and status['completed_epochs']==30 and status['reload_verified']
            assert len(h)==30 and v['epoch']==min(h,key=lambda r:r['score'])['epoch']
            assert abs(v['score']-statistics.mean(v['conditions'][c]['subject_macro_MAE'] for c in ['clean','h264_500k']))<1e-10
            if (p/'best.pt').exists():assert digest(p/'best.pt')==status['checkpoint_sha256']
            for c,r in v['conditions'].items():
                assert r['n']==42 and len(r['records'])==42
                errors=[abs(t['pred_hr']-t['gt_hr']) if t['pred_hr'] is not None else 135. for t in r['records']]
                assert all(abs(e-t['abs_error'])<1e-9 for e,t in zip(errors,r['records']))
                macro=statistics.mean(statistics.mean(t['abs_error'] for t in r['records'] if t['subject']==s) for s in {t['subject'] for t in r['records']})
                assert abs(statistics.mean(errors)-r['window_MAE'])<1e-9 and abs(macro-r['subject_macro_MAE'])<1e-9
                assert abs(np.sqrt(np.mean(np.square(errors)))-r['window_RMSE'])<1e-9
                assert r['invalid']==sum(t.get('invalid',False) for t in r['records'])
            runs[name]={'validation':v,'history':h,'status':status,'provenance':prov}
    for seed in SEEDS:
        c=runs[f'control_{seed}'];a=runs[f'augmentation_{seed}']
        assert c['provenance']['initial_model_sha256']==a['provenance']['initial_model_sha256']
        assert c['provenance']['code_sha256']==a['provenance']['code_sha256']
        assert c['provenance']['protocol_sha256']==a['provenance']['protocol_sha256']==digest(root/'protocol.json')
        assert all(ch['sample_order_sha256']==ah['sample_order_sha256'] for ch,ah in zip(c['history'],a['history']))
        cc=dict(c['provenance']['config']);ac=dict(a['provenance']['config'])
        for x in (cc,ac):x.pop('mode');x.pop('training_mix')
        assert cc==ac
        assert all(h['compressed_train_clips']==0 for h in c['history'])
        audit[str(seed)]={'same_initial_weights':True,'same_sample_order_all_30_epochs':True,'same_code_and_protocol':True,
             'augmentation_fraction':sum(h['compressed_train_clips'] for h in a['history'])/(30*161)}
    aggregate={m:{c:stats([runs[f'{m}_{s}']['validation']['conditions'][c]['subject_macro_MAE'] for s in SEEDS]) for c in ['clean','h264_500k']} for m in ['control','augmentation']}
    for m in aggregate:aggregate[m]['score']=stats([runs[f'{m}_{s}']['validation']['score'] for s in SEEDS])
    improvement=1-aggregate['augmentation']['score']['mean']/aggregate['control']['score']['mean']
    delta_clean=aggregate['augmentation']['clean']['mean']-aggregate['control']['clean']['mean']
    n_improved=sum(runs[f'augmentation_{s}']['validation']['conditions']['h264_500k']['subject_macro_MAE']<runs[f'control_{s}']['validation']['conditions']['h264_500k']['subject_macro_MAE'] for s in SEEDS)
    decision={'score_relative_improvement':improvement,'clean_mae_delta_bpm':delta_clean,'h264_improved_seeds':n_improved,
        'keep':improvement>=.10 and delta_clean<=.25 and n_improved>=2,
        'interpretation':'Development keep/reject rule only; validation used for selection, not an independent test.'}
    paired=[]
    for r in joined:
        d=dict(r);d['new_models']={}
        for name,run in runs.items():
            d['new_models'][name]={}
            for cond,v in run['validation']['conditions'].items():
                item=next(x for x in v['records'] if key(x)==key(r))
                assert abs(item['start_s']-r['start_s'])<1e-8 and item['gt_hr']==r['gt_hr']
                d['new_models'][name][cond]={k:item[k] for k in ['pred_hr','abs_error']}
        paired.append(d)
    save('paired_window_analysis.json',paired)
    changes=[]
    for r in paired:
        c=statistics.mean(r['new_models'][f'control_{s}']['h264_500k']['abs_error'] for s in SEEDS)
        a=statistics.mean(r['new_models'][f'augmentation_{s}']['h264_500k']['abs_error'] for s in SEEDS)
        changes.append({'subject':r['subject'],'start_s':r['start_s'],'gt_hr':r['gt_hr'],'control_error_mean':c,'augmentation_error_mean':a,'delta':a-c})
    changes.sort(key=lambda x:x['delta']);save('validation_window_change_ranking.json',changes)
    summary={'protocol':protocol,'aggregate':aggregate,'decision':decision,'pairing_audit':audit,
        'runs':{n:{'epoch':v['validation']['epoch'],'score':v['validation']['score'],'conditions':{c:{k:r[k] for k in ['subject_macro_MAE','window_RMSE','invalid']} for c,r in v['validation']['conditions'].items()},'checkpoint_sha256':v['status']['checkpoint_sha256']} for n,v in runs.items()},
        'pos':pos_summary,'test_evaluated_this_round':False}
    summary['training_epoch_seconds_total']=sum(h['elapsed_s'] for run in runs.values() for h in run['history'])
    train_pos=read(OUT/'pos_h264_training_results.json')
    assert train_pos['clean_POS_reproduced'] and len(train_pos['records'])==161 and train_pos['split']=='train'
    summary['train_pos']={c:{'window_MAE':train_pos[c]['MAE'],'subject_macro_MAE':statistics.mean(train_pos[c]['by_subject'].values()),
        'signed_bias_bpm':statistics.mean(r[c+'_pos_hr']-r['gt_hr'] for r in train_pos['records']),
        'large_underestimates':sum(r[c+'_pos_hr']<r['gt_hr']-10 for r in train_pos['records']),
        'RMSE':train_pos[c]['RMSE'],'errors_over_10bpm':sum(r[c+'_pos_error']>10 for r in train_pos['records'])} for c in ['clean','h264']}
    train_models={}
    for name,run in runs.items():
        tr=read(root/name/'train_diagnostics.json')
        assert tr['checkpoint_sha256']==run['status']['checkpoint_sha256'] and not tr['test_opened']
        for cond,data in tr['conditions'].items():
            assert data['n']==161
            es=[abs(x['pred_hr']-x['gt_hr']) if x['pred_hr'] is not None else 135. for x in data['records']]
            assert abs(statistics.mean(es)-data['window_MAE'])<1e-8
            assert abs(np.sqrt(np.mean(np.square(es)))-data['window_RMSE'])<1e-8
            macro=statistics.mean(statistics.mean(x['abs_error'] for x in data['records'] if x['subject']==s) for s in {x['subject'] for x in data['records']})
            assert abs(macro-data['subject_macro_MAE'])<1e-8
        train_models[name]=tr['conditions']
    summary['train_models']={m:{c:stats([train_models[f'{m}_{s}'][c]['subject_macro_MAE'] for s in SEEDS]) for c in ['clean','h264_500k']} for m in ['control','augmentation']}
    train_rows=[]
    for r in train_pos['records']:
        tr=dict(r);tr['new_models']={}
        for name,conditions in train_models.items():
            tr['new_models'][name]={}
            for cond,data in conditions.items():
                x=next(x for x in data['records'] if key(x)==key(r));assert x['gt_hr']==r['gt_hr']
                tr['new_models'][name][cond]={k:x[k] for k in ['pred_hr','abs_error']}
        train_rows.append(tr)
    save('paired_train_window_analysis.json',train_rows)
    save('comparison.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,cond,title in zip(axes,['clean','h264_500k','score'],['Clean validation','H264 validation','Selection score (50/50)']):
        for j,s in enumerate(SEEDS):
            y=[aggregate[m][cond]['values'][j] for m in ['control','augmentation']]
            ax.plot([0,1],y,'o-',label=f'Seed {s}')
        ax.set(xticks=[0,1],xticklabels=['Control','H264 aug.'],ylabel='Subject-macro MAE (bpm)',title=title,xlim=(-.15,1.15));ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Paired experiments: 3 seeds × 30 epochs per branch')
    fig.savefig(OUT/'paired_ablation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    subjects=list(pos['clean']['by_subject'])
    for ax,cond in zip(axes,['clean','h264_500k']):
        for m,color,dx in [('control','#4477aa',-.18),('augmentation','#44aa77',.18)]:
            values=[statistics.mean(r['new_models'][f'{m}_{s}'][cond]['abs_error'] for s in SEEDS for r in paired if r['subject']==sub) for sub in subjects]
            ax.bar(np.arange(7)+dx,values,width=.36,label=m,color=color)
        ax.set(xticks=range(7),xticklabels=[s.replace('subject','S') for s in subjects],ylabel='MAE (bpm)',title=cond);ax.legend();ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'subject_errors.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained',sharex=True)
    for j,s in enumerate(SEEDS):
        for m,color in [('control','#4477aa'),('augmentation','#44aa77')]:
            run=runs[f'{m}_{s}'];h=run['history'];epochs=[x['epoch'] for x in h]
            axes[0,j].plot(epochs,[x['train_mse'] for x in h],label=m,color=color)
            axes[1,j].plot(epochs,[x['score'] for x in h],label=m,color=color)
            axes[1,j].scatter([run['validation']['epoch']],[run['validation']['score']],color=color,marker='*',s=100,zorder=5)
        axes[0,j].set(title=f'Seed {s}',ylabel='Train MSE')
        axes[1,j].set(xlabel='Epoch',ylabel='Validation selection MAE (bpm)',yscale='log')
        for ax in axes[:,j]:ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('All 30 epochs; stars mark validation-selected checkpoints')
    fig.savefig(OUT/'learning_curves.png',dpi=170);plt.close(fig)
    lines=['THÍ NGHIỆM POS VÀ AUGMENTATION H.264 — 15/10/2026','',
        'Kết luận theo tiêu chí đã chốt: '+('GIỮ augmentation để kiểm tra tiếp.' if decision['keep'] else 'CHƯA ĐƯA augmentation này vào pipeline chính; lưu lại như một ứng viên có cải thiện nhỏ.'),
        f"Điểm chọn checkpoint giảm {improvement*100:.2f}%; thay đổi clean MAE {delta_clean:+.4f} bpm; H264 tốt hơn ở {n_improved}/3 seed.",
        '', 'Thiết kế: 28 người train (161 đoạn); 7 người validation (42 đoạn), cùng split. 6 lượt train, mỗi lượt 30 epoch. Không mở lại test.',
        'EfficientPhys36; 301 frame/30 Hz, batch 2, MSE PPG sai phân, AdamW, OneCycleLR max 0.009. Cùng seed, khởi tạo và thứ tự mẫu trong từng cặp.',
        'Control dùng dữ liệu sạch. Augmentation lấy mẫu Bernoulli 50% sạch / 50% video H264 thực ở từng clip/epoch. Mã hóa full-frame libx264, yuv420p, preset fast, target/maxrate 500k, bufsize 1000k.',
        'Giữ nguyên timestamps, PPG và tọa độ ROI gốc. Nén trước khi crop/resize. Chọn checkpoint theo trung bình MAE validation sạch/H264 với trọng số bằng nhau; lấy epoch sớm nhất khi hòa.',
        'Tiêu chí giữ: giảm điểm trung bình >=10%; clean MAE tăng <=0.25 bpm; H264 tốt hơn >=2/3 seed. Tiêu chí được lưu trước khi có kết quả.',
        '', 'KẾT QUẢ (bpm; mean ± sample SD qua 3 seed, SD không phải khoảng tin cậy):']
    for m in aggregate:
        lines.append(m+': '+ '; '.join(f"{c} {aggregate[m][c]['mean']:.4f} ± {aggregate[m][c]['sample_sd']:.4f}" for c in aggregate[m]))
    lines+=['','TỪNG SEED:']
    for n,v in summary['runs'].items():lines.append(f"{n}, best epoch {v['epoch']}: clean {v['conditions']['clean']['subject_macro_MAE']:.4f}; H264 {v['conditions']['h264_500k']['subject_macro_MAE']:.4f}; score {v['score']:.4f}; invalid {sum(c['invalid'] for c in v['conditions'].values())}")
    lines+=['','POS VÀ CÁC ĐOẠN LỖI:',
        f"POS clean MAE {pos['clean']['MAE']:.4f}, RMSE {pos['clean']['RMSE']:.4f}; H264 MAE {pos['h264']['MAE']:.4f}, RMSE {pos['h264']['RMSE']:.4f}. Không loại mẫu invalid; H264 invalid=0.",
        f"Số đoạn POS lỗi >10 bpm: sạch {pos_summary['clean']['errors_over_10bpm']}, nén {pos_summary['h264']['errors_over_10bpm']}; {pos_summary['new_errors_over_10bpm']} lỗi mới vượt 10 bpm.",
        'POS sạch được tái tính từ video gốc và khớp chính xác cả 42 đoạn trước khi diễn giải tác động nén.',
        'Các trường hợp dưới đây dùng nhãn chỉ để chẩn đoán, không sửa dự đoán/estimator theo nhãn:']
    for subject,window in [('subject9',3),('subject18',1),('subject32',0),('subject18',5),('subject18',2)]:
        r=next(r for r in paired if key(r)==(subject,window))
        lines.append(f"{subject} {r['start_s']:.2f}–{r['start_s']+10:.2f}s: GT {r['gt_hr']:.2f}; POS sạch {r['clean_pos_hr']:.2f}, H264 {r['h264_pos_hr']:.2f}; gap lớn nhất {r['max_gap_s']:.3f}s.")
        for m in ['control','augmentation']:
            lines.append('  '+m+' H264 HR (seed 2027/2028/2029): '+', '.join(f"{r['new_models'][f'{m}_{s}']['h264_500k']['pred_hr']:.2f}" for s in SEEDS))
    lines+=['Nén làm đổi tương quan công suất các đỉnh phổ: ở subject9/31.60s và subject18/11.60s, thành phần tần số thấp vốn có trước nén trở thành đỉnh lớn nhất. Đây là quan sát phổ, chưa chứng minh nguyên nhân cơ học riêng lẻ (quantization/chroma/motion).',
        'RGB sạch và nén tương quan cao không đảm bảo thành phần mạch đập nhỏ còn nguyên. Không gọi mọi đỉnh thấp là hài 1/2: 70 bpm không bằng nửa 116 bpm.',
        'Không thể quy hết cho mất frame: subject18/11.60s có gap 48 ms, còn gap ở subject9 tồn tại cả trước nén. Không khẳng định thông tin đã mất hoàn toàn vì một số mô hình vẫn dự đoán đúng cùng đoạn.',
        'subject32/1.61s đã khó trước nén; PPG có đỉnh phụ mạnh khoảng 99 bpm ngoài đỉnh 88 bpm. Cần kiểm tra nhịp biến thiên/chất lượng tham chiếu, không tự sửa nhãn.',
        'subject18/21.59s là lỗi POS sẵn có trước nén (GT ~128, POS ~60), phải phân biệt với lỗi mới do nén.',
        f"POS H264 có bias {pos_summary['h264']['signed_bias_bpm']:.3f} bpm trên validation; {pos_summary['h264']['large_underestimates']}/{pos_summary['h264']['errors_over_10bpm']} lỗi >10 bpm là dự đoán thấp hơn GT. Trên train: bias {summary['train_pos']['h264']['signed_bias_bpm']:.3f} bpm; {summary['train_pos']['h264']['large_underestimates']}/{summary['train_pos']['h264']['errors_over_10bpm']} lỗi lớn là dự đoán thấp.",
        '', 'KIỂM TRA POS TRÊN TRAIN (161 đoạn, 28 người; không dùng để đổi cấu hình thí nghiệm):',
        *[f"{c}: window MAE {v['window_MAE']:.4f}; subject-macro MAE {v['subject_macro_MAE']:.4f}; RMSE {v['RMSE']:.4f}; {v['errors_over_10bpm']}/161 đoạn lỗi >10 bpm." for c,v in summary['train_pos'].items()],
        'Lỗi các checkpoint trên train (subject-macro MAE, mean ± SD qua 3 seed; chỉ chẩn đoán, không đổi checkpoint):',
        *[m+': '+ '; '.join(f"{c} {v['mean']:.4f} ± {v['sample_sd']:.4f}" for c,v in conditions.items()) for m,conditions in summary['train_models'].items()],
        '', 'KIỂM TRA VÀ GIỚI HẠN:',
        'Mọi cache được kiểm SHA-256/kích thước trước train; timestamps/PPG/ROI/clip rows sạch-nén được đối chiếu bằng nhau. Cùng khởi tạo, code, protocol và thứ tự shuffle đủ 30 epoch trong từng cặp. Cả 6 checkpoint được nạp lại và tái xác nhận validation MAE; báo cáo tái tính MAE/RMSE từ từng dự đoán.',
        'Validation đã được dùng để chọn checkpoint và ra quyết định nhiều lần. Kết quả là development evidence trên 7 người UBFC; không phải test độc lập, không chứng minh tổng quát hóa sang challenge hay khả năng top 3.',
        'Chỉ thử một recipe H264 ~500 kbps. ROI lấy từ video gốc nên chưa đo lỗi detector trên video nén. Bitrate file thực có thể nhỉnh hơn 500 kbps vì overhead/rate control.',
        'Dữ liệu SpO2/BP và bộ challenge vẫn cần xin quyền; kết quả này chỉ là HR.',
        '', 'Tệp kèm: comparison.json, paired_window_analysis.json, matched_error_analysis.json, pos_spectral_failures.png, paired_ablation.png, subject_errors.png; các run/checkpoint trong results/h264_ablation_v1.',
        'Server P100: /data_pool/khangle/facevitals/results/h264_ablation_v1',
        'Server A100: /data/khangle/facevitals/results/pos_h264_validation_v1; datasets/UBFC-rPPG/training_h264_500k_v1']
    lines+=['','THAY ĐỔI LỖI H264 THEO ĐOẠN VALIDATION (MAE trung bình qua 3 seed):','Ba thay đổi tốt nhất:']
    for r in changes[:3]:lines.append(f"{r['subject']} {r['start_s']:.2f}s, GT {r['gt_hr']:.2f}: control {r['control_error_mean']:.3f} -> augmentation {r['augmentation_error_mean']:.3f}; delta {r['delta']:+.3f} bpm.")
    lines+=['Ba thay đổi xấu nhất:']
    for r in reversed(changes[-3:]):lines.append(f"{r['subject']} {r['start_s']:.2f}s, GT {r['gt_hr']:.2f}: control {r['control_error_mean']:.3f} -> augmentation {r['augmentation_error_mean']:.3f}; delta {r['delta']:+.3f} bpm.")
    def train_error(r,m):return statistics.mean(r['new_models'][f'{m}_{s}']['h264_500k']['abs_error'] for s in SEEDS)
    lines+=['','Bốn đoạn train khó nhất theo MAE H264 lớn hơn giữa hai nhánh (trung bình seed; chọn chỉ để chẩn đoán):']
    for r in sorted(train_rows,key=lambda r:max(train_error(r,'control'),train_error(r,'augmentation')),reverse=True)[:4]:
        lines.append(f"{r['subject']} {r['start_s']:.2f}s, GT {r['gt_hr']:.2f}: POS sạch {r['clean_pos_hr']:.2f}, POS H264 {r['h264_pos_hr']:.2f}; control error {train_error(r,'control'):.3f}, augmentation error {train_error(r,'augmentation'):.3f}; max gap {r['max_gap_s']:.3f}s.")
    hard=next(r for r in train_rows if key(r)==('subject11',5))
    lines+=['','Lỗi chưa giải quyết cần ưu tiên: subject11 51.64–61.64s, GT 126.56 bpm.',
        'Augmentation dự đoán H264 theo seed 2027/2028/2029: '+', '.join(f"{hard['new_models'][f'augmentation_{s}']['h264_500k']['pred_hr']:.2f}" for s in SEEDS)+' bpm; cùng checkpoint trên clip sạch đều sai dưới 0.44 bpm.',
        'PPG tham chiếu có đỉnh chính rõ 126.56 bpm; POS sạch 124.37 bpm. POS H264 còn đỉnh phụ 123.93 bpm với công suất bằng khoảng 92% đỉnh mạnh nhất 86.13 bpm. Vì vậy không có cơ sở coi clip này hoàn toàn mất thông tin HR; mạng vẫn chọn thành phần sai.',
        'Control seed 2028 trên H264 chỉ sai 5.27 bpm, trong khi augmentation cùng seed sai 55.37 bpm. Đây là regression quan sát được, không được che bởi MAE trung bình giảm.',
        'Kết luận về nguyên nhân mới dừng ở việc nén thay đổi biểu diễn/đỉnh phổ và mô hình chưa chọn được thành phần đúng; chưa xác lập riêng vai trò lượng tử hóa, chroma subsampling, chuyển động hay hàm loss. Không tự sửa nhãn hoặc chọn đỉnh bằng GT.']
    (OUT/'REPORT_VI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'aggregate':aggregate,'decision':decision,'audit':audit},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pos-only',action='store_true');a=p.parse_args()
    pos,joined,summary=pos_analysis()
    if a.pos_only:print(json.dumps(summary,indent=2))
    else:ablation(pos,joined,summary)
