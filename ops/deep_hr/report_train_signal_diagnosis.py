"""Summarize descriptive train-only signal diagnostics and timing gate."""
import json,statistics,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'reports/train_signal_diagnosis_20261019'
BASE=OUT/'results/train_compression_diagnosis_20261019';TIMING=OUT/'results/aipex_timing_gate_20261019'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def hit(s,gt):return any(abs(p['bpm']-gt)<=6 for p in s['top3'])
def descript(x):return {'n':len(x),'median':float(np.median(x)) if x else None,'q25':float(np.percentile(x,25)) if x else None,'q75':float(np.percentile(x,75)) if x else None}
def main():
    timing=read(TIMING/'result.json');result=read(BASE/'results.json');protocol=read(BASE/'protocol.json')
    assert result['complete'] and result['all_saved_HR_predictions_reproduced'] and not result['test_opened'] and not result['validation_opened']
    assert result['protocol_sha256']==sha(BASE/'protocol.json')
    rows=result['records'];assert len(rows)==161 and len({r['subject'] for r in rows})==28 and all(r['split']=='train' for r in rows)
    assert timing['complete'] and timing['decoded_videos']==18 and not timing['independent_model_benchmark_gate_passed']
    assert protocol['true_band_bpm_halfwidth']==6
    summary={'timing_gate_passed':False,'train_windows':161,'train_subjects':28,'families':{},'input':{},'all_frozen_predictions_reproduced':True,
       'validation_scope_note':'Shared clean-cache integrity verification hashes both train and validation files. No validation predictions, metrics or selection were performed; no test data used.'}
    for c in ['clean','h264_500k']:
        summary['input'][c]={'global_POS_top3_near_reference_count':sum(hit(r['input'][c]['pos'],r['gt_hr']) for r in rows),
            'any_patch_POS_top3_near_reference_count':sum(any(hit(p,r['gt_hr']) for p in r['input'][c]['patch_pos']) for r in rows),
            'global_POS_reference_band_fraction':descript([r['input'][c]['pos']['true_band_fraction'] for r in rows]),
            'oracle_best_patch_reference_band_fraction':descript([r['input'][c]['oracle_best_patch_true_band_fraction'] for r in rows])}
    cases=[]
    for family in ['mse','spectral']:
        allerrors={c:[] for c in ['clean','h264_500k']};severe=[];improved=[]
        for r in rows:
            for seed in [2027,2028,2029]:
                a=r['models'][f'{family}_{seed}_clean'];b=r['models'][f'{family}_{seed}_h264_500k']
                for c,m in [('clean',a),('h264_500k',b)]:
                    assert abs(m['error']-abs(m['hr']-r['gt_hr']))<1e-8;allerrors[c].append(m['error'])
                if a['error']<=5 and b['error']>20:
                    item={'family':family,'seed':seed,'subject':r['subject'],'window':r['window'],'gt_hr':r['gt_hr'],'clean_error':a['error'],'compressed_error':b['error'],
                       'compressed_hr':b['hr'],'compressed_model_trueband_fraction':b['true_band_fraction'],
                       'compressed_global_POS_top3_near_reference':hit(r['input']['h264_500k']['pos'],r['gt_hr']),
                       'compressed_any_patch_POS_top3_near_reference':any(hit(p,r['gt_hr']) for p in r['input']['h264_500k']['patch_pos']),
                       'compressed_global_POS_trueband_fraction':r['input']['h264_500k']['pos']['true_band_fraction'],
                       'compressed_oracle_best_patch_trueband_fraction':r['input']['h264_500k']['oracle_best_patch_true_band_fraction'],
                       'max_coordinate_step_px':r['input']['clean']['max_coordinate_step_px']}
                    severe.append(item);cases.append(item)
        summary['families'][family]={'conditions':{c:{'window_MAE_pooled':float(np.mean(es)),'errors_over20':sum(e>20 for e in es),'errors_over10':sum(e>10 for e in es)} for c,es in allerrors.items()},
           'clean_le5_compressed_gt20_count':len(severe),'severe_with_global_POS_top3_near_reference':sum(r['compressed_global_POS_top3_near_reference'] for r in severe),
           'severe_with_any_patch_POS_top3_near_reference':sum(r['compressed_any_patch_POS_top3_near_reference'] for r in severe),
           'severe_global_POS_trueband_fraction':descript([r['compressed_global_POS_trueband_fraction'] for r in severe]),
           'severe_oracle_best_patch_trueband_fraction':descript([r['compressed_oracle_best_patch_trueband_fraction'] for r in severe])}
    cases.sort(key=lambda r:r['compressed_error'],reverse=True)
    summary['limitations']=['Train-only descriptive analysis; labels used for diagnostic frequency bands, never inference selection.',
       'A nearby spectral peak is not proof of a physiological pulse; multiple patches and three peaks increase chance matches.',
       'Absence from POS peaks does not prove irreversible signal loss. No per-dataset FPS or model hyperparameters fitted.',
       'Cannot assign causal responsibility to quantization, chroma subsampling, tracking or motion from these comparisons.']
    control=read(OUT/'matching_control_result.json')
    assert control['protocol']['source_results_sha256']==sha(BASE/'results.json') and control['protocol']['halfwidth_bpm']==6
    summary['posthoc_background_match_control']=control
    (OUT/'verified_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'severe_train_cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    lines=['PHÂN TÍCH TÍN HIỆU TRAIN VÀ KIỂM TRA THỜI GIAN AIPEX — 2026-10-19',
      'Đã hoàn tất 6 bản ghi AiPEX/18 ROI và 161 cửa sổ train UBFC/28 người với 6 checkpoint đóng băng.',
      'AiPEX: kiểm tra container và cache đạt, nhưng chưa xác nhận được ánh xạ thời gian sinh lý. Không dùng kết quả trên AiPEX để xếp hạng mô hình.',
      'Nhãn AiPEX chỉ là HR trung bình mỗi bản ghi; dữ liệu phát hành thiếu timestamp thu nhận gốc và ánh xạ quy trình xuất video.',
      'Đã tái lập toàn bộ dự đoán HR train được lưu của 6 checkpoint. Không đánh giá validation/test, không huấn luyện hay chọn lại checkpoint. Bước xác minh checksum cache sạch có kiểm tra cả file train/validation trong inventory dùng chung.',
      'Định nghĩa lỗi tăng mạnh: clean <=5 bpm và H264 >20 bpm. Dải tham chiếu cố định: GT +/-6 bpm.',
      json.dumps(summary,ensure_ascii=False,indent=2),
      'Các trường hợp lỗi tăng mạnh (cửa sổ lặp qua seed không phải quan sát độc lập):',json.dumps(cases,ensure_ascii=False,indent=2),
      'Kết luận giới hạn: chỉ có thể định vị lỗi và mức năng lượng phổ gần nhịp tham chiếu; chưa đủ để khẳng định thông tin đã mất hoặc quyết định một khối sửa lỗi có tác dụng.',
      'Giữ baseline ROI cũ và hai nhóm checkpoint MSE+H264/spectral+H264. Chưa có bằng chứng độc lập để nâng một nhóm thành phương pháp cuối.']
    lines+=['ĐỐI CHỨNG HẬU NGHIỆM CHO VIỆC TÌM NHIỀU ĐỈNH PHỔ',
       'Dùng 1.000 lần gán HR từ người train khác, seed cố định 20271019; không chỉnh tham số. Đây là tỷ lệ trùng nền mô tả, không phải kiểm định tín hiệu sinh lý hay khoảng tin cậy.',
       'H264/global POS: 136/161 (84,47%) có top3 gần HR thật; gán HR người khác vẫn trùng trung bình 41,31%.',
       'H264/9 ô ảnh: 161/161 có một top3 gần HR thật; gán HR người khác vẫn trùng trung bình 95,74%.',
       'Do đó, phát hiện một đỉnh ở đâu đó trong 9 ô ảnh là tiêu chí quá dễ đạt, không đủ để biện minh cho chọn ROI bằng nhãn hoặc khẳng định multi-ROI sẽ cải thiện mô hình.',
       'Global POS cung cấp dấu hiệu mô tả hữu ích hơn; vẫn cần đối chứng mô hình và dữ liệu độc lập để xác nhận một cách sửa có hiệu quả.']
    (OUT/'REPORT_VI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for c,color,label in [('clean','#4477aa','Clean'),('h264_500k','#cc6677','H264 500k')]:
        v=[r['input'][c]['pos']['true_band_fraction'] for r in rows];axes[0].hist(v,bins=np.linspace(0,1,21),alpha=.55,color=color,label=label)
    axes[0].set(xlabel='Global POS power fraction within reference HR +/-6 bpm',ylabel='Train windows',title='Reference-conditioned diagnostic, not inference confidence');axes[0].legend()
    x=np.arange(2);w=.35
    for j,c in enumerate(['clean','h264_500k']):
        axes[1].bar(x+(j-.5)*w,[summary['families'][f]['conditions'][c]['errors_over20'] for f in ['mse','spectral']],w,label=c,color=['#4477aa','#cc6677'][j])
    axes[1].set_xticks(x,['MSE + H264','Spectral + H264']);axes[1].set(ylabel='Errors >20 bpm / 483 train predictions',title='Frozen models, three seeds; training data only');axes[1].legend()
    fig.savefig(OUT/'train_signal_diagnosis.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');x=np.arange(3);w=.36
    selected=[control['conditions']['clean']['global'],control['conditions']['h264_500k']['global'],control['conditions']['h264_500k']['nine_patches']]
    ax.bar(x-w/2,[r['actual_fraction']*100 for r in selected],w,label='Actual reference HR',color='#4477aa')
    ax.bar(x+w/2,[r['reassigned_fraction_mean']*100 for r in selected],w,label='Other-subject HR (mean, 1000 draws)',color='#cc6677')
    ax.set_xticks(x,['Clean / global POS','H264 / global POS','H264 / any of 9 patches'])
    ax.set(ylabel='Windows with one of top3 peaks within +/-6 bpm (%)',ylim=(0,120),title='Multiple-peak matching: post-hoc background control (train only)');ax.legend(fontsize=9)
    fig.savefig(OUT/'matching_control.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2));print('SEVERE CASES',json.dumps(cases[:5],indent=2))
if __name__=='__main__':main()
