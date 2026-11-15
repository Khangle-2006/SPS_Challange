"""Generate T2 report from measured outputs; missing conditions remain missing."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_t2_saved import STAGES,metrics,save,sha

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def f(v):return 'NA' if v is None else f'{v:.3f}'
def table(rows):
    lines=['| Dataset/model | Condition | Stage | Macro MAE | Window MAE | P95 | >20 | Coverage |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for name,cond,stage,m in rows:
        lines.append(f"| {name} | {cond} | {stage} | {f(m['subject_macro_MAE'])} | {f(m['window_MAE'])} | {f(m['P95'])} | {m['errors_over20']}/{m['n_total']} | {100*m['coverage']:.1f}% |")
    return lines

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out)
    matched=read(out/'matched_results.json');accept=read(out/'acceptance.json')
    ubfc=read(out/'ubfc_pos_results.json') if (out/'ubfc_pos_results.json').exists() else {}
    historic=read(out/'ubfc_historical_pos_results.json') if (out/'ubfc_historical_pos_results.json').exists() else {}
    mcd={c:read(out/f'mcd_{c}_results.json') for c in ['native','h264_500k'] if (out/f'mcd_{c}_results.json').exists()}
    lines=['# T2 — HR decoder v2: kết quả cố định tham số, 30/10/2026','',
      '**Kết luận: không đạt tiêu chí thay decoder. Giữ argmax đang dùng; không đưa harmonic/HMM/abstention v2 vào mặc định.** Sáu model đều không đạt kiểm tra clean +5% và giảm P95 500k ≥30%. Không dò lại β/κ/τ.','',
      '## Phạm vi và nguồn số liệu','',
      '- 6 checkpoint matched: MSE và spectral-KL, mỗi nhóm seed 2027/2028/2029. 4 điều kiện ×42 cửa sổ ×6 model =1008 waveform; 7 validation subject9/18/26/32/35/43/45. Mỗi cửa sổ 300 mẫu, 30Hz. Không mở train/test; không train.',
      '- So sánh ablation legacy → a → b → c → d → e. Các chỉ số bảng chính dùng mọi dự đoán hữu hạn; bảng accepted riêng thể hiện abstention. Mỗi hàng gộp sáu model có 252 lượt dự đoán theo seed/cửa sổ, không coi là 252 mẫu độc lập.',
      '- Metadata/split và số liệu lịch sử đã được xem trước khi freeze. Đây là validation đã dùng phát triển và chọn checkpoint, không phải đánh giá chưa chạm. Protocol mới được lưu trước khi chạy decoder mới, SHA '+sha(out/'protocol_frozen.json')+'.',
      '- Con số bối cảnh 12/203 và P95 28,7 không thuộc 42 cửa sổ validation của sáu model matched. Baseline matched 500k có 0/252 lỗi >20 bpm, nên không có mẫu số dương để chứng minh giảm 30% lỗi này. Không trộn 161 cửa sổ train với 42 cửa sổ validation.',
      '- Tất cả 1008 dự đoán legacy 0,75–3 Hz được tái lập chính xác từ waveform. hr_decoder.py hiện tại đã chuyển sang 0,7–3 Hz từ 20/10; legacy trong báo cáo là đối chứng tường minh, không mô tả sai mặc định hiện tại.','',
      '## Phương pháp đã khóa','',
      'β=0,5; κ=10; τ=2 (chọn trước từ 1,5/2/3). FFT 4096; smoothness detrend λ=100, Butterworth bậc 1 zero-phase như mã lịch sử. Giữ PSD đầy đủ tới Nyquist sau filter để đọc hài, vì PSD đã cắt 42–180 bpm không chứa năng lượng 4–5 Hz cần ở 10 fps. Giới hạn: filter HR làm yếu hài ngoài band; chưa có bằng chứng đây là lựa chọn tối ưu.',
      'a mở band 42–180 bpm. b chuẩn hóa PSD từng vùng theo năng lượng band, trọng số maxbin/bandenergy; với R=1 thì a=b. c dùng P(f)+0,5P(fold(2f)). HMM có grid 42..180 bước 1 bpm, Laplace 10 bpm/10 s, emission exp(10×L/maxL); reset giữa bản ghi, sau gap hỗ trợ và khi ba cửa sổ liên tiếp có peak mới cách >10 bpm vượt likelihood tại peak đang theo. τ dùng peak/floor riêng: PSD tại HR hậu nghiệm chia median PSD ngoài ±6 bpm; không nhầm với trọng số fusion [0,1]. Confidence chưa được hiệu chuẩn thành xác suất đúng.',
      'Causal dùng cửa sổ hoàn chỉnh rồi forward; không có cửa sổ tương lai, nhưng filter trong cửa sổ và POS không phải dự đoán từng frame tức thời. Offline thực sự forward-backward qua decode_sequence; fixedlag1 trả dự đoán trễ một cửa sổ, cửa sổ cuối chỉ forward. decode đơn cửa sổ ở mode offline không tự tạo context tương lai. API ghi rõ độ trễ.','',
      '## Bảng pooled sáu model — tất cả cửa sổ','']
    pooled=[]
    for cond in ['clean','h264_250k','h264_500k','h264_1000k']:
        for stage in STAGES:
            rows=sum([v[stage]['records'] for k,v in matched.items() if k.endswith('/'+cond)],[])
            pooled.append(('6 models',cond,stage,metrics(rows)))
    lines+=table(pooled)
    lines+=['','Chuyển .75→.7 đã thêm peak42–45bpm ở một số cửa sổ nén; phần harmonic thêm ca chọn sai. HMM phục hồi một phần nhưng không vượt baseline theo tiêu chí đã định. Đây là mô tả ablation, không chứng minh một cơ chế vật lý gây lỗi. τ=2 không từ chối bất kỳ cửa sổ model nào; coverage100% không có nghĩa không còn lỗi lớn.','',
      '## UBFC POS từ cache RGB đã lưu','',
      'Bốn điều kiện đều có cache 36×36; chỉ đọc bảy người validation. Cách trích POS tái dùng chẩn đoán subject32 đã có: nội suy 301 ảnh ở 30 Hz rồi lấy mean RGB, POS trên cửa sổ, giữ 300 mẫu. Đây là frontend model đã resize và POS theo cửa sổ; khác POS toàn bản ghi với tọa độ ROI native trong báo cáo cũ. Không gắn các con số mới cho baseline cũ.', '']
    lines+=table([('UBFC POS cache36',c,s,v[s]['all']) for c,v in ubfc.items() for s in STAGES])
    lines+=['','### POS lịch sử: waveform whole-recording đã lưu','',
      'Đối chứng trực tiếp dùng clean_pos/h264_pos đã lưu từ Haar native-coordinate ROI; không tính lại POS. Cùng 42 cửa sổ validation; cả 84 HR legacy tái lập khớp chính xác số lịch sử. Chỉ clean/500k có waveform gốc này. Phân biệt với cache 36×36 ở bảng trên; không gộp hai frontend thành một đường cong nén.','']
    lines+=table([('UBFC POS historical',c,s,v[s]['all']) for c,v in historic.items() for s in STAGES])
    lines+=['','## MCD FullHD POS','',
      '119 bản ghi/60 người dev; loại 4952 và 7412_after, giữ 9401. Native đã nén MPEG-4; 500k là nén thêm. T1 dùng timestamp thật, vùng PPG phủ và gap ≤0,2 s; nhãn PPG contact/raw giữ nguyên, không offset. RGB ba vùng FaceMesh đã có; POS mỗi vùng trên cửa sổ 10 s, grid 30 Hz. Legacy/a dùng trung bình PSD ba vùng; b+ dùng trọng số phổ. Vì vậy đây không phải bản sao baseline E0 vốn lấy mean RGB ba vùng trước POS.','']
    for c,v in mcd.items():
        lines += [f"{c}: {v['records']} bản ghi, {v['subjects']} người; {v['candidate_windows']} cửa sổ ứng viên T1, bỏ {v['support_rejected']} vì support/gap/ROI. Các tỷ lệ decoder bên dưới lấy mẫu số cửa sổ đã qua T1 và có reference HR; số loại trước decoder vẫn được báo riêng.",'']
        lines+=table([('MCD POS',c,s,v['stages'][s]['all']) for s in STAGES])
        lines += ['',f"Không tính được referenceHR: {v['stages']['legacy']['reference_failures']} cửa sổ.",'']
    if 'native' not in mcd:lines+=['Native đang chạy; chưa có bảng hoàn chỉnh119recording.','']
    if 'h264_500k' not in mcd:lines+=['500k chưa hoàn chỉnh vì E0 còn chạy. Không trình bày cohort một phần như đủ 119 bản ghi. Worker đang chờ và sẽ chạy replay cùng protocol sau khi E0 hoàn tất.','']
    lines+=['250k/1000k MCD chưa có trace lưu sẵn. T2 không encode lại dataset và không tạo waveform từ HR đơn lẻ hoặc PSD bị cắt band. Bảng bốn điều kiện MCD vì vậy chưa hoàn tất.','',
      '## Bảng accepted — không che số cửa sổ bị từ chối','']
    lines+=table([('UBFC POS',c,s,v[s]['accepted']) for c,v in ubfc.items() for s in ['legacy','e']]+[('MCD POS',c,s,v['stages'][s]['accepted']) for c,v in mcd.items() for s in ['legacy','e']])
    lines+=['','## Từng seed/model, mọi ablation','']
    lines+=table([(k.split('/')[0]+'/'+k.split('/')[1],k.split('/')[2],s,v[s]['all']) for k,v in matched.items() for s in STAGES])
    lines+=['','## Tiêu chí và kiểm tra','',
      'P95(500k) giảm ≥30%, số lỗi >20 giảm ≥30%, coverage ≥90%, clean MAE tăng ≤5%; cả sáu model phải đạt. Không model nào đạt. Baseline không có lỗi >20 khiến mức giảm tương đối không ước lượng được; không gọi 0→0 là giảm 30%. acceptance.json lưu từng điều kiện. POS cũng không đạt yêu cầu giảm tail ở 500k.',
      'Bảy unit test qua bằng python -m unittest trên P100: sin 1,5 Hz + nhiễu + hài ở 10/30 Hz; Nyquist fold; parity legacy/R1; HMM ổn định; tính nhân quả trên tiền tố; invalid mask; trọng số vùng. Test offline/fixedlag kiểm tra output hữu hạn và flag độ trễ. Các test không chứng minh cải thiện HR thật.',
      'Runner POS V1 lỗi tra đường dẫn file T1 khi tạo SHA trước bản ghi MCD đầu tiên. V1.1 tra qua module.__file__, không đổi dữ liệu, decoder hoặc protocol; code V1 và log được giữ. UBFC đã hoàn tất không chạy lại.','',
      'Đường cong coverage–accuracy chỉ để mô tả, không dùng chọn τ mới. Xếp hạng theo confidence từ dự đoán; nhãn chỉ dùng để tính sai số sau đó.','',
      '![Model coverage–accuracy](coverage_accuracy.png)','',
      '![POS coverage–accuracy](pos_coverage_accuracy.png)','',
      'Các đầu ra: matched_results.json; ablation.csv; acceptance.json; ubfc_pos_results.json; mcd_native_results.json (khi hoàn tất); mcd_h264_500k_results.json (khi có); tests.log; provenance.json; SHA256.json. Không thay inference_hr.py hoặc decoder mặcđịnh.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for ax,(dataset,conditions) in zip(axes,[('UBFC cache36',ubfc),('UBFC historical',historic),('MCD',{c:v['stages'] for c,v in mcd.items()})]):
        for condition,v in conditions.items():
            rows=sorted([r for r in v['e']['records'] if r['gt_hr'] is not None and np.isfinite(r['pred_hr'])],key=lambda r:r['confidence'],reverse=True)
            if not rows:continue
            errors=np.array([abs(r['pred_hr']-r['gt_hr']) for r in rows]);counts=np.arange(1,len(rows)+1)
            total=v['e']['all']['n_total'];ax.plot(counts/total,np.cumsum(errors)/counts,label=condition)
        ax.set(title=dataset+' POS',xlabel='Coverage (ranked confidence)',ylabel='Window MAE (bpm)');ax.grid(alpha=.2)
        if conditions:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'pos_coverage_accuracy.png',dpi=160);plt.close(fig)
    own=['decode_hr_v2.py','evaluate_t2_saved.py','evaluate_t2_pos.py','evaluate_t2_historical_pos.py','report_t2_decoder.py','test_decode_hr_v2.py','wait_t2_mcd500k.py','hr_decoder.py']
    hashes={n:sha(Path(__file__).with_name(n)) for n in own}
    for name in ['evaluate_t2_pos_v1.py','protocol_frozen.json','pos_replay_addendum.json','HISTORICAL_POS_REPLAY.txt']:
        if (out/name).exists():hashes['report/'+name]=sha(out/name)
    save(out/'SHA256.json',hashes)
    save(out/'status.json',dict(state='complete_available_conditions' if set(mcd)=={'native','h264_500k'} else 'partial_complete_pending_MCD500k',matched_models=6,matched_conditions=4,
        windows_per_model_condition=42,ubfc_pos_conditions=list(ubfc),mcd_pos_completed_conditions=list(mcd),
        ubfc_historical_pos_conditions=list(historic),mcd500k_watcher_pid=735186,
        accepted=False,default_decoder_unchanged=True,parameters_retuned=False,
        missing=['MCD250k/1000k saved traces unavailable']+([] if 'h264_500k' in mcd else ['MCD500k pending full E0 traces'])+([] if 'native' in mcd else ['MCDnative replay running']),
        report='REPORT.md'))

if __name__=='__main__':main()
