"""Read-only reporting of frozen BTC OOF outputs; no fit or threshold selection."""
import ast, csv, json, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
P = Path(__file__).resolve().parent
def read(name): return json.loads((P/name).read_text(encoding='utf-8-sig'))
# Use the exact already-tested metric definition without importing Linux workers.
source = (P/'evaluate.py').read_text(encoding='utf-8-sig')
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'summarize')
exec(compile(ast.Module(body=[node], type_ignores=[]), str(P/'evaluate.py'), 'exec'))
rows = list(csv.DictReader((P/'VIDEO_PREDICTIONS.csv').open(encoding='utf-8-sig')))
rows += list(csv.DictReader((P/'BTC_V4_SUPPLEMENT.csv').open(encoding='utf-8-sig')))
for r in rows:
    for k in ('truth','prediction','lower90','upper90','sqi'): r[k] = float(r[k]) if r.get(k) else None
    r['fold'] = int(r['fold'])
manifest = json.loads((P.parent/'round1_v4_20261105/btc_zeroshot/dev_manifest.json').read_text(encoding='utf-8-sig'))
by = {v['recording']: v for v in manifest['videos']}
methods = ['constant','POS_old_argmax','CHROM_old','UBFC_median15','V3_median15','V4_original_padding_median15','calibrated_median_POS_CHROM_V3','Bayes_none','Bayes_constant_epsilon']
labels = dict(zip(methods,['Hằng số từ train-fold','POS cũ','CHROM cũ','Chỉ UBFC (trung vị 15 checkpoint)','V3 (trung vị 15 checkpoint)','V4 padding cũ (trung vị 15 checkpoint)','Trung vị POS/CHROM/V3 + hiệu chỉnh','Bayes đầy đủ','Bayes với epsilon hằng']))
def fmt(x): return '—' if x is None else f'{x:.2f}'
def ci(x): return f'[{x[0]:.2f}; {x[1]:.2f}]'
def table(header, rr): return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rr])
groups = {'all':set(by)}
for v in by.values():
    activity=v['group']+'-Resting' if v['task'].lower()=='resting' else 'Walking' if v['task'].lower()=='walking' else 'Task'
    for g in [activity,'skin='+v['skin'],'lighting='+v['lighting'],'resolution='+v['media']['width']+'x'+v['media']['height'],'HDR='+str(v['media']['transfer'] in ['arib-std-b67','smpte2084']),'rotation='+v['media']['rotation']]: groups.setdefault(g,set()).add(v['recording'])
results=[];coverage=[];availability=[]
snr=read('BLIND_SNR_VIDEO.json');snr_groups={};snr_edges={}
for cond in ('native','c5'):
    rr=[r for r in snr['rows'] if r['condition']==cond];values=[r['snr_db'] for r in rr if r['snr_db'] is not None];edges=np.quantile(values,[.25,.5,.75]);snr_edges[cond]=edges.tolist();snr_groups[cond]=defaultdict(set)
    for r in rr:snr_groups[cond]['SNR unknown' if r['snr_db'] is None else 'SNR Q'+str(1+np.searchsorted(edges,r['snr_db']))].add(r['recording'])
for cond in ('native','c5'):
    lookup={m:{r['recording']:r for r in rows if r['condition']==cond and r['method']==m} for m in methods}
    keys=set.intersection(*(set(x) for x in lookup.values()))
    for v in by.values():
        missing=[m for m in methods if v['recording'] not in lookup[m]]
        availability.append(dict(condition=cond,recording=v['recording'],person=v['person'],included_all_methods=not missing,missing_methods=';'.join(missing)))
    coverage.append(dict(condition=cond,common_videos=len(keys),common_people=len({by[k]['person'] for k in keys}),dev_videos=221,dev_people=161))
    for group, gkeys in (groups | snr_groups[cond]).items():
        selected=keys&gkeys
        if not selected: continue
        constant=[lookup['constant'][k] for k in sorted(selected)]
        pos=[lookup['POS_old_argmax'][k] for k in sorted(selected)]
        for m in methods:
            rr=[lookup[m][k] for k in sorted(selected)]
            results.append(dict(condition=cond,group=group,method=m,**summarize(rr),delta_constant=summarize(rr,constant),delta_POS=summarize(rr,pos)))
(P/'BTC_COMMON_METRICS.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf8')
with (P/'BTC_COVERAGE_BY_VIDEO.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(availability[0]));w.writeheader();w.writerows(availability)
compact=[]
for r in results:
    compact.append({k:r[k] for k in ['condition','group','method','people','videos','MAE','RMSE_mean_person','bias','Pearson_person_means']}|{'MAE_CI_low':r['CI95'][0][0],'MAE_CI_high':r['CI95'][0][1],'delta_constant':r['delta_constant']['MAE'],'delta_CI_low':r['delta_constant']['CI95'][0][0],'delta_CI_high':r['delta_constant']['CI95'][0][1]})
with (P/'BTC_COMMON_METRICS.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(compact[0]));w.writeheader();w.writerows(compact)
lines=['# BTC: kết quả OOF của pipeline Bayes, ngày 09/11/2026','',
'**Kết luận: chưa thay ứng viên hiện tại.** Bayes đầy đủ không vượt hằng số một cách có ý nghĩa thống kê; SQI cũng không đạt điều kiện sai số giảm đều đã đăng ký. Giữ nguyên kết quả, không chỉnh ngưỡng hay fit lại theo điểm validation. Holdout chưa được mở.',
'','## Đọc bảng như thế nào','',
'Native là video BTC gốc; C5 là bản nén x264 cũ đã khóa. Kiểm soát G4 không kích hoạt, nên không tạo C5′. Một video chỉ có một nhãn PR của BTC. Mỗi người được giữ hoàn toàn ngoài tập fit của fold kiểm tra; người có nhiều video không được tính nặng hơn người khác.',
'', 'MAE là độ lệch tuyệt đối trung bình, đơn vị bpm (nhịp/phút), càng thấp càng tốt. Tính trung bình lỗi các video của từng người trước, rồi trung bình giữa người. Δ = MAE phương pháp − MAE hằng số trên **đúng cùng video**: âm là tốt hơn. CI95% được bootstrap ghép cặp 10.000 lần theo người, seed 0; CI chứa 0 nghĩa là chưa đủ bằng chứng khác biệt.',
'', 'Các bảng chính dưới đây chỉ lấy giao của tất cả tám phương pháp. Không so số từ cohort 205 video trước với số 218 video mới như một cải thiện của thuật toán. Phương pháp cũ lấy trung vị các cửa sổ 10 giây; Bayes mới dùng phổ toàn video và Viterbi nếu đủ dài. Đây là so sánh các pipeline hoàn chỉnh, không phải thí nghiệm chỉ thay một thành phần.',
'','## 1. Kết quả chính','']
for cond in ('native','c5'):
    c=next(x for x in coverage if x['condition']==cond)
    lines += [f"### {cond}: {c['common_videos']}/221 video, {c['common_people']}/161 người",'',table(['Phương pháp','MAE','CI95% MAE','Δ so hằng số','CI95% Δ'],[[labels[r['method']],fmt(r['MAE']),ci(r['CI95'][0]),fmt(r['delta_constant']['MAE']),ci(r['delta_constant']['CI95'][0])] for r in results if r['condition']==cond and r['group']=='all']),'']
lines += ['Bản hiệu chỉnh tuyến tính có MAE native thấp nhất trong các phương án liệt kê, nhưng CI ghép cặp trên cohort mở rộng vẫn chứa 0. Không giữ nguyên kết luận có ý nghĩa thống kê của báo cáo 205 video khi mẫu đã đổi.','', '## 2. Các cổng và chất lượng đầu vào','']
g1=read('G1.json');g2=read('G2.json')
lines += [table(['Kiểm tra','Kết quả','Ý nghĩa'],[
 ['G0 holdout','0 lần','72 video/41 người vẫn khóa; chỉ đánh giá một lần ngày 26–27/11'],
 ['G1 loader',f"PASS: {g1['videos_face95']}/221 video có ≥95% khung thấy mặt; 0 lỗi decode",'Thấy mặt không đồng nghĩa tín hiệu mạch đủ tốt'],
 ['G2 thắng hằng số','FAIL','CI95% Δ native chứa 0'],
 ['SQI native và C5','FAIL cả hai','Không dùng phiên bản phụ thuộc SQI này làm cấu hình đã được chấp nhận'],
 ['G3 fusion học sâu','Chưa chạy','Không có sweep GPU mới hoặc kết luận fusion đạt cổng'],
 ['G4 codec','Không kích hoạt','Cả ba flat-field POS bằng 0; không chứng minh mọi video C5 sạch artifact']]),'']
for cond in ('native','c5'):
    v=read('SQI_VALIDITY_'+cond+'.json')
    lines += [f"MAE theo bốn nhóm SQI tăng dần ({cond}): "+' → '.join(fmt(x) for x in v['MAE_quartiles'])+' bpm. Nhóm cao nhất tốt hơn nhưng các nhóm giữa không giảm đều; không hạ yêu cầu sau khi thấy số.','']
qc=g1['details'];qct=[]
for tag in ['Normal-Resting','Abnormal-Resting','Task','Walking']:
    vs=[r for r in qc if r['recording'] in groups.get(tag,set())]
    if vs:qct.append([tag,len(vs),fmt(np.mean([r['face_fraction'] for r in vs])*100),*[fmt(np.mean([r['roi_valid'][j] for r in vs])*100) for j in range(3)]])
lines += [table(['Nhóm','Video','Thấy mặt %','Trán hợp lệ %','Má trái %','Má phải %'],qct),'','Phần trăm trên là trung bình tỷ lệ khung của từng video, dùng cho QC, không phải metric chính theo người. ROI hợp lệ cần đủ pixel qua mask da/sáng; không suy ra khác biệt sinh lý từ bảng này.','']
for cond in ('native','c5'):
    rr=[r for r in rows if r['condition']==cond and r['method']=='Bayes_none'];n=sum(r['pulse_observed']=='False' for r in rr)
    fit=[r for r in read('FITS.json') if r['condition']==cond and r['ablation']=='none']
    lines += [f"{cond}: có {len(rr)} video Bayes được chấm; {n} video không có ứng viên mạch nên trả về prior. `pulse_observed=true` chỉ có nghĩa tồn tại ứng viên phổ, **không chứng minh ứng viên là mạch thật**.", '',table(['Fold','a','b của SQI','Sigma (bpm)'],[[r['fold'],fmt(r['a']),fmt(r['b']),fmt(r['sigma'])] for r in fit]),'']
lines += ['a và b điều khiển xác suất outlier ε = sigmoid(a + b·SQI); sigma là độ rộng Gaussian của likelihood. Đây là hệ số học trên train-fold, không phải đo sinh lý. Giá trị b gần 0 nghĩa là fit ít sử dụng SQI.','', '## 3. Resting phải tách Normal và Abnormal','']
for cond in ('native','c5'):
    for group in ['Normal-Resting','Abnormal-Resting','Task','Walking']:
        rr=[r for r in results if r['condition']==cond and r['group']==group]
        if rr: lines += [f'### {cond} — {group}', '',table(['Phương pháp','Người/video','MAE','CI95%','Δ so hằng số'],[[labels[r['method']],f"{r['people']}/{r['videos']}",fmt(r['MAE']),ci(r['CI95'][0]),fmt(r['delta_constant']['MAE'])] for r in rr]),'']
lines += ['## 4. Ablation và độ tin cậy','', 'Ablation tắt một thành phần rồi fit lại trên train-fold; Δ âm nghĩa là bản bỏ thành phần tốt hơn bản đầy đủ. Đây là phân tích đã đăng ký, không dùng để chọn thêm tham số.', '',table(['Điều kiện','Bỏ/thay thành phần','Δ MAE','CI95% Δ'],[[r['condition'],r['ablation'],fmt(r['delta_vs_full']['MAE']),ci(r['delta_vs_full']['CI95'][0])] for r in read('ABLATION.json')]),'',table(['Điều kiện','Người','Độ phủ khoảng posterior 90%','CI95% độ phủ'],[[r['condition'],r['people'],f"{r['coverage90']*100:.1f}%",ci([x*100 for x in r['CI95']])+'%'] for r in read('INTERVAL_COVERAGE.json')]),'', 'Khoảng posterior có nhãn “90%” không mặc nhiên chứa PR thật trong 90% trường hợp. Bảng đo độ phủ thực tế theo người trên tất cả video Bayes đủ dài; mẫu này khác giao tám phương pháp ở bảng chính.','', '## 5. Phân tầng và metric bổ sung','', 'Bảng CSV/JSON đi kèm chứa đầy đủ tám phương pháp trên cùng giao video cho từng nhóm tông da, ánh sáng, độ phân giải, HDR và xoay. RMSE_mean_person là tính căn trung bình bình phương lỗi trong từng người rồi trung bình giữa người. Bias có dấu = dự đoán − PR; âm là dự đoán thấp. Pearson dùng trung bình dự đoán và PR theo người. LoA = bias ±1,96 SD của lỗi trung bình mỗi người: mô tả phân bố giữa người, không phải giới hạn sai số cho từng video.', '', 'Các CI của phân tầng mang tính mô tả, chưa hiệu chỉnh nhiều phép so sánh; nhóm ít người không đủ để kết luận tổng quát. SNR và SQI là hai khái niệm khác nhau; bảng SQI không thay thế báo cáo SNR phổ trước đó.','', '## 6. Phạm vi và việc còn lại','', '221 video dev đều decode xong; hai video dưới 10 giây không được Bayes chấm. Giao các baseline còn thiếu thêm một video native và hai video C5. `BTC_COVERAGE_BY_VIDEO.csv` liệt kê chính xác tên và phương pháp thiếu; không gán dự đoán giả để lấp mẫu. Kiểm UBFC/MCD của pipeline mới chưa hoàn tất. Không có cơ sở nói Bayes mới đạt 4,67 bpm trên MCD. Chưa chọn cấu hình nộp cuối, chưa fine-tune BTC, chưa mở holdout.', '', '## 7. File và tái lập','', '- `BTC_COMMON_METRICS.csv` / `.json`: toàn bộ phân tầng, CI và chênh lệch ghép cặp.\n- `VIDEO_PREDICTIONS.csv`: dự đoán OOF cấp video, đủ kiểm lại mọi con số.\n- `FITS.json`, `CALIBRATION_FITS.json`: hệ số và danh sách người train từng fold.\n- `BTC_COVERAGE_BY_VIDEO.csv`: video bị thiếu ở từng phương pháp.\n- `G1.json`, `G2.json`, `SQI_VALIDITY_*.json`: cổng gốc giữ nguyên.\n- `CODEC_ASSOCIATION_REPORT.md`: 205 GOP và kiểm nền, không kết luận nguyên nhân duy nhất.\n- Chạy `python report_btc.py` chỉ tổng hợp các kết quả đã lưu; không train, không dùng holdout.','']
lines += ['## 8. SNR và V4 bổ sung','', 'V4 trong bảng chính đã đủ 15 checkpoint; đây là bản padding cũ đã trượt phép thử ảnh tĩnh, không phải padding-fixed và không được promote. Bảng của từng checkpoint gốc vẫn được giữ.','', 'SNR ở đây là trung vị SNR mù của đỉnh cao nhất trên các ứng viên POS/CHROM/G hợp lệ từ ba ROI; không dùng PR, không tính Viterbi. Chia tứ phân vị riêng từng điều kiện để mô tả, không dùng ngưỡng này fit lại mô hình. Người có nhiều video có thể xuất hiện ở hơn một nhóm. “Unknown” gồm video không có ứng viên phổ.', '']
for cond in ('native','c5'):
    lines += [f"### {cond} — ngưỡng SNR: "+', '.join(fmt(x) for x in snr_edges[cond])+' dB','']
    for group in ['SNR Q1','SNR Q2','SNR Q3','SNR Q4','SNR unknown']:
        rr=[r for r in results if r['condition']==cond and r['group']==group]
        if rr:lines += [group,'',table(['Phương pháp','Người/video','MAE','CI95%','Δ hằng số'],[[labels[r['method']],f"{r['people']}/{r['videos']}",fmt(r['MAE']),ci(r['CI95'][0]),fmt(r['delta_constant']['MAE'])] for r in rr]),'']
lines += ['## 9. Vì sao thiếu một số video trong bảng chung?', '',table(['Video','Điều kiện thiếu','Lý do từ kết quả đã lưu'],[
    ['S_N39_Resting, S_N41_Resting','Native và C5','Dưới 10 giây; không đệm video, không tạo nhãn cửa sổ giả'],
    ['S_AB74_Resting','Native và C5','Cả ba cửa sổ 10 giây của baseline cũ không đủ hỗ trợ ảnh/tracking; các hàng POS/CHROM đều ghi image_support_gap_or_tracking_failure'],
    ['S_AB83_Resting','Chỉ C5','Cả ba cửa sổ baseline C5 không đủ hỗ trợ ảnh/tracking; native vẫn có cửa sổ hợp lệ']]),'',
    'Bayes dùng ROI mới và có thể trả prior khi không thấy mạch; điều đó không làm video thiếu baseline tự nhiên trở thành mẫu so sánh được. Vì thế bảng chính dùng giao đủ phương pháp. Bằng chứng gốc trong COVERAGE_REASON_EVIDENCE.json; mỗi cửa sổ có hai hàng POS/CHROM nên sáu hàng lỗi tương ứng ba cửa sổ, không phải sáu cửa sổ.','']
if (P/'BTC_BLAND_ALTMAN_STATS.json').exists():
    lines += ['## 10. Bland–Altman: mức lệch và độ phân tán','',table(['Điều kiện','Phương pháp','Người','Bias bpm','LoA thấp','LoA cao'],[[r['condition'],labels[r['method']],r['people'],fmt(r['bias']),fmt(r['loa_low']),fmt(r['loa_high'])] for r in read('BTC_BLAND_ALTMAN_STATS.json')]),'',
    'Mỗi điểm của hình BTC_BLAND_ALTMAN.png là một người; file BTC_BLAND_ALTMAN_PEOPLE.csv chứa toàn bộ số đã vẽ để có thể đọc bằng bảng. Trục ngang là trung bình của dự đoán và PR; trục dọc là dự đoán trừ PR. Đường đen là bias; hai đường đứt là bias ±1,96 SD lỗi giữa người. Đây là mô tả trên tập dev, không phải khoảng tin cậy của từng dự đoán hoặc bằng chứng thiết bị đạt chuẩn lâm sàng.', '',
    'Độ dốc âm rõ cũng xuất hiện ở predictor hằng số: PR cao bị dự đoán thấp, PR thấp bị dự đoán cao. Không dùng riêng hình này để suy ra nhãn BTC sai hoặc codec là nguyên nhân.','']
text='\n'.join(lines).replace('tám phương pháp','chín phương pháp')
(P/'BTC_BAYES_REPORT.md').write_text(text,encoding='utf8')
(P/'BTC_REPORT_RECEIPT.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256((P/'VIDEO_PREDICTIONS.csv').read_bytes()).hexdigest(),V4_supplement_sha256=hashlib.sha256((P/'BTC_V4_SUPPLEMENT.csv').read_bytes()).hexdigest(),blind_SNR_sha256=hashlib.sha256((P/'BLIND_SNR_VIDEO.json').read_bytes()).hexdigest(),metric_source_sha256=hashlib.sha256(source.encode()).hexdigest(),coverage=coverage,holdout_evaluations=0),indent=2),encoding='utf8')
print(json.dumps(coverage));print('BTC_BAYES_REPORT.md written')
