"""Explain completed frozen GOP/background diagnostics without changing thresholds."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parent
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def truth(x):return str(x).lower()=='true'
def main():
    status=json.loads((R/'ASSOCIATION_STATUS.json').read_text());final=json.loads((R/'ASSOCIATION_FINAL.json').read_text())
    rr=rows(R/'CODEC_ASSOCIATION.csv');bg=rows(R/'BACKGROUND_ASSOCIATION.csv');exc=rows(R/'ASSOCIATION_EXCLUSIONS.csv')
    top=rows(R.parent/'round1_v4_20261105/btc_zeroshot/diagnosis205_20261108/TOP20.csv');lookup={r['recording']:r for r in rr}
    assert final['cohort_accounting_complete'] and len(rr)==198 and len(bg)==38
    wrong=[r for r in bg if float(r['skin_abserror'])>10]
    text=['# BTC: kiểm tra giả thuyết artifact GOP và phổ nền','',
    'Ngày 09/11/2026. Đã kiểm đủ phạm vi 205 video gốc đã khóa; 198 video có phổ POS toàn video hợp lệ, 7 video không đủ hỗ trợ thời gian liên tục. Trong 40 video ưu tiên kiểm nền, có 38 cặp phổ hợp lệ và 2 trường hợp loại. Không mở holdout.','',
    '**Kết luận:** có hiện tượng đỉnh POS trùng tần số nghi artifact trên toàn cohort, nhưng phép kiểm này chưa giải thích được phần lớn nhóm top-20 SNR. Phổ nền cũng chưa cho bằng chứng mạnh theo ngưỡng đã đăng ký. Không kết luận nhãn sai, pipeline không lỗi, hoặc codec là nguyên nhân duy nhất.','',
    '## 1. Đỉnh POS có gần tần số nghi artifact không?','',
    'Từ bitstream, lấy chu kỳ I-frame và khoảng giữa các frame không-B khi có B-frame. Với mỗi chu kỳ, xét sáu bội đầu, gấp tần số về dải Nyquist rồi chỉ giữ 0,7–3 Hz. Gọi là trùng khi đỉnh POS cách không quá 0,05 Hz, tương đương 3 bpm. Đây là khoảng cách tần số, không phải độ chính xác so với nhãn PR.','',
    '| Nhóm | Video hợp lệ | Số trùng | Tỷ lệ trùng | Tỷ lệ kỳ vọng từ PR train-fold |','|---|---:|---:|---:|---:|']
    for name,ss in [('Toàn cohort hợp lệ',rr),('Top-20 SNR đã chọn từ báo cáo cũ',[r for r in rr if truth(r['top20'])])]:
        n=len(ss);match=sum(truth(r['codec_match']) for r in ss);expected=sum(float(r['expected_train_person_match']) for r in ss)/n
        text.append(f'| {name} | {n} | {match} | {100*match/n:.2f}% | {100*expected:.2f}% |')
    text+=['',
    'Mốc kỳ vọng dùng PR của bốn fold train khác, mỗi người có trọng số bằng nhau. Bảng tỷ lệ trùng ở trên đếm theo video; không phải kiểm định độc lập trên từng cửa sổ. Phân bố PR và phân bố đỉnh của một phương pháp mất tín hiệu có thể khác nhau, nên phần vượt mốc kỳ vọng chưa tự nó chứng minh quan hệ nhân quả.',
    '',f"Có **{status['all']['near60_I1_count']}/198 video** đồng thời có đỉnh POS 58–62 bpm và chu kỳ I-frame trong 0,9–1,1 giây. Trong top-20, số tương ứng là **{status['top20']['near60_I1_count']}/19**. Các chu kỳ I-frame khác một giây vẫn có thể tạo tần số khác 60 bpm; con số này chỉ trả lời giả thuyết hẹp đã đặt.",
    '', '## 2. Da và nền có cùng đỉnh không?', '',
    '| Phép đếm | Kết quả |','|---|---:|',
    f"| Cặp phổ hợp lệ | {len(bg)}/40 |",
    f"| Da và nền gần nhau trong 0,05 Hz | {sum(truth(r['coincident']) for r in bg)}/{len(bg)} |",
    f"| Nền có SNR ≥6 dB theo ngưỡng khóa trước | {sum(truth(r['strong_background']) for r in bg)}/{len(bg)} |",
    f"| Vừa trùng đỉnh vừa có nền SNR ≥6 dB | {sum(truth(r['coincident']) and truth(r['strong_background']) for r in bg)}/{len(bg)} |",
    f"| Trong các video POS sai >10 bpm: trùng đỉnh da/nền | {sum(truth(r['coincident']) for r in wrong)}/{len(wrong)} |",'',
    'SNR ở đây là công suất quanh đỉnh ±0,1 Hz chia công suất còn lại trong băng, rồi đổi sang dB. 6 dB tương đương tỷ số gần 4. Không có nền đạt ngưỡng này nên chưa có bằng chứng mạnh để bật cơ chế hạ trọng số theo nền ở các mẫu đó. Không đạt ngưỡng không chứng minh video không có nhiễu chung. Vùng nền có thể gồm tóc, và mặt nạ màu chỉ là phép loại pixel kỹ thuật.','',
    '## 3. Không đánh đồng hai cách lấy dự đoán','',
    'Top-20 được chọn bằng SNR của các cửa sổ trong báo cáo cũ. Phép kiểm GOP hiện tại lấy Welch trên toàn video từ trace oval cũ, sau nội suy về 30 Hz để giữ cách xử lý cũ. Vì vậy đỉnh toàn video có thể khác trung vị dự đoán từng cửa sổ 10 giây. Nhánh Bayes mới dùng trace độ phân giải nguồn và FPS nguồn; đó là phép đánh giá riêng. Bảng dưới giữ cả hai con số để tránh coi chúng là một.','',
    '| Video top-20 | PR | POS cũ: trung vị cửa sổ | POS toàn video | Trùng tần số nghi GOP? |','|---|---:|---:|---:|---|']
    for t in top:
        r=lookup.get(t['recording']);text.append(f"| {t['recording']} | {float(t['truth']):.1f} | {float(t['video_POS']):.2f} | {float(r['POS_whole_hr']):.2f} | {'Có' if truth(r['codec_match']) else 'Không'} |" if r else f"| {t['recording']} | {float(t['truth']):.1f} | {float(t['video_POS']):.2f} | Không hợp lệ | Không chấm |")
    text+=['','## 4. Trường hợp loại và bằng chứng lưu lại','','| Video | Lý do |','|---|---|']
    for r in exc:
        reason='Trace nền có khoảng thiếu trên 0,2 giây' if r['reason'].startswith('background:') else 'Trace oval cũ có khoảng thiếu trên 0,2 giây; không nối để tính phổ toàn video'
        text.append(f"| {r['recording']} | {reason} |")
    text+=['','Một video thuộc nhóm kiểm nền đã bị loại từ bước phổ da; do đó 7 trường hợp loại phổ da cộng 1 trường hợp lỗi riêng phổ nền tương ứng với 7/205 phổ da và 2/40 cặp nền không hợp lệ. Không cộng hai mẫu số này thành một.','',
    'Bằng chứng: BTC_NATIVE_GOP.json; native_gop/*_frames.csv; CODEC_ASSOCIATION.csv; BACKGROUND_ASSOCIATION.csv; ASSOCIATION_EXCLUSIONS.csv; ASSOCIATION_FINAL.json. Các file PSD riêng nằm trên P100 trong association/. Không thay cổng G4: phép thử xám đã hoàn tất và không kích hoạt.','']
    (R/'CODEC_ASSOCIATION_REPORT.md').write_text('\n'.join(text),encoding='utf8')
if __name__=='__main__':main()
