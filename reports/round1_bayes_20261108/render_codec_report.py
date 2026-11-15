"""Readable tables from completed controls, no model selection or data rerun."""
from pathlib import Path
import json,csv

R=Path(__file__).resolve().parent
def load(name):return json.loads((R/name).read_text(encoding='utf-8-sig'))

def main():
    data=load('SYNTHETIC_RESULTS.json');green={(r['input'],r['recipe']):r for r in load('SYNTHETIC_GREEN_RECOVERY.json')}
    names={'old_x264':'x264 cũ','noB_x264':'x264 không B, GOP 10','part2_experimental':'Part 2 thực nghiệm'}
    text=['# Kiểm soát codec — 18 trường hợp đã hoàn tất','',
    'Đây là thí nghiệm có kiểm soát trên một ảnh mặt BTC và một đoạn UBFC dev, không phải đánh giá độ chính xác trên toàn bộ dataset. Không mở holdout. Tất cả đầu ra đều giữ lại, kể cả trường hợp không đạt bitrate.','',
    '## 1. Cổng G4 và điều có thể kết luận','',
    '**G4 không kích hoạt.** Ba nền xám phẳng cho POS bằng 0 và không có đỉnh 2,5 Hz. Theo cổng đăng ký, chưa được gắn nhãn mọi C5 cũ là nhiễm artifact hoặc tự tạo C5′. Điều đó không chứng minh video mặt không bị ảnh hưởng bởi codec.','',
    'Đoạn UBFC cùng đầu vào cho đỉnh POS trước encode khoảng 106,82 bpm. Sau x264 cũ, đỉnh mạnh nhất ở 149,88 bpm; năng lượng tại 2,5 Hz bằng 55,25 lần trung vị phổ trong băng. x264 không B/GOP 10 và Part 2 lần lượt cho đỉnh 70,44 và 59,88 bpm. Đây là sự thay đổi do toàn bộ quá trình encode/decode trong đối chứng này. Không đủ để khẳng định riêng B-frame, I-frame hay GOP là nguyên nhân duy nhất, và chưa kết luận recipe nào chính xác hơn về HR vì đây không phải phép so với PPG đồng thời.','',
    '## 2. Toàn bộ kết quả','',
    'Đỉnh POS là đỉnh mạnh nhất của phổ đã lọc 0,7–3 Hz. Tỷ số 2,5 Hz = công suất tại 2,5 Hz / trung vị công suất trong băng. Giá trị này khác SNR quanh đỉnh và không phải xác suất có artifact. Bitrate là dung lượng toàn file × 8 / 30 giây.','',
    '| Đầu vào | Recipe | kbps | ≤500? | Đỉnh POS (bpm) | Tỷ số 2,5 Hz |','|---|---|---:|---|---:|---:|']
    table=[]
    for r in data:
        inp='Xám phẳng' if r['input']=='gray' else 'UBFC thật' if r['input']=='UBFC_real' else f"Mặt + sin {float(r['input'].split('_')[1])*100:g}% DC"
        peak=r['POS_peaks'][0]['hr'] if r['POS_peaks'] else None
        text.append(f"| {inp} | {names[r['recipe']]} | {r['whole_file_kbps']:.2f} | {'Đạt' if r['bitrate_pass'] else 'KHÔNG'} | {f'{peak:.2f}' if peak is not None else 'Không có'} | {r['ratio_2_5_to_median']:.2f} |")
        table.append(dict(input=r['input'],recipe=r['recipe'],whole_file_kbps=r['whole_file_kbps'],bitrate_pass=r['bitrate_pass'],POS_peak_bpm=peak,ratio_2_5_to_median=r['ratio_2_5_to_median'],**{k:v for k,v in green[(r['input'],r['recipe'])].items() if k not in ['input','recipe']}))
    text+=['','**Part 2, sin 2%: 500,84 kbps, không đạt giới hạn.** Không sửa bitrate hoặc chạy lại để thay kết quả này. Mọi recipe dùng cùng bitrate đặt 450 kbps nhưng bitrate thực đo có thể khác.','',
    '## 3. Tín hiệu 72 bpm được giữ lại bao nhiêu?','',
    'Sin có tần số 1,2 Hz, tương đương 72 bpm. Biên độ đặt là tỷ lệ dao động kênh xanh so với độ sáng trung bình DC. Kênh đỏ nhận 1/4, kênh xanh dương nhận 1/10 biên độ tương đối đó. Ảnh được lượng tử về RGB 8-bit trước encode. ROI chèn là ellipse cố định trong hộp mặt, chưa phải mặt nạ da chính xác từng pixel.','',
    'Hệ số khôi phục là biên độ sin sau decode chia biên độ đo được trước encode, dùng cùng ROI và cách xử lý. Bảng tách kênh xanh và đầu ra POS, vì POS biến đổi và chuẩn hóa các kênh màu nên hai tỷ lệ không đồng nhất. 100% nghĩa là biên độ được giữ nguyên; không có nghĩa HR đã đúng.','',
    '| Biên độ đặt (% DC) | Recipe | Xanh trước encode (% DC) | Xanh sau decode (% DC) | Khôi phục xanh (%) | Khôi phục POS (%) |','|---:|---|---:|---:|---:|---:|']
    for r in data:
        if not r['input'].startswith('face_'):continue
        g=green[(r['input'],r['recipe'])]
        text.append(f"| {float(r['input'].split('_')[1])*100:g} | {names[r['recipe']]} | {100*g['input_green_amp_DC']:.4f} | {100*g['output_green_amp_DC']:.4f} | {100*g['green_recovery_ratio']:.2f} | {100*r['recovery_ratio']:.2f} |")
    text+=['','Bảng là đường đáp ứng theo biên độ, có thể đọc trực tiếp mà không cần biểu đồ. Không dùng tỷ lệ tại 1,2 Hz của video UBFC thật làm “khôi phục mạch”, vì chưa có cơ sở mạch thật của đoạn đó đúng 1,2 Hz.','',
    '## 4. Tái lập và giới hạn','',
    '- Dữ liệu 30 giây; đầu vào tổng hợp được lấy ở các thời điểm 0, 0,1, 0,2… giây, tương đương lấy khung 0, 3, 6… từ lưới 30 Hz. Không lọc trước khi bỏ khung.',
    '- x264 cũ: veryfast, GOP 20, keyint_min 20, sc_threshold 0; x264 đối chứng: bf 0, GOP 10; Part 2: mpeg4, bf 0, GOP 10. Cùng b:v 450k, maxrate 450k, bufsize 900k, yuv420p.',
    '- Mỗi thư mục synthetic/<đầu vào>_<recipe>/ chứa video, lệnh encode, log, frames.csv, PSD.csv, traces.npz và RESULT.json. INPUT.json / UBFC_INPUT.json lưu nguồn, ROI, chỉ số và PTS.',
    '- MATLAB VideoWriter MPEG-4 dùng H.264. Part 2 ở đây là đối chứng thực nghiệm, chưa chứng minh tương đương MATLAB.',
    '- Tín hiệu ở ROI nền, GOP trên 205 BTC và độ chính xác full-cohort là các phép đo riêng; chưa được suy ra từ 18 trường hợp này.',
    '- G4 là điều kiện trên nền xám. Đỉnh 150 bpm ở UBFC thật không tự động thay điều kiện G4 sau khi đã xem kết quả.','']
    (R/'CODEC_CONTROLS.md').write_text('\n'.join(text),encoding='utf8')
    with (R/'CODEC_CONTROLS.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

if __name__=='__main__':main()
