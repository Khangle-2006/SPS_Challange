from pathlib import Path
import json
P=Path(__file__).resolve().parent
text=(P/'VIS_REPORT.md').read_text(encoding='utf-8');rows=json.loads((P/'SELECTION.json').read_text());stats=json.loads((P/'READING_TABLES.json').read_text())
intro='''
**Kết quả chính:** cả 100 video đã có bản đồ mô hình và hình tín hiệu; 5 video codec đã có đủ hai recipe. Không thấy mẫu “BTC tập trung vào mép/ngoài-da vượt 40%, UBFC dưới 20%” theo cổng đã đặt. Điểm sáng về độ nhạy vẫn chủ yếu nằm ở trán và hai má trên cả UBFC lẫn BTC. Tuy nhiên, nhạy với vùng da **không đồng nghĩa** mô hình đã đo được mạch: bản đồ công suất gần nhãn trên BTC yếu hơn rõ so với UBFC. Không sửa mô hình hoặc thay ứng viên sau chẩn đoán này.

**Đọc nhanh:** mục 1 trả lời “đổi pixel ở đâu thì đầu ra đổi”; mục 2 trả lời “ở đâu có dao động gần nhãn”; mục 3–4 cho thấy dao động thay đổi theo thời gian; mục 5 so biểu diễn của mô hình; mục 6 cho thấy quyết định thực của encoder. Sáu câu hỏi này khác nhau, không lấy hình của câu này thay bằng chứng cho câu khác.

**Độ phủ:** có 283/300 cửa sổ ảnh tối đa, không đệm 17 cửa sổ thiếu hỗ trợ. Có 267 cửa sổ có nhãn; 16 cửa sổ Iriun thiếu PPG đủ hỗ trợ. `S_AB51_Resting` có PR=40 bpm: ba cửa sổ nằm ngoài dải đã khóa 42–180 bpm, nên bản đồ gần nhãn để trống, không tính là 0. Vì vậy phần bản đồ mạch có 264 cửa sổ trong dải và 99/100 video có giá trị hợp lệ. Việc thiếu này không bị che bằng trạng thái “worker hoàn tất”.
'''
text=text.replace('## Mẫu được chọn trước',intro+'\n## Mẫu được chọn trước',1)
text=text.replace('(SPATIAL_SUMMARY.png)','(SPATIAL_SUMMARY_COMMON.png)')
text=text.replace('vẫn được lưu trong dữ liệu và hình từng video','vẫn được lưu trong dữ liệu `occlusion_signed`; hình độ nhạy dùng trị tuyệt đối')
text=text.replace('**Cách đọc:** cột gradient', '**Cách đọc:** độ nhạy dùng cửa sổ 10 giây hợp lệ đầu tiên của mỗi video, gộp theo thời gian. Cột gradient')
text=text.replace('Bốn vùng được chia bằng landmark.', '''**Đọc hình đã kiểm:** UBFC có ba vùng nổi bật ở trán và hai má; hai nhóm BTC cũng giữ hình dạng tương tự, không chuyển thành một vành sáng bao quanh oval. Occlusion UBFC nhạy nhất ở hai ô má; BTC phân tán hơn giữa má, trán và một số ô trung tâm. Đây là mô tả trên hình trung bình nhóm, không có nghĩa từng video giống hệt nhau. Sự giống nhau giữa các nhóm cũng có thể chịu ảnh hưởng kiến trúc/mẫu không gian học được; kiểm gradient bằng sai phân không thay cho phép kiểm ngẫu nhiên hóa trọng số.

**Công suất được lấy thế nào?** Tần số `f` được decoder gốc chọn. Sau đó tính bình phương độ lớn hệ số Fourier có cửa sổ Hann của đầu ra đã cumsum và bỏ xu hướng tuyến tính tại chính `f`. Gradient đo đạo hàm của đại lượng này; occlusion đo công suất trước trừ sau khi che ô. Đây không phải đạo hàm của toàn bộ bộ lọc Butterworth/Welch trong decoder. Che bằng trung bình clip, không chuẩn hóa lại sau che.

Bốn vùng được chia bằng landmark.''',1)
text=text.replace('## 2. Mạch có xuất hiện ở khối nào?','## 2. Mạch có xuất hiện ở khối nào?\n\n![Bản đồ công suất gần nhãn](PULSE_SUMMARY.png)',1)
table='\n**Số đo để đọc hình:** trung bình các ô/cửa sổ hợp lệ trong mỗi video rồi trung bình video; đây là thống kê bản đồ, không phải MAE theo người.\n\n| Nhóm | Video hợp lệ | Cửa sổ có nhãn trong dải | G: công suất gần PR | POS: công suất gần PR |\n|---|---:|---:|---:|---:|\n'
for s in stats:table+=f"| {s['group']} | {s['POS_ratio_n']} | {s['in_band_label_windows']} | {100*s['G_ratio_mean']:.2f}% | {100*s['POS_ratio_mean']:.2f}% |\n"
table+='''
**Đọc hình:** POS trên UBFC có vùng gần nhãn rõ hơn ở trán và má, trung bình 21,07%. Trên BTC-best là 6,48%, BTC-worst 4,50% (19 video hợp lệ); FullHD 7,81% và Iriun 6,96%. Không thấy một vùng BTC sáng mạnh, ổn định trên hình trung bình để chọn ngay làm ROI mới. Không suy “toàn bộ BTC không còn mạch”: phép đo dùng ô rất nhỏ, dải ±3 bpm hẹp và nhãn cấp video; nó không giống POS toàn mặt hoặc một kiểm định có đối chứng xáo nhãn. Đặc biệt, BTC-best/worst được chọn bằng sai số của **bản hiệu chỉnh**, không phải sai số padding-fixed.
'''
text=text.replace('**Có gợi ý sửa gì không?** Chỉ xem đây',table+'\n**Có gợi ý sửa gì không?** Chỉ xem đây',1)
text=text.replace('Phần đọc hình cụ thể được ghi trong biên bản kiểm hình ở cuối.','Không dùng nhãn để chọn ROI cho suy luận.')
text=text.replace('**Có gợi ý sửa gì không?** Chỉ đề xuất kiểm tra tiếp', '''**Đọc hình:** `S_AB16_Resting` trong nhóm BTC-best có vân đỏ/xanh mạnh ở nửa dưới mặt; `S_AB38_Resting` trong nhóm worst có các đợt biên độ lớn, nhất là khoảng 18–23 giây. Ngược lại, các lát cắt UBFC khá nhạt trên cùng thang màu dù bản đồ POS gần PR tốt hơn. Vì vậy “vân mạnh hơn” không có nghĩa “mạch tốt hơn”. Các dải ở rìa và đầu/cuối đoạn còn có thể chịu tác động crop/biên lọc; không gọi chúng là macroblock codec nếu chưa khớp bitstream. EVM được nội suy trên các khung ROI còn hợp lệ; chưa có kiểm riêng về ảnh hưởng khoảng trống ROI cho hình EVM, nên không dùng hình này để kết luận mất tín hiệu.

**Có gợi ý sửa gì không?** Chỉ đề xuất kiểm tra tiếp''',1)
text=text.replace('Tương quan cực đại đã tìm qua nhiều độ trễ', '''**Đọc hình và số đo:** trên 43 cửa sổ Iriun có PPG, tương quan ở offset 0 trung bình **0,090**, sau tìm cực đại ±1 s là **0,481**; **6/43** cực đại nằm ngay biên. Đối chiếu cùng cách tính: UBFC 0,783 → 0,864 (58 cửa sổ), FullHD 0,512 → 0,569 (58 cửa sổ). Đây là trung bình cửa sổ chẩn đoán, không phải CI hay phép so sánh ghép cặp theo người. Ví dụ Iriun `1563` có năng lượng trải qua nhiều tần số và thay đổi giữa các đoạn; không bám liên tục vạch PR khoảng 89 bpm. Đoạn 0–10 giây không có PPG hợp lệ để vẽ sóng ghép cặp, nên để trống.

Trong 108 cửa sổ BTC, **0 cửa sổ có HR mô hình ≥175 bpm**. Đầu ra của bản padding-fixed hiện tại không lặp lại hiện tượng trần 180 bpm của bản lỗi cũ trong mẫu này. Dự đoán BTC-best có trung bình 74,48 và SD 17,17 bpm; BTC-worst 72,05 và SD 20,86 bpm. Các số đó mô tả dự đoán, không chứng minh đã đo đúng PR. Ví dụ `S_AB13_Resting`, PR=90, ba dự đoán lần lượt khoảng 70,75 / 45,26 / 90,97 bpm: có đoạn gần nhãn, có đoạn lệch rõ.

![Ví dụ BTC: PR 90 nhưng dự đoán thay đổi giữa ba đoạn](signal/000_S_AB13_Resting/temporal.png)

Tương quan cực đại đã tìm qua nhiều độ trễ''',1)
for r in rows:
    if r['group']=='MCD Iriun':text=text.replace(f"signal/{r['key']}/temporal.png",f"signal/{r['key']}/temporal_review.png")
text=text.replace('**Có gợi ý sửa gì không?** Hình tách cụm', '''**Đọc hình:** một phần FullHD tạo cụm riêng bên trái; phần khác nằm xen các miền ở bên phải. UBFC thiên về phía dưới-phải, Iriun thiên về phía trên; BTC trải giữa và có chồng lấp. Vì vậy hình không ủng hộ câu “toàn bộ BTC nằm ngoài đám mây train”. Trong 128 chiều chuẩn hóa, hai nhóm BTC cách nhau 4,75; FullHD–Iriun là 15,61, lớn hơn BTC-Normal–FullHD 7,22. Khác biệt giữa hai camera của MCD cũng lớn, không chỉ BTC mới khác miền. Các khoảng cách không có đơn vị bpm và không phải độ chính xác.

**Có gợi ý sửa gì không?** Hình tách cụm''',1)
text=text.replace('**Có gợi ý sửa gì không?** Đây là chứng cứ', '''**Đọc hình:** cả năm video có skip trong bbox mặt **cao hơn ở bf0**: default 54,5–70,5%, bf0 72,4–83,7%. QP trung bình khoảng 34,65–37,77. Default có 227–236 khung B mỗi clip, bf0 có 0; thứ tự loại khung đã khớp ffprobe. Ví dụ `S_AB1_Resting`, skip tăng 54,5% → 72,4% trong khi QP gần như ngang nhau (36,99 → 36,87). Như vậy số khối skip riêng lẻ không giải thích được vì sao bf0 từng có kết quả mạch tốt hơn. Năm video này đều Abnormal (4 Resting, 1 Task), không đại diện toàn BTC.

**Có gợi ý sửa gì không?** Đây là chứng cứ''',1)
text=text.replace('## Nguồn và tái lập','''## Tra nhanh ký hiệu

| Ký hiệu / tên | Nghĩa trong báo cáo |
|---|---|
| PR, bpm | Nhịp mạch tham chiếu, số nhịp/phút; BTC có một số cho video |
| Hz | Chu kỳ/giây; nhân 60 để đổi sang bpm |
| PPG | Sóng quang đo tiếp xúc dùng làm tham chiếu UBFC/MCD |
| G / POS | Chuỗi xanh lá / phép phối hợp RGB để giảm một số nhiễu chung |
| ROI / oval | Vùng ảnh mặt / mặt nạ hình oval của pipeline |
| Gradient / occlusion | Đổi pixel rất nhỏ / che một ô để đo độ nhạy đầu ra |
| Cumsum | Cộng dồn sai phân đầu ra để tái tạo dạng sóng |
| SD | Độ lệch chuẩn, mức dao động quanh trung bình |
| Tương quan | Mức hai dạng sóng biến thiên cùng nhau; không đồng nghĩa sai số bpm nhỏ |
| UMAP / centroid | Phép chiếu đặc trưng để nhìn / vị trí trung bình của nhóm |
| QP / skip / khung B | Mức lượng tử hóa / khối suy từ dự đoán / khung dự đoán hai hướng |
| C5′ default / bf0 | Hai bản nén ffmpeg; bf0 tắt khung B, chưa chứng minh tương đương MATLAB |

## Nguồn và tái lập''',1)
text+='\n## Mở hình từng video\n\nTrong hình phân vùng: xanh dương = da ước lượng; đỏ = mép; hồng = mắt–miệng; xanh ngọc = ngoài oval.\n\n| Nhóm | Video | Bản đồ không gian | Phổ / dạng sóng |\n|---|---|---|---|\n'
for r in rows:
    temporal='temporal_review.png' if r['group']=='MCD Iriun' else 'temporal.png'
    text+=f"| {r['group']} | {r['recording']} | [Mở](signal/{r['key']}/spatial.png) | [Mở](signal/{r['key']}/{temporal}) |\n"
(P/'VIS_REPORT.md').write_text(text,encoding='utf-8');print('Added measured readings, coverage, glossary and 100-video figure index')
