"""Render measured diagnostics only. Final prose interpretation requires visual review."""
from pathlib import Path
import json,csv
import numpy as np
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def num(x):return 'không có' if x is None or not np.isfinite(x) else f'{x:.3f}'
rows=read(P/'SELECTION.json');groups=sorted({r['group'] for r in rows});mass=read(P/'REGIONAL_MASS.json');trigger=read(P/'EDGE_TRIGGER.json');features=read(P/'UMAP_RESULT.json') if (P/'UMAP_RESULT.json').exists() else None;codec=read(P/'CODEC_V3_COMPLETE.json') if (P/'CODEC_V3_COMPLETE.json').exists() else None
lines=['# Nhìn trực tiếp mô hình và tín hiệu FaceVitals','', '**Phạm vi:** chẩn đoán, không huấn luyện, không sửa mô hình và không mở holdout. Mô hình là padding-fixed. Các hình không thay thế cổng đánh giá và không tự chứng minh nguyên nhân sai số.','',
'## Mẫu được chọn trước','', 'UBFC, FullHD và Iriun lấy bốn người ở mỗi fold. FullHD–Iriun ghép cùng người/cùng buổi. BTC lấy 20 video có sai số hiệu chỉnh nhỏ nhất và 20 lớn nhất trong tập dev đã chấm. Đây là chọn cực trị có dùng nhãn, phù hợp để tìm hiện tượng nhưng không đại diện ngẫu nhiên cho BTC. Iriun không được chọn theo tương quan PPG tốt.','',
'| Nhóm | Video khóa | Chuẩn bị được | Bản đồ mô hình | Hình tín hiệu |','|---|---:|---:|---:|---:|']
for g in groups:
 rr=[r for r in rows if r['group']==g];counts=[sum((P/folder/r['key']/flag).exists() for r in rr) for folder,flag in [('data','READY.json'),('model','COMPLETE.json'),('signal','COMPLETE.json')]];lines.append(f'| {g} | {len(rr)} | '+ ' | '.join(map(str,counts))+' |')
lines+=['','Mỗi video được dùng tối đa ba cửa sổ 10 giây trong 30 giây đầu. Cửa sổ không đủ dữ liệu không được đệm hay thay bằng video khác. Tên và fold của toàn bộ mẫu nằm ở cuối báo cáo.','',
'## 1. Đầu ra mô hình nhạy với vùng nào?','', '![Độ nhạy và bản đồ mạch theo nhóm](SPATIAL_SUMMARY.png)','',
'**Cách đọc:** cột gradient đo thay đổi rất nhỏ ở pixel; cột occlusion đo tác động khi thay cả ô 6×6 bằng trung bình clip. Giá trị được chuẩn hóa để tổng độ nhạy mỗi video bằng 1 trước khi lấy trung bình nhóm. Vùng sáng là vùng có tác động mạnh lên **công suất ở tần số mô hình đã chọn**, không có nghĩa là vùng đó chứa mạch. Với occlusion, bảng dùng trị tuyệt đối; dấu tăng/giảm công suất vẫn được lưu trong dữ liệu và hình từng video.','',
'Bốn vùng được chia bằng landmark. “Da ước lượng” là phần bên trong oval sau khi bỏ mắt–miệng và mép; đây **không phải phân đoạn da**, nên tóc hoặc vật che có thể còn bên trong. “Ngoài oval” là vị trí vốn bị đặt 0 trong đầu vào, không phải mô hình đang nhìn nền cảnh thật. Mép lấy rộng hai pixel mỗi phía ở kích thước 36×36.','',
'| Nhóm | Phép đo | n video có vùng hợp lệ | Da ước lượng | Mép oval | Mắt–miệng | Ngoài oval | Tổng ngoài-da ước lượng |','|---|---|---:|---:|---:|---:|---:|---:|']
for g in groups:
 for method in ['gradient','occlusion']:
  rr=[r for r in mass if r['group']==g and r['method']==method]
  if rr:lines.append(f'| {g} | {method} | {len(rr)} | '+' | '.join(f'{100*np.mean([r[k] for r in rr]):.1f}%' for k in ['skin_proxy','oval_edge','eyes_mouth','exterior','non_skin'])+' |')
lines+=['','Các phần trăm trên là tỷ lệ **khối lượng độ nhạy**, không phải phần trăm diện tích mặt. Tham chiếu diện tích từng vùng được lưu cùng mỗi kết quả; vùng rộng có thể nhận nhiều khối lượng dù không nhạy hơn trên mỗi pixel. Che ô cũng tạo đầu vào khác phân bố huấn luyện, nên cần đọc cùng gradient.','']
for t in trigger:lines.append(f'- Cổng gợi ý sửa ({t["method"]}): BTC {100*t["BTC_mean"]:.1f}% / UBFC {100*t["UBFC_mean"]:.1f}% ngoài-da ước lượng; ngưỡng BTC >40% và UBFC <20%: **'+('KÍCH HOẠT' if t['trigger'] else 'KHÔNG KÍCH HOẠT')+'**.')
lines+=['','**Có gợi ý sửa gì không?** '+('Có tín hiệu theo cổng đã ghi trước: đề xuất một phép thử mặt nạ mềm có kiểm soát. Chưa sửa mô hình và chưa chứng minh mép oval là nguyên nhân sai số BTC.' if any(t['trigger'] for t in trigger) else 'Cổng mép/ngoài-da đã đặt chưa kích hoạt. Điều này không chứng minh mô hình hoàn toàn không dùng mép; chưa có căn cứ từ cổng này để sửa mặt nạ.'),'',
'## 2. Mạch có xuất hiện ở khối nào?','',
'Hai cột cuối của hình tổng hợp ở mục 1 chia đầu vào thành lưới 8×8 ô. Với mỗi ô, lấy chuỗi màu theo thời gian rồi tính G và POS. Giá trị = công suất trong PR ±3 bpm / tổng công suất 42–180 bpm. Ví dụ 0,20 nghĩa là 20% công suất trong dải đo tập trung gần nhãn; không có nghĩa độ chính xác là 20%. Ô trắng là không đủ năng lượng hoặc thiếu nhãn, không được điền 0.','',
'Nhãn được dùng trực tiếp trong phép chẩn đoán này: BTC có một PR cho video; UBFC/MCD dùng PPG ở cửa sổ tương ứng. Iriun giữ offset 0, không coi đồng hồ camera đã đồng bộ sinh lý. Một đỉnh do chuyển động/nén trùng PR cũng có thể làm ô sáng; nhãn lệch thời điểm cũng có thể làm ô tối dù có mạch.','',
'**Có gợi ý sửa gì không?** Chỉ xem đây là gợi ý vị trí cần kiểm bằng đối chứng. Không chọn ROI mới hoặc chỉnh mô hình theo tập cực trị này. Phần đọc hình cụ thể được ghi trong biên bản kiểm hình ở cuối.','',
'## 3. Lát cắt không gian–thời gian','', '![Lát cắt YT](EVM_YT.png)','',
'Trục ngang là giây; trục dọc đi từ trên xuống dưới mặt. Màu đỏ/xanh là dao động kênh xanh lá sau lọc 0,7–3 Hz và nhân 50. Các hình dùng chung thang màu. Để pyramid level 4 còn có ý nghĩa, phần này lấy ROI thô 256×256 rồi hạ qua bốn mức còn 16×16; không lấy ảnh 36×36 rồi giả tạo thêm chi tiết. Đây là lát cắt EVM-style, không phải một video EVM hoàn chỉnh.','',
'Vân lặp theo thời gian có thể đến từ mạch, chuyển động, ánh sáng hoặc codec. Hệ số 50 chỉ phóng đại để nhìn; không phải biên độ mạch thật. So sánh cùng bản đồ PR và phổ, không đo HR bằng mắt từ riêng hình này.','',
'**Có gợi ý sửa gì không?** Chỉ đề xuất kiểm tra tiếp nếu vân có vị trí/thời gian phù hợp đồng thời với tín hiệu PPG hoặc hiện vật codec đo được; không suy ra nguyên nhân từ một lát cắt.','',
'## 4. Đầu ra dao động ở tần số nào?','',
'Mỗi video có hình `temporal.png`: phổ của đầu ra mô hình sau cumsum và phổ POS. Trục dọc là bpm; đường xanh đứt là nhãn. Mỗi cửa sổ có thang công suất tương đối riêng, nên một dải sáng có thể vẫn là tín hiệu rất yếu; không so biên độ tuyệt đối giữa video bằng độ sáng. Cửa sổ phổ dài 5 giây; nfft chỉ làm lưới vẽ mịn hơn, không tạo thêm độ phân giải vật lý.','',
'Ba cửa sổ mô hình được chuẩn hóa và tái tạo độc lập. Ranh giới 10 giây được đánh dấu, khoảng trắng là phần thiếu hỗ trợ hoặc mép cửa sổ phổ. Không cumsum xuyên qua ba đoạn để tạo gián đoạn giả. Trên Iriun, waveform được vẽ cạnh PPG chuẩn hóa, không áp dịch nhãn.','',
'| Iriun | Số cửa sổ có PPG | Tương quan tại offset 0 (TB cửa sổ) | Tương quan lớn nhất trong ±1 s (TB) | Số cực đại ở biên ±1 s |','|---|---:|---:|---:|---:|']
for r in rows:
 p=P/'signal'/r['key']/'METRICS.json'
 if r['group']=='MCD Iriun' and p.exists():
  cc=[x for x in read(p)['correlations'] if x is not None];lines.append(f'| {r["recording"]} | {len(cc)} | {num(np.mean([x["zero"] for x in cc]))} | {num(np.mean([x["max_signed"] for x in cc]))} | {sum(x["boundary"] for x in cc)} |')
example=next((r for r in rows if r['group']=='MCD Iriun' and (P/'signal'/r['key']/'temporal.png').exists()),None)
if example:lines+=['',f'![Ví dụ Iriun: {example["recording"]}](signal/{example["key"]}/temporal.png)']
lines+=['','Tương quan cực đại đã tìm qua nhiều độ trễ nên thường cao hơn tương quan tại 0. Nó không phải chứng nhận đồng bộ camera, và tín hiệu tuần hoàn có thể có nhiều cực đại. Kết quả nằm ở biên ±1 giây cần đọc đặc biệt thận trọng.','', '**Có gợi ý sửa gì không?** Không đổi offset hoặc decoder từ các hình này. Chỉ phân biệt mô tả: đỉnh cố định, đỉnh thay đổi, hay không thấy đỉnh ổn định; muốn quy cho hô hấp/chuyển động cần đo đối chứng tương ứng.','',
'## 5. Các nhóm có nằm khác nhau trong không gian đặc trưng?','', '![UMAP đặc trưng](UMAP.png)','',
'Mỗi điểm là một cửa sổ, lấy trung bình 128 đặc trưng lớp áp cuối. Tất cả nhóm dùng **cùng checkpoint fold 0** để các tọa độ có cùng ý nghĩa. Một phần người công khai đã có trong tập train của checkpoint này; đây không phải đánh giá OOF. Các heatmap và dự đoán ở mục khác vẫn dùng checkpoint đúng fold. BTC được chọn theo cực trị sai số nên hình không mô tả toàn bộ phân bố BTC.','',
'UMAP nén 128 chiều xuống 2 chiều để nhìn; khoảng trống trên hình có thể bị phép chiếu phóng đại. Vì vậy bảng dưới báo khoảng cách Euclid giữa centroid trong **128 chiều đã chuẩn hóa**, không dùng khoảng cách 2-D để chứng minh lệch miền. Nhiều cửa sổ cùng người có tương quan.','']
if features:
 gs=features['groups'];lines+=['| Nhóm | '+' | '.join(gs)+' |','|---|'+ '|'.join(['---:']*len(gs))+'|']
 for g,d in zip(gs,features['centroid_distance_standardized128']):lines.append('| '+g+' | '+' | '.join(f'{x:.2f}' for x in d)+' |')
lines+=['','**Có gợi ý sửa gì không?** Hình tách cụm có thể gợi ý miền khác nhau trong biểu diễn này, nhưng không chỉ ra cần sửa màu, nén hay mặt nạ. Không mở một nhánh thích nghi mới từ UMAP.','',
'## 6. Codec xử lý các khối trên mặt như thế nào?','',
'QP là chỉ số lượng tử hóa của macroblock: thường QP lớn làm lượng tử thô hơn. Skip là cờ giải mã cho khối được suy ra bằng dự đoán, **không có nghĩa khối đó bị xóa hay không có pixel**. Bản đồ lấy trực tiếp log decoder H.264 và được đối chiếu đúng thứ tự `pict_type` với ffprobe; không suy skip từ vector chuyển động bằng 0. Khung xanh chỉ bbox mặt, không phải phân đoạn da.','',
'| Video | Recipe | Khung | Khung B | Tỷ lệ skip trong bbox mặt (TB thời gian) | QP trong bbox mặt (TB) |','|---|---|---:|---:|---:|---:|']
if codec:
 for r in codec['rows']:lines.append(f'| {r["recording"]} | {r["variant"]} | {r["frames"]} | {r["B_frames"]} | {100*r["face_bbox_skip_fraction"]:.1f}% | {r["face_bbox_QP"]:.2f} |')
 if codec['rows']:
  rid=codec['rows'][0]['recording'];lines+=['',f'![Macroblock QP/skip — {rid}](codec_v3/{rid}/CODEC_MAP.png)']
 lines+=['',f'Lỗi/chưa hoàn tất codec: {codec["errors"]}.']
lines+=['','**Có gợi ý sửa gì không?** Đây là chứng cứ về quyết định mã hóa tại vị trí mặt. Chênh lệch default/bf0 chưa cô lập tác động riêng của B-frame; thay recipe làm encoder thay đổi nhiều quyết định. Không kết luận skip/QP đã xóa mạch chỉ vì vùng đó có nhiều skip. Cả hai C5′ vẫn là proxy ffmpeg, chưa xác minh tương đương MATLAB VideoWriter.','',
'## Kiểm tra và giới hạn','', '- Gradient dùng float64 để kiểm sai phân nhỏ; dự đoán và occlusion dùng float32. Trọng số, loss, ngưỡng và tập người không thay đổi. Xem NUMERICAL_ADDENDUM.md và toàn bộ lỗi ban đầu được bảo lưu.','- Các nhóm cực trị BTC không dùng để báo một MAE đại diện mới hay chọn phương pháp nộp. Cấu hình đóng băng trước đó vẫn giữ.','- Không thể phân biệt da, tóc và vật che hoàn hảo chỉ bằng landmark. Không gọi phân vùng này là segmentation da chuẩn.','- EVM và phổ là hình minh họa có lọc, không phải đo sinh lý độc lập. Iriun còn bất định đồng hồ.','- Không mở 72 video/41 người holdout, D3 hoặc 7 UBFC test. Không huấn luyện.','',
'## Nguồn và tái lập','', '- [FFmpeg 6.1.1: định nghĩa ký hiệu QP/MB type](https://raw.githubusercontent.com/FFmpeg/FFmpeg/n6.1.1/libavcodec/mpegutils.c).','- [UMAP: cách dùng và tham số](https://umap-learn.readthedocs.io/en/latest/basic_usage.html).','- PROTOCOL.md, FROZEN.json, SELECTION.json; mỗi đầu ra có SHA256 và checkpoint đúng fold. Các lệnh thực thi/PID nằm trong LAUNCH*.json; không chạy lại worker trên thư mục đã có kết quả.','',
'## Danh sách mẫu khóa','', '| Nhóm | Video | Người | Fold | Chọn theo |','|---|---|---|---:|---|']
for r in rows:lines.append(f'| {r["group"]} | {r["recording"]} | {r["person"]} | {r["fold"]} | '+(f'Hiệu chỉnh: sai số tuyệt đối={r["calibrated_abs_error"]:.3f} bpm' if 'calibrated_abs_error' in r else 'Seed 0, không dùng sai số/tương quan')+' |')
(P/'VIS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (P/'SELECTION.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['group','recording','person','fold','source','source_sha256','calibrated_abs_error'],extrasaction='ignore');w.writeheader();w.writerows(rows)
print('VIS_REPORT.md rendered; requires visual QA and final reading notes')
