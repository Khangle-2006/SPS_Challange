# Dữ liệu, đồng hồ và cách tính số

## Một mẫu là gì?

| Bộ | Dữ liệu | Nhãn dùng cho PR/HR | Cần chú ý |
|---|---|---|---|
| BTC | Video điện thoại Normal/Abnormal, nhiều tình huống của cùng người | Một số PR cấp video từ bảng BTC | Không có PPG đồng bộ để dựng waveform target; không biết chắc thời điểm đo PR |
| UBFC-rPPG | Video mặt và `ground_truth.txt` | PPG, HR và thời gian theo định dạng của bộ | UBFC1 và UBFC2 khác schema; không coi mỗi hàng nhãn là một frame |
| MCD-rPPG | FullHD/USB/Iriun và contact PPG | PPG có đồng hồ riêng | FullHD đã đo là 640×480 MPEG-4 Part 2; tên camera không đảm bảo độ phân giải |
| AiPEX | Ba crop ROI của cùng một lần quay | HR trung bình cấp người | 88 người, không phải 264 người; giả định 15 fps còn giới hạn xác minh |

BTC dev có 221 video hiện diện của 161 người; báo cáo ghép cặp chính dùng 218/158 đủ điều kiện. Holdout 72/41 bị khóa. Không tự gộp holdout vào thống kê, chuẩn hóa, prior hoặc QC. Có hai video thiếu đã được hoãn, không tạo dữ liệu giả để điền.

### FPS và timestamp

`fps` = khung/giây. FPS container, FPS cảm biến và lưới nội suy mô hình là ba đại lượng khác nhau. Mô hình thường dùng 301 điểm ảnh cho 10 s ở 30 Hz, sau sai phân còn 300 điểm. Điều này **không** chứng minh video nguồn đủ 10 s: phải kiểm khoảng thời gian nguồn, support, gap và valid mask. Không nối qua gap lớn hoặc ngoại suy khi thiếu đoạn cuối.

MCD119 nguồn: 640×480, MPEG-4 Part 2; bitrate toàn file chia thời lượng theo source clock có trung vị 1446,118 kbps. UBFC ở chẩn đoán T9 có FPS nguồn khoảng 23,18–29,97; nhãn `h26430` lịch sử nghĩa là giữ toàn bộ frame, không mặc định 30 fps chính xác. BTC có FPS/rotation/HLG/10-bit đa dạng nên phải dùng metadata từng file.

MCD PPG canonical được nhân −1 **đúng một lần**, UBFC +1. Sửa dấu không phải đồng bộ thời gian. Offset V5 là ước lượng pha, không là chứng cứ thời điểm phơi sáng camera đã đồng bộ vật lý.

## Ký hiệu

| Tên | Giải thích |
|---|---|
| PR | Pulse rate: số nhịp mạch/phút, đáp án BTC từ thiết bị đo mạch |
| HR | Heart rate: số nhịp tim/phút; tên thường dùng trong decoder/research. Không mặc nhiên đồng nhất mọi phép đo PR với ECG HR |
| bpm | Beats per minute, nhịp/phút; 1,5 Hz tương ứng 90 bpm |
| PPG | Dạng sóng quang đo tiếp xúc; giá trị biên độ không phải bpm |
| rPPG | Tín hiệu nhịp mạch suy ra từ biến thiên màu video |
| SpO₂ | Độ bão hòa oxy ngoại vi (%), không phải mục tiêu Round 1 này |
| ROI | Vùng ảnh lấy tín hiệu, như trán/hai má/oval mặt |
| POS / CHROM / G / PBV | Các cách rút tín hiệu từ chuỗi màu RGB; G chỉ kênh xanh lá |
| native | Không thêm nén của thí nghiệm; video gốc vẫn có thể đã nén |
| C5 / C5′ | Điều kiện FPS thấp + bitrate thấp. C5′ ở đây là ffmpeg proxy; không gọi là MATLAB parity |
| bf0 | Encoder không dùng B-frame; không tự chứng minh mọi artifact đã mất |
| d_k | Đầu ra sai phân của mô hình. Decoder đã cumsum trước detrend/filter/phổ |
| OOF | Dự đoán cho người thuộc fold validation mà bước fit tương ứng không được dùng |
| CI95% | Khoảng tin cậy bootstrap; CI của chênh lệch chứa 0 nghĩa là chưa chứng minh cải thiện theo phép thử đó |
| Static PASS | Lặp cùng ảnh, decoder phải từ chối trả HR. Phạm vi PASS chỉ đúng đường code đã kiểm |

## Công thức

Với người p có n_p video: `MAE_p = mean_v |prediction_pv − PR_pv|`.
`MAE = mean_p MAE_p`: mỗi người cùng trọng số, không để người có nhiều Task lấn át.
`bias = mean_p mean_v (prediction_pv − PR_pv)`: âm nghĩa là dự đoán thấp.
Delta báo là `phương án mới − mốc`; delta âm tốt hơn.

Bootstrap hiện tại: lấy mẫu lại **người**, giữ tất cả dữ liệu của người đó và cặp phương pháp, 10.000 lần, seed 0. Phân vị 2,5 và 97,5 tạo CI95%. Không bootstrap từng cửa sổ như các mẫu độc lập. Chẩn đoán T9 lịch sử dùng seed 20261031, ghi riêng.

`S_public = (MAE_UBFC_native + MAE_UBFC_C5_old + MAE_MCD_native)/3`.
P95 có thể là phân vị của lỗi cửa sổ hoặc MAE từng người: các báo cáo ghi rõ, không tráo hai cách tính. Báo coverage video và window; từ chối dự đoán không tự được tính thành 0 lỗi. Bảng nhóm có người xuất hiện ở nhiều task nên không cộng số người nhóm để suy ra toàn cohort.
