# Pipeline FaceVitals, giải thích chi tiết

Tài liệu này đi theo thứ tự dữ liệu chạy qua: **video → vùng da → sóng mạch → bằng chứng theo tần số → nhịp tim**. Mỗi khối có ba phần: nó làm gì, cơ chế bên trong, và câu tóm tắt để trình bày. Tham số chính xác nằm trong `READER_V4_DEFINITION.txt` và `P4M_DEFINITION.txt`. Lệnh chạy nằm trong [README](../README.md#cách-chạy-pipeline).

![Pipeline](figures/pipeline.png)

---

## 0. Ý tưởng vật lý

Mỗi lần tim co bóp, một đợt máu được đẩy tới mao mạch dưới da mặt. Máu, cụ thể là hemoglobin, hấp thụ ánh sáng, mạnh nhất ở dải lục. Vì vậy theo mỗi nhịp, da **tối đi rất nhẹ rồi sáng lại**, khoảng 0,1–0,5% độ sáng. Mắt không thấy, nhưng camera ghi được.

Dãy giá trị đó theo thời gian là **sóng thể tích máu** (BVP, hay tín hiệu rPPG). Đây cũng là thứ máy đo SpO₂ kẹp ngón tay đo được, chỉ khác là không cần tiếp xúc. **Nhịp tim là tần số lặp lại của sóng này.**

Khó khăn nằm ở chỗ biến động do cử động và ánh sáng lớn hơn tín hiệu mạch hàng chục lần.

---

## 1. Frontend: tìm và cắt vùng da

**Làm gì.** Biến mỗi khung hình (ví dụ 1920×1080) thành một ảnh da mặt nhỏ.

1. **Giải mã theo timestamp thật (PTS).** Video điện thoại không đều 30 khung/giây. Nếu giả định đều thì đồng hồ của tín hiệu bị lệch và nhịp tính ra sai.
2. **MediaPipe FaceMesh** cho 468 điểm trên mặt. Pipeline lấy **36 điểm viền mặt** và nối thành hình oval.
3. **Mặt nạ oval.** Điểm ảnh ngoài oval (tóc, nền, quần áo) bị đặt về 0, vì chúng không chứa mạch.
4. **Thu nhỏ vùng mặt về 36×36 bằng lấy trung bình vùng (INTER_AREA).** Mỗi ô 36×36 là trung bình của hàng trăm điểm ảnh gốc. Nhiễu ngẫu nhiên của cảm biến bị triệt, còn mạch (đồng đều trên da) vẫn giữ.
5. **Crop thứ hai: hộp chặt quanh oval, không che, 72×72**, cho các mô hình huấn luyện sẵn, vì chúng học trên ảnh mặt bình thường.

**Trợ giúp cho constraint.** Các trợ giúp này chỉ đổi cách *tìm* mặt; tín hiệu luôn lấy từ ảnh gốc.
- **C3 nhiều người — khóa mục tiêu `lock0`.** Bộ dò toàn khung, Kalman, gán Hungarian và chữ ký ngoại hình. Hạt giống là hộp mặt hợp lệ đầu tiên.
- **C7 che mặt, C1 mất mặt — giữ theo template `track`.** Khi FaceMesh trượt, tìm lại phần trên khuôn mặt (trán, mắt) bằng tương quan chéo chuẩn hóa. Nếu điểm khớp ≥ 0,6 thì dời oval cũ theo độ dịch tìm được.
- **C1 tối — trợ giúp dò `aid`.** FaceMesh chạy trên một bản sao đã tăng sáng và làm mờ.

> **Câu trình bày:** "Tín hiệu mạch chỉ nằm trên da, nên chúng em khoanh đúng vùng da bằng 36 điểm viền mặt, bỏ nền, rồi lấy trung bình không gian để triệt nhiễu cảm biến."

---

## 2. Nguồn tín hiệu: mỗi nguồn cho ra một sóng mạch

Không nguồn nào cho ra nhịp tim trực tiếp. Mỗi nguồn cho ra **một sóng**, mỗi khung một số. Video 60 s ở 30 fps cho khoảng 1.800 số.

| Nguồn | Đầu vào mỗi khung | Cách tạo sóng | Dùng ở |
|---|---|---|---|
| POS, CHROM | màu trung bình R, G, B của crop oval | công thức vật lý, không học | BTC; công khai (CHROM) |
| CNN của đội | chuỗi crop oval 36×36, tiền lọc 0,1 s | EfficientPhys + TSM; dự đoán độ thay đổi giữa hai khung, cộng dồn thành sóng | chỉ công khai |
| FacePhys, ME-chunk, ME-flow | crop hộp 72×72 (mô hình tự thu về 36×36) | mạng huấn luyện sẵn trên RLAP / PURE, dùng nguyên | cả hai |

### 2a. POS và CHROM

Hai quan sát:
- Khi **độ sáng** đổi (đèn, nghiêng đầu), cả ba kênh R, G, B đổi **cùng tỉ lệ**.
- Khi **lượng máu** đổi, ba kênh đổi theo **tỉ lệ khác**; kênh lục đổi mạnh nhất.

Cách tách:
1. Chia mỗi kênh cho trung bình của nó. Thay đổi độ sáng khi đó thành dịch chuyển dọc hướng [1, 1, 1].
2. Chiếu tín hiệu lên mặt phẳng **vuông góc** với hướng [1, 1, 1]. Phần do độ sáng bị triệt, phần do máu còn lại.
3. POS trộn hai trục của mặt phẳng đó theo tỉ lệ độ lệch chuẩn của chúng. CHROM dùng hai tổ hợp màu khác, cùng tinh thần.

Vì không cần huấn luyện, hai phương pháp này không bị lệch miền dữ liệu.

### 2b. CNN của đội

- **TSM** dịch một phần kênh đặc trưng sang khung kề bên, để mạng 2D nhẹ vẫn thấy được thay đổi theo thời gian.
- **Sửa padding.** Bản gốc đệm số 0 ở biên thời gian, tạo ra một dao động giả ở 3 Hz. Đội thay bằng đệm lặp giá trị biên.
- **Tiền lọc 0,1 s (P4m).** Lấy trung bình trượt ⌈fps/10⌉ khung, có trọng số theo cờ hợp lệ, trước khi đưa vào CNN. Nhiễu đổi từng khung nên giảm khoảng √3 lần. Mạch đổi chậm (mỗi nhịp kéo dài ít nhất 10 khung) nên gần như không mất.
- **Chỉ dùng cho bộ công khai.** Trên BTC, CNN một mình sai 17 bpm, kém cả đoán hằng số, vì lệch miền nặng.

### 2c. Ba mô hình huấn luyện sẵn

- **Cách chọn.** Thử các mô hình của open-rppg bằng chọn tiến **lồng trong fold train**: mỗi fold chỉ dùng dữ liệu train để quyết định. Giữ mô hình được chọn ở ≥ 4/5 fold.
- **Kết quả chọn.** FacePhys.rlap 5/5, ME-chunk.pure 5/5, ME-flow.pure 4/5.
- **Kiểm chứng.** Đối chứng xáo nhãn không tạo ra cải thiện.

### Sóng thật của sáu nguồn trên cùng một đoạn 10 s

![Sóng của từng nguồn](figures/waves.png)

Hình trên là một bản ghi MCD, nhịp thật 76 bpm. Cả sáu nguồn đều lặp lại ở 75–76 bpm, nhưng chất lượng khác nhau:
- POS và CHROM có biên độ chập chờn.
- Ba mô hình huấn luyện sẵn cho sóng đều và sạch nhất.
- Sóng thô của CNN trôi dần, vì nó là tổng cộng dồn các độ thay đổi.

Sóng trên mặt có thể **lệch pha hoặc ngược dấu** so với PPG ở ngón tay: máu tới mặt và tới ngón tay vào lúc khác nhau, và mỗi mô hình có quy ước dấu riêng. Điều đó không ảnh hưởng nhịp tim, và cũng là lý do bước kết hợp làm trên phổ chứ không làm trên sóng (xem mục 4).

> **Câu trình bày:** "Mỗi nguồn sai theo cách riêng: công thức vật lý bền nhưng nhạy chuyển động, mô hình học được mạnh nhưng có thể lệch miền. Chúng em dùng cả hai loại."

---

## 3. Readout: từ sóng theo thời gian sang bằng chứng theo nhịp tim

**Mục đích.** Biến **một sóng theo thời gian** (300 số, trục giây) thành **một đường chấm điểm cho từng giá trị nhịp tim** (314 số, trục 42–180 bpm). Mọi sóng đi qua cùng một chuỗi, mỗi sóng xử lý riêng:

| Bước | Làm gì | Vì sao |
|---|---|---|
| 1 | Cửa sổ 10 s, trượt mỗi 5 s; nội suy về 30 Hz theo timestamp | nhịp tim thay đổi theo thời gian; 10 s chứa 7–30 nhịp |
| 2 | Khử xu hướng (smoothness priors, λ = 100) | bỏ trôi chậm do ánh sáng và tư thế |
| 3 | Lọc Butterworth 0,7–3 Hz | chỉ giữ dải 42–180 bpm; bỏ nhịp thở (~0,25 Hz) và nhiễu nhanh |
| 4 | Periodogram, cửa sổ Hann, nfft 4096 | đổi trục từ giây sang tần số: sóng lặp lại mạnh ở đâu |
| 5 | Log, rồi chuẩn hóa z trên 42–180 bpm | đưa mọi nguồn về cùng thang để cộng được |

![Readout trên một cửa sổ thật](figures/readout.png)

Đọc hình readout theo thứ tự:
- **(a)** Màu trung bình thô: trục dọc là thay đổi tương đối (0,02 = 2%). Thấy rõ trôi chậm; ba kênh đi cùng nhau, đó là thay đổi độ sáng; mạch bị chôn bên trong.
- **(b)** Sóng POS trước (xám) và sau (đỏ) khử xu hướng và lọc. Đếm được khoảng 12–13 đỉnh trong 10 s.
- **(c)** Bằng chứng Z(f) **của riêng POS**: đỉnh tại 76 bpm, trùng nhịp tham chiếu. Bướu nhỏ gần 150 bpm là **họa âm**, vì sóng mạch không phải hình sin.
- **(d)** Kết quả của **toàn hệ thống P4m** trên cả bản ghi, bám theo nhịp tham chiếu khi nhịp đổi từ 79 xuống 66 rồi lên 92 bpm. Đây là một bản ghi ánh sáng tốt, chọn để minh họa; số đánh giá chính thức là trên toàn bộ dev.

> **Câu trình bày:** "Bước 1 đến 5 chuyển sóng mạch từ miền thời gian sang miền nhịp tim: mỗi nguồn cho ra một đường chấm điểm cho mọi giá trị từ 42 đến 180 bpm, cùng một thang, để các nguồn rất khác nhau bỏ phiếu chung."

---

## 4. Kết hợp và đọc ra nhịp tim

### Vì sao kết hợp trên phổ, không cộng thẳng các sóng?

Các sóng lệch pha và có cái ngược dấu nhau. Nếu cộng thẳng, đỉnh của sóng này rơi vào đáy của sóng kia và chúng triệt tiêu nhau. Phổ chỉ giữ thông tin "lặp lại nhanh hay chậm", bỏ qua pha, dấu và biên độ, nên các nguồn đều đồng ý ở cùng một chỗ.

![Bằng chứng từng kênh và bản gộp](figures/evidence.png)

### 4a. Bộ công khai: P4m, một nhịp cho mỗi cửa sổ 10 s

1. **Kết hợp.** E(f) = 1·Z_CNN + 0·Z_POS + 0,25·Z_CHROM + 1·mean(Z_FacePhys, Z_ME-chunk, Z_ME-flow).
   - Trọng số chọn trên fold train từ lưới {0; 0,5; 1; 2}⁴. Cả 5 fold và bản deploy ra cùng một bộ.
   - POS về 0 vì khi CNN đã được tiền lọc, POS không thêm thông tin mới.
2. **Làm mượt theo thời gian bằng forward–backward trên HMM.**
   - Trạng thái ẩn: nhịp tim thật ở mỗi cửa sổ.
   - Chuyển trạng thái: phân bố Gauss quanh nhịp cũ, vì nhịp không nhảy vọt giữa hai cửa sổ cách nhau 5 s.
   - Xác suất quan sát: tỉ lệ với exp(τ·E).
   - Trọng số chất lượng: cửa sổ có đỉnh phổ yếu được tin ít hơn và bị các cửa sổ bên cạnh kéo về.
   - Ước lượng: trung vị hậu nghiệm.
   - Tham số (γ, σ, τ) riêng cho từng bộ dữ liệu, fit trên fold train.
3. **Một reader cho mọi điều kiện.** Fit gộp trên native + 6 điều kiện mô phỏng, rồi áp nguyên cho mọi điều kiện, kể cả những điều kiện không có trong fit (C5, C1 gain 0,5 và 0,125, C7 che trắng).

### 4b. BTC: reader_v4, một nhịp cho mỗi video

Đặc điểm dữ liệu BTC:
- nhãn là **một con số cho cả video**, đo bằng máy SpO₂, không đồng bộ thời gian với video;
- khoảng 40% người chỉ có file đã bị ứng dụng nhắn tin nén lại.

Cơ chế:
1. **Bằng chứng cả video:** Z4 = (2·Z_cổ điển + Z_FacePhys + Z_ME-chunk + Z_ME-flow) / 5, trong đó Z_cổ điển là trung bình của Z_POS và Z_CHROM. Lấy trung bình qua mọi cửa sổ.
2. **Bayes:** hậu nghiệm(f) ∝ prior(f) × exp(β·Z4(f)).
   - **Prior** là phân bố nhịp tim của người trong fold train, ước lượng bằng KDE.
   - **β** là mức tin vào video, học riêng cho hai loại file. Loại "thấp" là file có bitrate luồng video < 4 Mb/s.
     - File gốc: β lớn, kết quả bám theo video.
     - File đã nén lại: β gần 0, vì mạch đã bị xóa; kết quả co về prior thay vì bám một đỉnh nhiễu.
3. **Ước lượng:** trung vị hậu nghiệm (tối ưu cho sai số tuyệt đối), cộng hiệu chỉnh theo ô (loại file × Task) để sửa lệch hệ thống, ví dụ nhãn đo sau vận động.

**Vì sao không thấp hơn được.** Biên độ mạch, khoảng 0,3 mức xám, nằm dưới ngưỡng mà bộ nén H.264 làm tròn về 0. Ở file đã nén lại, tín hiệu không còn trong dữ liệu. Kể cả bộ đọc hoàn hảo cũng có trần khoảng 6 bpm; riêng nhãn đo lúc nghỉ đã có sai số khoảng 3,6 bpm.

> **Câu trình bày:** "Khi video không đáng tin, mô hình Bayes tự giảm trọng số bằng chứng và dựa vào phân bố tiên nghiệm, thay vì đưa ra một con số sai một cách tự tin."

---

## 5. Bảy constraint

| Constraint | Vấn đề | Cách xử lý |
|---|---|---|
| C1 thiếu sáng | FaceMesh mất mặt; nhiễu lớn | `aid` (dò trên bản sao tăng sáng), `track`; tiền lọc 0,1 s; HMM tin ít cửa sổ yếu |
| C2 màu da, bạch biến | vùng mất sắc tố có ít tín hiệu | trung bình trên cả oval; nhiều nguồn bằng chứng |
| C3 nhiều người | bộ dò nhảy sang mặt khác | khóa mục tiêu `lock0` |
| C4 độ phân giải thấp | ít điểm ảnh | pipeline vốn chỉ cần 36×36 |
| C5 ≤ 500 kbps, ≤ 10 fps | nén video xóa mạch | chế độ tách: thiết bị chạy frontend, gửi crop không mất mát, khoảng 200–310 kbps |
| C6 da lão hóa | kết cấu da thay đổi | mô phỏng bằng unsharp; không cần sửa |
| C7 che ≥ 40% nửa dưới mặt | FaceMesh mất mặt | `track`: giữ oval theo template phần trên mặt |

## 6. Đánh giá

- Kiểm chéo 5 fold chia theo người. MAE tính theo người trước, rồi trung bình.
- Khoảng tin cậy bằng bootstrap ghép cặp theo người, 10.000 lần, seed 0.
- Ba cổng cho mỗi điều kiện: (a) còn ≥ 90% cửa sổ hoặc video có đầu ra; (b) không kém đoán hằng số; (c) tốt hơn đoán hằng số.
- Đối chứng xáo nhãn cho mọi bước có chọn lựa.
- Holdout BTC và 7 người test của UBFC2 vẫn đóng.
