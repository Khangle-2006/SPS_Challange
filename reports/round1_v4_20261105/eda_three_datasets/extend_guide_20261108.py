from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'SP_Cup_Data_preprocessing_before_rounds_fps_20261108.md').read_text(encoding='utf-8')
rounds='''### Ba vòng thi cần đo những gì?

| Vòng | Đầu ra cần ước lượng từ video RGB | Hiểu đơn giản |
| --- | --- | --- |
| Round 1 | Nhịp mạch / nhịp tim (PR / HR), đơn vị bpm | Mỗi phút có bao nhiêu nhịp |
| Round 2 | HR và SpO₂ | Ngoài số nhịp, cần ước lượng độ bão hòa oxy, đơn vị % |
| Round 3 | HR, SpO₂ và huyết áp (BP) | Thêm hai số huyết áp: tâm thu và tâm trương, đơn vị mmHg |

Round 2 đối chiếu với số đo máy đo mạch/SpO₂. Round 3 yêu cầu kiểm chứng với dữ liệu hoặc thiết bị tham chiếu phù hợp, đồng thời trình diễn trực tiếp/thời gian thực dưới ít nhất hai điều kiện hạn chế. Raspberry Pi là hướng triển khai được khuyến khích, không phải yêu cầu bắt buộc cho mọi đội. Cảm biến dùng để lấy **đáp án tham chiếu**; đầu vào của phương pháp vẫn là video RGB.

Nguồn: [Luật chính thức FaceVitals SP Cup 2027, mục 3.2.2–3.2.3](https://2027.ieeeicassp.org/wp-content/uploads/sites/13/2019/08/2027-SP-Cup-Official-Document-FaceVitals-FINAL-VERSION.pdf).

**Liên hệ với ba bộ dữ liệu:** BTC có các cột PR, SpO₂ và huyết áp; UBFC1 có HR, SpO₂ và sóng PPG; UBFC2 có HR và PPG; AiPEX có HR trung bình. Có tên cột chưa đồng nghĩa nhãn đã đủ chất lượng hoặc đồng bộ thời gian để dùng cho cả ba vòng. Phải đọc cấu trúc và giới hạn của từng bộ bên dưới.

'''
s=s.replace('Ba bộ dữ liệu được đọc',rounds+'Ba bộ dữ liệu được đọc',1)
btc_fps='''### FPS: video chạy bao nhiêu hình mỗi giây?

**fps** là số khung hình mỗi giây. Các số dưới là tốc độ trung bình khai báo trong file đã khảo sát, không phải tốc độ của máy đo mạch.

| Nhóm | Số file | FPS thấp nhất | FPS trung vị | FPS cao nhất |
| --- | ---: | ---: | ---: | ---: |
| Normal | 197 | 19,98 | 29,97 | 240,37 |
| Abnormal | 99 | 19,99 | 29,97 | 30,00 |

**Cách đọc:** một file điển hình có khoảng 30 hình/giây, nhưng không phải tất cả. Normal có file gần 20 và gần 240 hình/giây. Không đổi nhãn bằng cách coi mọi video là 30 fps. Với tốc độ không đều, phải dùng **PTS**, tức mốc thời gian phát của từng khung hình; FPS trung bình không cho biết mọi khoảng cách giữa hai khung đều giống nhau.

'''
s=s.replace('### Nhãn mạch tập trung ở mức nào?',btc_fps+'### Nhãn mạch tập trung ở mức nào?',1)
ubfc_fps='''### FPS của UBFC1 và UBFC2

| Phần dữ liệu đã khảo sát | Số video | FPS thấp nhất | FPS trung vị | FPS cao nhất |
| --- | ---: | ---: | ---: | ---: |
| UBFC1 | 8 | 28,64 | 28,67 | 28,67 |
| UBFC2 phát triển | 35 | 23,21 | 29,69 | 29,98 |

Đây là FPS trung bình theo file video. Không áp các khoảng trên cho 7 video kiểm thử chưa mở. Ví dụ UBFC1 `10-gt` có khoảng **28,67 hình/giây**, còn UBFC2 `subject1` khoảng **29,26 hình/giây**. Tín hiệu PPG có đồng hồ riêng; các dòng cảm biến cách nhau khoảng 0,016 giây trong ví dụ UBFC1 không có nghĩa camera quay 62,5 fps.

'''
s=s.replace('### Điểm cần kiểm trước khi dùng nhãn',ubfc_fps+'### Điểm cần kiểm trước khi dùng nhãn',1)
aipex_fps='''### FPS của các video ROI AiPEX

| Nơi thu | Số file ROI | FPS khai báo trong mọi file |
| --- | ---: | ---: |
| India | 132 | 15 |
| Sierra Leone | 132 | 15 |

15 fps nghĩa là trình phát hiển thị 15 ảnh trong một giây phát. **Chưa xác minh được 1 giây phát luôn bằng 1 giây thu thật.** Vì thế, ghi rõ “15 fps theo file ROI”, không khẳng định camera gốc thu 15 fps và không tự chọn lại FPS để làm kết quả HR gần nhãn hơn.

'''
s=s.replace('### Nhãn HR nằm trong khoảng nào?',aipex_fps+'### Nhãn HR nằm trong khoảng nào?',1)
btc_end='''### Hai video ví dụ và nơi mở dữ liệu BTC

Hai ví dụ đều thuộc phần phát triển. Chúng là **hai lần quay của cùng S_N1**, giúp nhìn sự khác nhau giữa Resting và Task; không phải mẫu đại diện cho toàn bộ nhóm Normal/Abnormal. Bản xem nhanh chỉ lấy 10 giây đầu, thu nhỏ ảnh, bỏ âm thanh để dễ phát; không dùng bản này làm đầu vào đo mạch.

| Ví dụ | Bản nguồn | FPS nguồn | Điều cần quan sát |
| --- | --- | ---: | --- |
| [BTC_S_N1_Resting.mp4](examples_20261108/BTC_S_N1_Resting.mp4) | S_N1_Resting.mov; 30,765 giây | 29,97 | Toàn khuôn mặt lúc nghỉ; một file đi với một dòng PR |
| [BTC_S_N1_Task1.mp4](examples_20261108/BTC_S_N1_Task1.mp4) | S_N1_Task1.MOV; 30,498 giây | 29,97 | Cùng người nhưng một lần quay Task1 khác |

**Thư mục trên máy P100:**

```text
/mnt/samsung_t5/khangle/facevitals/datasets/FaceVitals_BTC_20261107/
```

Trong thư mục này, mở `Normal/` hoặc `Abnormal/`. Bản Resting ở `Normal/1k3Te1Px2Pd2Swtl8yPhtqK-q2SlcRZ2l/S_N1_Resting.mov`; bản Task1 ở `Normal/1z-r8EEFhmiDyUkeYWGzf2bkgxYAGTho1/S_N1_Task1.MOV`.

**Mở nguồn trên web:** [thư mục Normal](https://drive.google.com/drive/folders/1AAUGnu4uwFUQKgzar0HahgztZSZmSjZV) · [thư mục Abnormal](https://drive.google.com/drive/folders/1t_A3cQFFgAdcBql_9FlM96-TW44WAo0z) · [bảng nhãn BTC](https://docs.google.com/spreadsheets/d/1aZbks-8KG_mDc2Z7zwiirge0Fk-eexP5/edit).

Đường dẫn bắt đầu bằng `/mnt/` hoặc `/data_pool/` là trên **máy chủ P100**, không phải ổ C: của Windows. Kết nối P100 bằng trình duyệt file SSH/WinSCP hoặc VS Code Remote SSH, rồi dán đường dẫn vào. Link web phía trên mở được bằng trình duyệt khi tài khoản có quyền xem.

'''
ubfc_end='''### Hai video ví dụ và nơi mở dữ liệu UBFC

Hai bản xem nhanh dài khoảng 10 giây, thu nhỏ ảnh và bỏ âm thanh. Một ví dụ từ mỗi phần giúp đối chiếu hai cách lưu nhãn; đây không phải hai bản lấy từ tập kiểm thử giữ kín.

| Ví dụ | Video nguồn | FPS nguồn | File nhãn nằm cạnh video |
| --- | --- | ---: | --- |
| [UBFC1_10-gt.mp4](examples_20261108/UBFC1_10-gt.mp4) | DATASET_1/10-gt/vid.avi; 42,760 giây | 28,67 | gtdump.xmp: thời gian, HR, SpO₂, PPG |
| [UBFC2_subject1.mp4](examples_20261108/UBFC2_subject1.mp4) | DATASET_2/subject1/vid.avi; 52,863 giây | 29,26 | ground_truth.txt: ba dòng PPG, HR, thời gian |

**Thư mục trên máy P100:**

```text
/data_pool/khangle/facevitals/datasets/UBFC-rPPG/
```

Ghép đường dẫn trên với `DATASET_1/10-gt/` hoặc `DATASET_2/subject1/` để mở đúng thư mục ví dụ. Mỗi thư mục có cả video và nhãn. [Trang phát hành UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg) cung cấp thông tin và nguồn dữ liệu; đường dẫn P100 cần mở qua kết nối máy chủ.

'''
aipex_end='''### Hai video ROI ví dụ và nơi mở dữ liệu AiPEX

Hai bản xem nhanh dưới đây lấy từ **cùng India/video1**: một video trán và một video má trái. Mỗi bản lấy 10 giây đầu từ file dài 32 giây. Mục đích là thấy ảnh ROI chỉ chứa một phần da, không phải toàn khuôn mặt; hai ROI vẫn dùng chung nhãn HR 85,413 bpm.

| Ví dụ | Vùng ảnh | FPS theo file nguồn | Đường dẫn tương đối từ thư mục AiPEX |
| --- | --- | ---: | --- |
| [AiPEX_India_video1_forehead.mp4](examples_20261108/AiPEX_India_video1_forehead.mp4) | Trán | 15 | Data/India/video1_forehead.avi |
| [AiPEX_India_video1_leftcheek.mp4](examples_20261108/AiPEX_India_video1_leftcheek.mp4) | Má trái | 15 | Data/India/video1_leftcheek.avi |

**Thư mục trên máy P100:**

```text
/data_pool/khangle/facevitals/datasets/AiPEX/
```

Video ở `Data/India/` và `Data/Sierra_Leone/`; nhãn ở `Ground_Truth/India_GT.csv` và `Ground_Truth/Sierra_Leone_GT.csv`. [Kho AiPEX của nhóm tác giả](https://github.com/AiPEX-Lab/rppg_biases) là điểm truy cập nguồn. Đường dẫn P100 cần mở qua kết nối máy chủ.

'''
s=s.replace('## 2. UBFC-rPPG',btc_end+'## 2. UBFC-rPPG',1).replace('## 3. AiPEX',ubfc_end+'## 3. AiPEX',1).replace('## 4. Từ một video',aipex_end+'## 4. Từ một video',1)
glossary='''## 6. Bảng tra ký hiệu và thuật ngữ

Không cần nhớ hết bảng. Khi gặp một từ lạ, tìm ở cột bên trái rồi đọc nghĩa và ví dụ bên phải. **SpO₂** (cũng viết SpO2) là tên đúng của chỉ số oxy, không phải “SP2”.

### Nhãn sinh lý và đơn vị

| Ký hiệu / tên cột | Nghĩa dễ hiểu | Ví dụ và lưu ý |
| --- | --- | --- |
| PR, pulse rate | Nhịp mạch | PR = 104 bpm: 104 nhịp mỗi phút |
| HR, heart rate | Nhịp tim | PR và HR liên quan chặt nhưng là hai tên đại lượng đo; giữ đúng tên nguồn nhãn |
| bpm | Beats per minute: nhịp/phút | 60 bpm tương ứng trung bình một nhịp mỗi giây |
| PPG | Sóng quang đo bằng cảm biến tiếp xúc, thay đổi theo mạch | PPG = 40 là độ lớn tín hiệu ở một thời điểm, không phải HR = 40 |
| rPPG | Ước lượng tín hiệu mạch từ thay đổi màu da trong video | Chữ r là remote: đo từ xa bằng camera |
| SpO₂ / SpO2 | Độ bão hòa oxy ngoại vi do máy đo ước lượng, đơn vị % | 97 nghĩa là 97%; không phải 97 bpm hoặc lượng oxy tính bằng ml |
| BP | Blood pressure: huyết áp | Thường gồm hai số tâm thu và tâm trương |
| BP-SYS / SBP | Huyết áp tâm thu | Số thứ nhất trong cách ghi 120/80 |
| BP-DIA / DBP | Huyết áp tâm trương | Số thứ hai trong cách ghi 120/80 |
| mmHg | Milimét thủy ngân, đơn vị huyết áp | 120/80 mmHg chỉ là ví dụ cách ghi, không phải nhãn mọi video |
| T / Temperature | Nhiệt độ | Cần đọc đơn vị cột; °C và °F là hai thang khác nhau |
| Nhãn / ground truth / GT | Số đo hoặc tín hiệu dùng làm đáp án đối chiếu | BTC có số PR theo lần quay; UBFC có cả chuỗi theo thời gian |

### Video, vùng ảnh và đồng hồ

| Thuật ngữ | Nghĩa dễ hiểu | Cách dùng trong tài liệu |
| --- | --- | --- |
| RGB | Ba kênh màu đỏ, xanh lá, xanh dương | Video màu thông thường là đầu vào của bài thi |
| Frame / khung hình | Một ảnh trong chuỗi video | 300 ảnh không tự động là 10 giây nếu chưa biết thời gian |
| fps / FPS | Số khung hình mỗi giây | 30 fps: khoảng 30 ảnh/giây; khác số mẫu PPG/giây |
| Hz | Số chu kỳ hoặc số mẫu mỗi giây, tùy ngữ cảnh | Tần số mạch 1,5 Hz tương ứng 1,5 × 60 = 90 bpm |
| s / ms | Giây / mili giây | 1 giây = 1.000 mili giây |
| Timestamp / PTS | Mốc thời gian của mẫu / khung hình | Dùng chọn đúng đoạn 10–20 giây; không chỉ đếm ảnh |
| ROI | Region of interest: vùng ảnh được chọn để phân tích | Có thể là oval toàn mặt, trán hoặc một bên má |
| Forehead / leftcheek / rightcheek | Trán / má trái / má phải | Ba tên ROI trong file AiPEX |
| Pixel / độ phân giải | Điểm ảnh / số điểm ảnh ngang × dọc | 36 × 36 là 1.296 điểm ảnh mỗi khung, không phải 36 fps |
| Codec, H.264, HEVC / H.265 | Cách mã hóa/nén video | Đuôi MOV/MP4/AVI là loại file chứa video, không xác định duy nhất codec |
| Bitrate, kb/s, Mb/s | Lượng dữ liệu video mỗi giây | 500 kb/s = 0,5 Mb/s; không phải 500 khung/giây |
| HDR / SDR / HLG | Dải sáng rộng / dải sáng chuẩn / một cách biểu diễn HDR | Cần quy tắc chuyển màu thống nhất trước khi phân tích màu da |
| 8-bit / 10-bit | Số bit biểu diễn mức của một thành phần ảnh | Khác độ phân giải và FPS; 10-bit không có nghĩa 10 hình/giây |
| Rotation metadata | Thông tin xoay được ghi trong file | Phải áp đúng để mặt đứng thẳng khi đọc ảnh |
| Native / C5 | Bản nguồn / bản thử nghiệm nén 10 fps, mục tiêu ≤500 kb/s | Native vẫn có thể đã được camera nén; C5 không phải một người mới |

### Mã dữ liệu, nhóm và cách đọc số liệu

| Thuật ngữ / ký hiệu | Nghĩa dễ hiểu | Ví dụ / điều không nên nhầm |
| --- | --- | --- |
| BTC | Ban tổ chức | Bộ BTC có video và bảng nhãn theo lần quay |
| ID / subject / mã người | Khóa để nhận diện bản ghi hoặc người; đọc theo ngữ cảnh | S_N1 là người; S_N1_Resting là một bản ghi của người đó |
| Recording / lần quay | Một lần thu video | Một người có thể có Resting và nhiều Task |
| Normal / Abnormal | Hai nhóm do nguồn BTC đặt | Không đồng nghĩa train / test; không suy ra chẩn đoán chỉ từ tên nhóm |
| PROTOCOL; Resting; Task; Walking | Điều kiện quay; nghỉ; nhiệm vụ; đi bộ | Task1/2/3 là số nhiệm vụ, chưa mô tả đầy đủ hành động |
| Bright / Med / Dim | Sáng / ánh sáng trung bình / tối | Med ở tên ánh sáng khác Medium Fair ở cột tông da |
| Fair / Medium Fair / Dark | Các nhãn tông da do nguồn ghi | Giữ nguyên cách phân nhóm của nguồn; không thay bằng quốc tịch |
| India / Sierra Leone | Hai nơi thu AiPEX: Ấn Độ / Sierra Leone | video1 của hai nơi là hai bản khác nhau |
| Train / dev / test / holdout | Tập học / phát triển / kiểm thử / phần giữ riêng | Holdout chỉ dùng kiểm tra cuối, không dùng chọn cách xử lý |
| Window / cửa sổ | Một đoạn thời gian lấy từ video | 0–10 giây và 10–20 giây là hai cửa sổ không chồng nhau |
| n / N | Số đối tượng được đếm; phải đọc đơn vị đi kèm | n = 10 video khác n = 10 người |
| Trung bình | Tổng các giá trị chia số giá trị | (1 + 3 + 8) ÷ 3 = 4 |
| Trung vị / median | Giá trị giữa sau khi xếp thứ tự | Với 1, 3, 8: trung vị = 3; không giống trung bình = 4 |
| Min / max | Nhỏ nhất / lớn nhất | FPS min không phải FPS của mọi file |
| MAE | Trung bình độ lớn sai số, cùng đơn vị với nhãn | Sai số 1, 3, 5 bpm cho MAE = 3 bpm; càng thấp càng tốt |
| Dấu giá trị tuyệt đối | Bỏ dấu âm của chênh lệch | Dự đoán 103, nhãn 104: độ lớn của 103 − 104 bằng 1 |
| % / tỷ lệ | Phần trong tổng số, nhân 100 | 105 ÷ 194 × 100 ≈ 54%; phải nói rõ tổng là video hay người |
| CSV / TXT / XMP / XLSX | Các kiểu file nhãn/bảng số liệu | XMP của UBFC1 trong bộ này chứa bốn cột; TXT UBFC2 chứa ba dòng |

Các video xem nhanh đi kèm nằm trong thư mục `examples_20261108` cạnh bản Markdown/PDF. PDF không phát video bên trong trang: mở file MP4 đi kèm hoặc xem khối video trên Notion. Khi chuyển tài liệu sang máy khác, giữ cả thư mục ví dụ để các liên kết tương đối còn dùng được.

'''
s=s.replace('## Nguồn đối chiếu',glossary+'## Nguồn đối chiếu',1)
s += '\n- FPS lấy từ cột `avg_fps` của `media.csv`, không đo lại phần holdout. Ví dụ video và nguồn gốc: `examples_20261108/MANIFEST.json`.\n- Cập nhật hướng dẫn Round 2–3, FPS, video ví dụ và bảng thuật ngữ ngày 08/11/2026; số lượng/thời lượng vẫn là mốc khảo sát 07/11.\n'
(root/'SP_Cup_Data_preprocessing_readable_20261108.md').write_text(s,encoding='utf-8')
print('Updated source',len(s))
