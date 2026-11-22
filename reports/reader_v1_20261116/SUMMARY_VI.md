# Reader v1 — kết quả 16/11

Tái lập đúng toàn bộ218 dự đoán. Chưa đạt cổng MAE≤10 và cải thiện≥1bpm so với hiệu chỉnh có CI loại0. Holdout chưa mở.

| Điều kiện | Video/người | MAE | Bias | Spearman | Δ hiệu chỉnh [CI95%] |
|---|---:|---:|---:|---:|---|
|native|218/158|10.570|-3.796|0.491|-0.804 [-1.829;+0.223]|
|c5|218/158|12.140|-0.908|-0.114|+0.766 [-0.023;+1.520]|
|c5prime_default|161/109|13.041|-0.372|-0.258|+1.621 [+0.618;+2.595]|
|c5prime_bf0|158/106|13.777|-2.487|-0.135|+2.203 [+1.078;+3.318]|

Các delta dùng baseline native đã lưu trên cùng video. Không phải baseline hiệu chỉnh được tính lại từ video nén. C5′ default và bf0 khác độ phủ; không so chéo thành kết luận nhân quả. C5′ là proxy ffmpeg, chưa xác minh MATLAB.

Ảnh tĩnh20/20PASS: trả prior+abstain. Xáo nhãn20lần:MAE trung bình11.873, khoảng11.684–12.076.
Octave đãPASS5/5videoROIthật, sai khác0bpm; chưa chạyMATLABthật. Ghép phổ W2 chờ đủ5foldhợp lệ.