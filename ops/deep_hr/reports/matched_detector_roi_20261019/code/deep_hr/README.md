# ops/deep_hr/reports/matched_detector_roi_20261019/code/deep_hr

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../../../docs/REPRODUCING.md) · [Kết quả](../../../../../../docs/RESULTS.md) · [Experiment map](../../../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [evaluate_matched_detector_roi.py](evaluate_matched_detector_roi.py) | Không có argparse; đọc source/config |
| [prepare_matched_roi_training.py](prepare_matched_roi_training.py) | Không có argparse; đọc source/config |
| [train_h264_ablation.py](train_h264_ablation.py) | `--mode`, `--seed` |
| [train_matched_detector_roi.py](train_matched_detector_roi.py) | `--arm`, `--seed` |
| [verify_matched_detector_roi.py](verify_matched_detector_roi.py) | Không có argparse; đọc source/config |
| [video_roi_frontend.py](video_roi_frontend.py) | Không có argparse; đọc source/config |
