# ops/deep_hr/reports/end_to_end_detection_20261019/code/deep_hr

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../../../docs/REPRODUCING.md) · [Kết quả](../../../../../../docs/RESULTS.md) · [Experiment map](../../../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [EfficientPhys_fixed.py](EfficientPhys_fixed.py) | Không có argparse; đọc source/config |
| [evaluate_end_to_end_detection.py](evaluate_end_to_end_detection.py) | Không có argparse; đọc source/config |
| [evaluate_spectral_ensemble.py](evaluate_spectral_ensemble.py) | Không có argparse; đọc source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [inspect_end_to_end_rois.py](inspect_end_to_end_rois.py) | Không có argparse; đọc source/config |
| [spectral_ensemble.py](spectral_ensemble.py) | Không có argparse; đọc source/config |
| [train_efficientphys.py](train_efficientphys.py) | `--root`, `--out`, `--config`, `--smoke-only`, `--resume` |
| [train_h264_ablation.py](train_h264_ablation.py) | `--mode`, `--seed` |
| [verify_spectral_ensemble.py](verify_spectral_ensemble.py) | Không có argparse; đọc source/config |
| [verify_video_roi_frontend.py](verify_video_roi_frontend.py) | Không có argparse; đọc source/config |
| [video_roi_frontend.py](video_roi_frontend.py) | Không có argparse; đọc source/config |
