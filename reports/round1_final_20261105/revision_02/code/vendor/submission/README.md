# reports/round1_final_20261105/revision_02/code/vendor/submission

Snapshot mã nghiên cứu; xem root README để xác định vai trò và kết quả.

[Hướng dẫn chạy](../../../../../../docs/REPRODUCING.md) · [Kết quả](../../../../../../docs/RESULTS.md) · [Experiment map](../../../../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [__init__.py](__init__.py) | Không có argparse; đọc source/config |
| [mat_writer.py](mat_writer.py) | Không có argparse; đọc source/config |
| [metrics.py](metrics.py) | Không có argparse; đọc source/config |
| [process_round1.py](process_round1.py) | `--constraints`, `--constraint-manifest`, `--dataset`, `--root`, `--recording`, `--audit-dir`, `--subject-id`, `--reference-clock-shared`, `--video-timestamps`, `--checkpoint`, `--roi-cache`, `--external-video-reference`, `--protocol` |
| [run_e2e.py](run_e2e.py) | `--dataset`, `--root`, `--audit-dir`, `--recording`, `--out`, `--roi-cache` |
| [test_submission.py](test_submission.py) | Không có argparse; đọc source/config |
