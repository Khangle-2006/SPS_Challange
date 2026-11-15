# reports/round1_v4_20261105/btc_zeroshot

Dev-only preprocessing/inference, fold-safe calibration, diagnostics.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [build_split.py](build_split.py) | Không có argparse; đọc source/config |
| [control_summary.py](control_summary.py) | Không có argparse; đọc source/config |
| [finalize_dev.py](finalize_dev.py) | Không có argparse; đọc source/config |
| [infer_dev.py](infer_dev.py) | Không có argparse; đọc source/config |
| [preprocess.py](preprocess.py) | `--limit` |
| [preprocess_pilot_v1.py](preprocess_pilot_v1.py) | `--limit` |
| [report_watch.py](report_watch.py) | Không có argparse; đọc source/config |
| [summarize_dev.py](summarize_dev.py) | Không có argparse; đọc source/config |
| [test_contract.py](test_contract.py) | Không có argparse; đọc source/config |
| [test_gate.py](test_gate.py) | Không có argparse; đọc source/config |
| [test_inference.py](test_inference.py) | Không có argparse; đọc source/config |
| [test_summary.py](test_summary.py) | Không có argparse; đọc source/config |
| [validate_media.py](validate_media.py) | Không có argparse; đọc source/config |
