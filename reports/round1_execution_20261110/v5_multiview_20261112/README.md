# reports/round1_execution_20261110/v5_multiview_20261112

Offset eligibility, augmentation, train_v2.py, evaluation. Final V5 không qua BTC gate.

[Hướng dẫn chạy](../../../docs/REPRODUCING.md) · [Kết quả](../../../docs/RESULTS.md) · [Experiment map](../../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [attenuate.py](attenuate.py) | Không có argparse; đọc source/config |
| [codec_common.py](codec_common.py) | Không có argparse; đọc source/config |
| [codec_common_v2.py](codec_common_v2.py) | Không có argparse; đọc source/config |
| [dispatch.py](dispatch.py) | Không có argparse; đọc source/config |
| [dispatch_v2.py](dispatch_v2.py) | Không có argparse; đọc source/config |
| [dispatch_v3.py](dispatch_v3.py) | Không có argparse; đọc source/config |
| [fold_ready.py](fold_ready.py) | Không có argparse; đọc source/config |
| [offsets.py](offsets.py) | Không có argparse; đọc source/config |
| [offsets_v2.py](offsets_v2.py) | Không có argparse; đọc source/config |
| [prepare_aug.py](prepare_aug.py) | Không có argparse; đọc source/config |
| [prepare_sideviews.py](prepare_sideviews.py) | Không có argparse; đọc source/config |
| [provider.py](provider.py) | Không có argparse; đọc source/config |
| [public_summary.py](public_summary.py) | Không có argparse; đọc source/config |
| [recover_aug_20261112.py](recover_aug_20261112.py) | Không có argparse; đọc source/config |
| [report.py](report.py) | Không có argparse; đọc source/config |
| [runtime.py](runtime.py) | Không có argparse; đọc source/config |
| [train.py](train.py) | Không có argparse; đọc source/config |
| [train_v2.py](train_v2.py) | Không có argparse; đọc source/config |
| [validate.py](validate.py) | `--fold` |
| [validate_v2.py](validate_v2.py) | `--fold` |
