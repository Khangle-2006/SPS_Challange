# ops/t9_fps_codec_v2_20261031

35/35 UBFC; report kết quả T9 giữ riêng.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [data_adapters.py](data_adapters.py) | Không có argparse; đọc source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [oval_frontend.py](oval_frontend.py) | Không có argparse; đọc source/config |
| [qa_mask.py](qa_mask.py) | Không có argparse; đọc source/config |
| [report_t9.py](report_t9.py) | `--input`, `--output` |
| [run_t5_qa.py](run_t5_qa.py) | `--audit`, `--dataset`, `--watch`, `--limit` |
| [run_t9.py](run_t9.py) | `--limit` |
| [score_round1.py](score_round1.py) | `--reference`, `--output`, `--metrics` |
| [simulate.py](simulate.py) | Không có argparse; đọc source/config |
| [t8_t1_loader.py](t8_t1_loader.py) | Không có argparse; đọc source/config |
| [test_t9.py](test_t9.py) | Không có argparse; đọc source/config |
| [window_support.py](window_support.py) | Không có argparse; đọc source/config |
