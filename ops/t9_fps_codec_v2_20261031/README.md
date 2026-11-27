# ops/t9_fps_codec_v2_20261031

35/35 UBFC; the T9 results report is kept separately.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [data_adapters.py](data_adapters.py) | No argparse interface; read the source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [oval_frontend.py](oval_frontend.py) | No argparse interface; read the source/config |
| [qa_mask.py](qa_mask.py) | No argparse interface; read the source/config |
| [report_t9.py](report_t9.py) | `--input`, `--output` |
| [run_t5_qa.py](run_t5_qa.py) | `--audit`, `--dataset`, `--watch`, `--limit` |
| [run_t9.py](run_t9.py) | `--limit` |
| [score_round1.py](score_round1.py) | `--reference`, `--output`, `--metrics` |
| [simulate.py](simulate.py) | No argparse interface; read the source/config |
| [t8_t1_loader.py](t8_t1_loader.py) | No argparse interface; read the source/config |
| [test_t9.py](test_t9.py) | No argparse interface; read the source/config |
| [window_support.py](window_support.py) | No argparse interface; read the source/config |
