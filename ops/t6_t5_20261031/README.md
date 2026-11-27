# ops/t6_t5_20261031

Research code snapshot; see the root README for its role and results.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [data_adapters.py](data_adapters.py) | No argparse interface; read the source/config |
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [qa_mask.py](qa_mask.py) | No argparse interface; read the source/config |
| [run_t5_qa.py](run_t5_qa.py) | `--audit`, `--dataset`, `--watch`, `--limit` |
| [score_round1.py](score_round1.py) | `--reference`, `--output`, `--metrics` |
| [t8_t1_loader.py](t8_t1_loader.py) | No argparse interface; read the source/config |
| [test_adapter.py](test_adapter.py) | No argparse interface; read the source/config |
| [test_qa_mask.py](test_qa_mask.py) | No argparse interface; read the source/config |
| [window_support.py](window_support.py) | No argparse interface; read the source/config |
