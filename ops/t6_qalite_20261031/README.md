# ops/t6_qalite_20261031

Research code snapshot; see the root README for its role and results.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [hr_baseline.py](hr_baseline.py) | `--root`, `--out`, `--self-test` |
| [mcd_e0_metrics.py](mcd_e0_metrics.py) | No argparse interface; read the source/config |
| [mcd_errata_audit.py](mcd_errata_audit.py) | `--root`, `--audit`, `--fold-csv` |
| [qa_mask.py](qa_mask.py) | No argparse interface; read the source/config |
| [run_e0_qa.py](run_e0_qa.py) | `--limit` |
| [test_qa_mask.py](test_qa_mask.py) | No argparse interface; read the source/config |
