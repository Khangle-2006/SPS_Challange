# ops/a100

Research code snapshot; see the root README for its role and results.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [audit_label_values.py](audit_label_values.py) | No argparse interface; read the source/config |
| [check_report.py](check_report.py) | No argparse interface; read the source/config |
| [compact_ubfc2_20261030.py](compact_ubfc2_20261030.py) | No argparse interface; read the source/config |
| [encode_cv_secondary.py](encode_cv_secondary.py) | `--rate` |
| [inspect_h264_roi_jump.py](inspect_h264_roi_jump.py) | No argparse interface; read the source/config |
| [inspect_h264_rois.py](inspect_h264_rois.py) | No argparse interface; read the source/config |
| [pilot_ffv1_20261030.py](pilot_ffv1_20261030.py) | `--subject`, `--replace-verified` |
| [pos_h264_validation.py](pos_h264_validation.py) | `--split` |
| [prepare_efficientphys.py](prepare_efficientphys.py) | No argparse interface; read the source/config |
| [prepare_efficientphys_test.py](prepare_efficientphys_test.py) | No argparse interface; read the source/config |
| [prepare_end_to_end_inputs.py](prepare_end_to_end_inputs.py) | No argparse interface; read the source/config |
| [prepare_h264_rates.py](prepare_h264_rates.py) | `--rate` |
| [prepare_h264_training.py](prepare_h264_training.py) | No argparse interface; read the source/config |
| [prepare_h264_validation.py](prepare_h264_validation.py) | No argparse interface; read the source/config |
| [prepare_roi_ablation.py](prepare_roi_ablation.py) | `--stage` |
| [run_ubfc_cpu.py](run_ubfc_cpu.py) | No argparse interface; read the source/config |
| [verify_source_checksums.py](verify_source_checksums.py) | No argparse interface; read the source/config |
