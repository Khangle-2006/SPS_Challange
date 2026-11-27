# Round 1 packaging — current P4m R1

Use [`predict_public.py`](../predict_public.py) for raw public video and [`docs/SUBMISSION_ROUND1.md`](../docs/SUBMISSION_ROUND1.md) for the checked package workflow. Install the locked frontend/experts/core environments plus `torch-cpu`. DEV uses the matching `--fold`; deployment uses the fixed convention documented there.

`build_public_baseline.py` requires all three raw-video parity gates to pass, exports the six-variable MAT and verifies decoded frames with mat73. `process_public_round1.py` and `.m` are installed in the private bundle as `process_round1.py` / `process_round1.m`. The `.m` wrapper invokes Python, using relative package paths and an explicit repository location; it does not claim a native MATLAB implementation.

MAT dependencies: `h5py==3.11.0`, `hdf5storage==0.2.2`, `mat73==0.65` in addition to numpy/scipy/OpenCV. `requirements.txt` documents the historical full environment; do not overwrite an existing locked runtime environment to install packaging dependencies.

`run_holdout.py` and `score_holdout.py` are separate, sealed stages. Only 3 explicitly DEV inputs are permitted without the exact user confirmation token enforced by the stage runner. This historical package records the holdout as unopened.

Historical checkpoint/Haar36 code is copied intact to [historical/](historical/). The old top-level `process_round1.py`, `run_e2e.py` and support modules remain for compatibility and existing tests; they are **not** the current P4m bundle entry point. See [SCHEMA](SCHEMA.md) for representation details and limitations. No media, MAT, personal labels or spectra are distributed in this repository.
