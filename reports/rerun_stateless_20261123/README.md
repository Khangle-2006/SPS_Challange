# R1: isolate open-rppg state, 2026-11-23

Scientific code and aggregate tables from the run covering 63 model-condition combinations. See the [results report](../../docs/RESULTS.md). The CSV files contain no individual labels or predictions. `PUBLIC_FITS.json` records all fold-specific weights/readouts for P4, P4m, N4, and N4m. P4m deployment parameters are in `models/public_p4m_r1.json` at the repository root.

These scripts are snapshots of the P100 run and require private manifests, crops, and caches at the paths specified in the code. **Do not run them against an existing output directory**. They are not commands for running a new video; use `predict.py` at the root for that purpose. Private assets are not included on GitHub, and the holdout must not be read.

Execution order: `prepare.py` locks the cohort/hashes → check reset/self-warm/A-B-A behavior → `worker.py` runs inference for each condition → `score_btc.py` and `public_score_wrapper.py` → `public_freeze_wrapper.py` → `report.py`. Only the spectral paths in the z13/z14 snapshots were changed; the grid and operators were unchanged. Native BTC reused all 218 z17 spectra after the decision to ADOPT self-warm.

Original R1 deployment directory: `/mnt/samsung_t5/khangle/facevitals/reports/rerun_stateless_20261123`. Inference uses W's `venv_orppg`; scoring uses `/data_pool/khangle/facevitals/envs/hr`. Runs use CPU, nice19, and two BLAS threads, with RAM/disk checks before launch. No training or holdout access. Orchestration scripts and receipts containing PIDs, private paths, or individual outputs are kept outside the repository.

Earlier results with state carried across videos are audit columns only, not valid performance baselines. The selection of 3 out of 17 models using the old spectra has not been repeated.
