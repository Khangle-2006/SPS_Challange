# Reproduction guide

**Fresh setup:** use [GETTING_STARTED.md](GETTING_STARTED.md), [ASSETS.md](ASSETS.md), and [clean-environment validation](CLEAN_ENV_VALIDATION.md). Sections below preserve historical snapshot context; requirements-research.txt does not replace the four new profiles.

## 1. Choose a reproduction level

| Level | Available from this repository? | Additional requirements |
|---|---|---|
| Source/hash/syntax and synthetic scorer checks | Yes | Python, NumPy, SciPy |
| POS baseline from your RGB CSV | Yes | Correct timestamps and valid mask; no labels needed |
| MATLAB/Octave POS from ROI MAT | Code available | Octave or MATLAB; `roi_data` as specified below |
| CPU/GPU public-checkpoint inference | Historical code available | Checkpoints, cache/source, frontend receipts, PyTorch environment |
| Exact BTC/public table reproduction | Algorithms/report code available, **not a one-command package** | Authorized data, splits/manifests/references, checkpoints, original hash receipts |
| Retrain sweeps | Code snapshots available | All above plus Linux/FFmpeg/GPU, preregistration, and a separate budget |

Do not rerun controllers in the original project. Team experiments have ended; authorized users should create isolated reproduction environments. Do not disable assertions to bypass manifest/hash/split mismatches.

## 2. Environment

The historical README quick start uses only NumPy/SciPy. Research environments are mainly Linux (`fcntl`, `taskset`, `nice`, `du`, FFmpeg/ffprobe); Windows can run the baseline but not every Linux controller automatically.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt
ffmpeg -version
ffprobe -version
python -c "import torch, cv2, av, mediapipe; print(torch.__version__)"
```

`requirements-research.txt` is a starting list, **not a clean environment validated across all branches**. Original `requirements*.txt` files remain in their directories; select the experiment-matched version. `submission/requirements.txt` uses CPU Torch wheels; GPU requires CUDA/driver-compatible wheels. Do not silently substitute and claim bitwise parity.

## 3. Prepare authorized assets

Repository scripts do not download data. Obtain data from owners under their terms. Keep raw inputs immutable and outputs separate.

| Asset | Required contents | Consumer |
|---|---|---|
| UBFC2 | `subject*/vid.avi`, `ground_truth.txt` | `ops/deep_hr/data_adapters.py` |
| MCD | Camera videos, original PPG, mapping tables, clock receipts | Same adapter; do not infer timestamps from container FPS alone |
| BTC dev | Videos, PR table, person mapping, frozen `btc_split.json` | `reports/round1_v4_20261105/btc_zeroshot/` |
| AiPEX | ROI and mean HR for the correct person/recording | `ops/aipex_diagnostic_20261031/`, `aipex_data.py` |
| T8 oval cache | `crop/rgb.npy`, `meta.npz`, `manifest.json`, `windows.json`, per-condition receipts | `ops/t8_oval_20261031/prepare_t8_cache.py` |
| Training index | Person folds, cache hashes, eligibility, schedule | `ops/t8_oval_training_20261031/t8_plan.py` |
| Checkpoint | Final-epoch `final.pt` + configuration/provenance/frontend hashes | Evaluator matching the exact model version |
| Evaluation artifacts | Reference canonical + raw copy, prediction CSV, failure/support masks | `t8_evaluate.py`, aggregate/report code |

Checkpoints are absent from Git. This historical snapshot had no verified public download. Obtain an authorized asset bundle from the maintainer before full reproduction. Do not reuse fold 0 across folds or include validation folds/labels in fitting/calibration/priors.

Legacy snapshots use `/mnt/samsung_t5/khangle/facevitals`, `/data_pool/khangle/facevitals`, and `/data/khangle/facevitals`. On a reproduction machine, use the corresponding layout in isolation or create an **explicitly documented new configuration**. `code/t8_oval_20261031` is the training bundle for `ops/t8_oval_training_20261031`; `code/t8_polarity_v3_20261102` corresponds to `ops/t8_polarity_v3_20261102`. Preserve report-directory relationships. Do not change code/paths while presenting old frozen hashes as unchanged.

## 4. Run the standalone baseline and scorer

```bash
python tools/pos_trace.py --input /path/to/rgb.csv --output output/prediction.csv
python score_round1.py --help
python selection_round1.py --help
python -m unittest test_score_round1
python -m unittest discover -s tools -p 'test_*.py'
```

The scorer is an internal tool, not the organizers' `app.py`. Scorer inputs:

```text
reference_signal.csv:
  subject_id,recording_id,window_start_s,video_support_valid,
  ppg_timestamps_json,ppg_values_json
processed_output.csv:
  subject_id,recording_id,window_start_s,hr_bpm,abstain
```

Create reference manifests independently of predictions. Report missing/invalid windows; do not select favorable intersections. The public waveform scorer is not directly a BTC video-level PR-label scorer.

## 5. T8 / V3 / padding-fixed

Flow: raw + PPG → adapter/clock → canonical windows → full-frame encoding → oval crop36 → cache/index → person-fold training → final checkpoint → inference → cumsum/detrend/filter/HR → aggregation.

Actual CLI examples from the code (run only after preparing inputs and guards):

```bash
python ops/t8_oval_20261031/prepare_t8_cache.py --dataset UBFC2 \
  --root /authorized/UBFC2 --output /scratch/cache/UBFC2 \
  --reuse-root /scratch/legacy-cache --budget-bytes 6000000000 --smoke
python ops/t8_oval_training_20261031/t8_plan.py \
  --ubfc /scratch/cache/UBFC2 --mcd /scratch/cache/MCD \
  --folds /authorized/folds_frozen.json --output /scratch/training_index.json
```

These commands illustrate the schema; **example paths are not confirmed to exist**. MCD needs `--audit` pointing to valid receipts. Original T8 `t8_run.py` contains an old queue; do not substitute it for polarity-corrected V3 or final padding-fixed runs. Use the [experiment index](EXPERIMENTS.md) to select the correct runner.

- V3: `ops/t8_polarity_v3_20261102/`; reuse 15 controls and train 15 polarity-corrected joint runs. Source tests are in the same folder. Raw/canonical reference hashes and parity are required; do not ignore mismatches.
- Padding implementation/regression: `reports/round1_v4_20261105/tsm_padding/replicate_tsm.py`, `test_static_tsm.py`; pilot repair in `padding_pilot_20261107/`; final folds in `reports/round1_execution_20261110/final_priorities_20261111/`.
- V5: `v5_multiview_20261112/`; distinct from the earlier failed five-run adaptation sweep.
- Evaluation must use the matching model frontend and final epoch. Do not select checkpoints using outer validation.

Example static-input check (inspect checkpoint implementation before selecting `--padding`):

```bash
cd reports/round1_v4_20261105/tsm_padding
python test_static_tsm.py
python check_checkpoint_static.py /authorized/final.pt --padding replicate
```

Some tests import original libraries or frozen artifacts; report missing dependencies rather than counting skipped tests as PASS.

## 6. BTC zero-shot, calibration, C5′, and audits

`btc_zeroshot/` retains preprocessing/inference/reports. `round1_bayes_20261108/` retains multi-ROI models, fitted priors, calibration, and reports; `round1_execution_20261110/` retains C5′, AdaBN, adaptation, PBV/G, and final audits.

Read each directory's `README.md` and `docs/ENTRYPOINTS.json`: this AST-derived index does not import/launch workers. Many files use constants instead of CLIs; root changes require a separate revision. Controllers require existing approval/budget/frozen/source-index receipts; this release **does not fabricate authorization receipts** to enable execution.

Table reproduction order: lock person splits → export dev-only manifests → preprocessing/support → per-checkpoint/condition predictions → exact person/video/window-key joins → train-fold calibration → person-first metrics → paired bootstrap → coverage/failure/static gates. Run reports only with complete dependencies. Each BTC table needs a training-fold constant and saved calibrated comparator on identical videos.

C5′: use `c5prime_v2.py` or the version imported by the current caller; `-bf 0` is a separate condition. Old C5, C5′default, and C5′bf0 are distinct recipes. Retain bitstream hashes, commands, frame mappings/PTS, and measured bitrate, not only target settings. See imports in `common_repair.py` and `ladder_physical_v3.py` for the final revision; initial faulty container-clock output is not the primary PPG result.

## 7. MATLAB / Octave baseline

MAT input requires `roi_data.t` (N×1, increasing seconds), `roi_data.rgb` (N×3), and `roi_data.valid` (N×1). The .m script reads no video; export ROI first.

```bash
octave --traditional --quiet --eval "addpath('reports/round1_execution_20261110'); process_round1('roi_data.mat','output/processed_output');"
```

Create the output directory first and use a new prefix. The script writes CSV and `.mat -v7`. This is an Octave-validated POS baseline, not the final BTC estimator or a MATLAB end-to-end PASS.

## 8. Release scope

`SOURCE_MANIFEST.json` checks **each released source file**, not a complete server inventory. SSH/authentication/monitoring helpers are excluded. Old frozen bundles hashed as entire directories need original asset bundles for exact replay; do not generate a different hash and call it the old one.

Cloning/importing the quick start launches no jobs. Holdout is not opened. The repository contains no tokens. See `VALIDATION.md` for the exact release checks performed.
