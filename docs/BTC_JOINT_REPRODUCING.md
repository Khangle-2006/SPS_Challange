# Fit and replay the frozen BTC joint reader

`predict.py` uses the shipped models and does not fit anything. This guide is
for maintainers with the authorized **218-video / 158-person dev bundle** who
want to reproduce the existing fit or its OOF predictions. It does not open
holdout, train the CNN/experts, choose new hyperparameters, or overwrite the
models used for inference.

Use Linux/Python 3.12 and the core environment for inventory/replay. For fitting,
create the separate numerical environment with `tools/setup_env.py joint-fit`:
NumPy 2.5.4 and SciPy 1.18.1. The older inference core pins must not be used to
refit: the discrete beta/weighted-median decisions can change model parameters
across numerical-library versions. The fit CLI checks its versions up front.
No frontend, GPU, raw video or expert inference is needed for these commands.
The numerical recipe is published in `training/btc_joint/`; its original
source files and hashes are recorded in [BTC_JOINT_TRAINING_SOURCES.json](BTC_JOINT_TRAINING_SOURCES.json).

## 1. Obtain the private inputs

Set `--reports` to an authorized copy of the project's reports directory.
These paths are relative to that directory:

| Input | Fit | Replay | Contents |
|---|---|---|---|
| `w2_btc_20261116/export_extracted/VIDEOS.json` | Yes | Yes | Ordered 218 dev records: recording, person, fold, pr_bpm, task |
| `w2_btc_20261116/export_extracted/blocks/<recording>.npz` | Yes | Yes | Each record's t, valid, rgb_mean, boxes |
| `w2_btc_20261116/delivery/claude_server_20261118/dev_video_meta.csv` | Yes | Yes | recording, bit_rate, width, height, avg_fps, codec from the original capture |
| `matlab_native_20261124/fixtures/btc_native.npz` | Yes | Yes | C, P, three expert spectra; person, fold, truth, low, task |
| `post_s1_improvement/dev_tta/DEV_SPECTRAL_VIEWS_PRIVATE.npz` | Yes | No | shift10: the frozen CHROM/POS spectra on the shifted window grid |
| `post_s1_gradient_20261124/headroom/DEV_PRIVATE.npz` | No | Yes | nested_router reference predictions, used only for parity assertions |
| `post_s1_round2/fallback/DEV_OOF.npz` | No | Yes | signal reference predictions, used only for parity assertions |

The spectral arrays are `(218, 277)`, ordered exactly as `VIDEOS.json`; the
three expert keys are `FacePhys.rlap`, `ME-chunk.pure`, `ME-flow.pure`.
`shift10` is `(218, 2, 277)`, channel order CHROM, POS. Native and shifted
spectra use the same ordered cohort. Keep the supplied arrays and their hashes;
do not replace them with predictions or different preprocessing. The fit loader
checks the native cohort against `VIDEOS.json`, activity/capture class against
metadata, dimensions, finite spectra, and person-disjoint folds.

The trace arrays are `t: (T,)`, `valid: (T,)`, `rgb_mean: (T,3)`, and
`boxes: (T,4)`. Time is finite and strictly increasing. Raw video, contact PPG,
and demographic tables are not inputs to this cached-reader fit. The supplied
bundle still contains individual data and must remain private.

## 2. Create and check an explicit inventory

```bash
.venv/core/bin/python tools/btc_joint_assets.py create \
  --reports /path/to/authorized/reports \
  --manifest output/joint_assets/assets.local.json --purpose both

.venv/core/bin/python tools/btc_joint_assets.py check \
  --reports /path/to/authorized/reports \
  --manifest output/joint_assets/assets.local.json --purpose both
```

The creator reads the declared video list and the exact paths above, without
scanning a dataset root. It refuses missing assets, duplicate recording keys,
cross-fold people and existing output manifests. `--purpose fit` or `replay`
requires only that column's inputs. `both` lists 224 files, including 218 traces.

The inventory has `schema_version: 1`, `kind: "btc-joint-dev"`,
`scope: "dev-only"`, `videos: 218`, `people: 158`, `purposes`, and `files`.
Every file entry has a reports-relative `path`, byte count and SHA256. Paths
must stay within the supplied reports root. Subsequent commands verify every
hash before using the inputs. The inventory records the bundle supplied by its
maintainer; generating hashes does not independently prove dataset provenance.
Keep it under ignored `output/` or outside the repository.

## 3. Replay the released OOF predictions

```bash
.venv/core/bin/python tools/reproduce_btc_joint.py \
  --reports /path/to/authorized/reports \
  --assets-manifest output/joint_assets/assets.local.json
```

This applies the shipped fold models, recomputes classical spectra from each
trace, and checks predictions against the two reference arrays. It does not
fit models. Expected result: 218 videos, 158 people, person-first MAE
8.020099854 bpm and maximum reference difference at most 1e-9 bpm.

## 4. Fit and export, then compare with the released models

```bash
python3.12 tools/setup_env.py joint-fit
.venv/joint-fit/bin/python tools/fit_btc_joint.py \
  --reports /path/to/authorized/reports \
  --assets-manifest output/joint_assets/assets.local.json \
  --output output/joint_refit --verify-shipped
```

This preserves the original native spectra for fitting calibrated readers and
shifted spectra for their predictions. Router and fallback training use inner
person folds within each outer training fold. The fallback retains the original
Huber loss, penalty, feature scaling, and ±8 bpm bound. Deployment repeats the
same recipe using all dev people; its predictions are never scored as OOF.
No data or parameters are selected using the outer validation labels.

The new output directory contains `models/router_{0..4,deploy}.json`,
`models/fallback_{0..4,deploy}.json`, `models/manifest.json`, and
`FIT_COMPLETE.json`. Existing output directories and the repository's `models/`
are refused. Fold exports retain the signal fallback; the deploy export also
retains the historical capture-only fallback, matching the released structure.

With `--verify-shipped`, the command requires matching model structure and
parameters (`atol=rtol=1e-9`), then compares inference from the exported and
shipped models on all 218 dev traces, for both the corresponding fold and deploy
models: 436 comparisons, unchanged branch choices, maximum PR delta ≤1e-9 bpm.
Only after those checks pass are models and the completion receipt written.
The receipt records input-manifest/code/export hashes and aggregate checks;
it does not publish individual predictions or labels. JSON serialization can
differ in whitespace or insignificant floating-point digits; inference parity
and model checks do not require identical serialized file hashes.

## Relation to the other entrypoints

- P4m raw-video inference: [SUBMISSION_ROUND1.md](SUBMISSION_ROUND1.md).
  After bootstrap, install the additional `torch-cpu` environment.
- P4m/R1 cache replay: [REPRODUCE_SERVER.md](REPRODUCE_SERVER.md); it uses a
  different private dev cache with its own frozen manifest.
- Archived experiments: [GETTING_STARTED.md](GETTING_STARTED.md).
  `tools/repro.py prepare` now includes `runtime/` in the isolated workspace.
  The R1 helper locates runtime relative to that workspace/checkout instead of
  `git_push_20261122/newcomer_20261123`. Data, clock and gate receipts are still
  required; this import fix does not turn every archived experiment into a
  self-contained video runner.
