# Round 1 / S1 — raw-video P4m and private package

S1 does not train, refit, change a grid or read holdout. The user confirmed on 24 November that **public stride remains 10 seconds**, matching R1 (BTC uses 5 seconds).

## Raw-video public CLI

```bash
python3.12 tools/bootstrap_video.py
python3.12 tools/setup_env.py torch-cpu
python3.12 predict_public.py --video /path/video.mp4 --dataset UBFC2 --output output/public_new
```

For a DEV recording use `--fold k` for its held-out person fold. This selects both the matching exported CNN and `w_foldk` / `ro_foldk_DATASET`. New data defaults to `w_deploy` / `ro_deploy_DATASET` and the **existing fold-0 seed-2027 final-epoch-30 checkpoint**, a fixed first-checkpoint convention, not a best-validation selection. It is trained on that fold's training people, not all public dev; no new all-data CNN is claimed. It must not be used to claim OOF performance for people in its training folds.

Only CPU is used. Each child runs nice 19; available RAM must be at least 15 GiB and per-child RSS at most 8 GiB. The source, reader and checkpoint hashes are recorded. Expert state resets per video, followed by two passes on that same video, using pass 2. The runner exports each source's pulse waveforms and window HR/quality/abstention. P4m fuses spectral evidence; it does not supply a physiologically defined single fused waveform.

### Acquisition clock for public benchmarks

Without `--clock`, the runner uses display PTS for new data. A public benchmark may have a separate acquisition clock; container time cannot substitute for it. A label-free NPZ passed with `--clock` has exactly:

- `t`: source-frame acquisition timestamps, including duplicates;
- `source_valid`: original corruption/support mask on unique timestamps;
- `window_start`, `window_eligible`: the preregistered benchmark windows/support eligibility;
- `source_sha256`: binding to the exact input video.

It has no PPG, HR, person labels, reference predictions or cached ROI boxes. Frames are decoded again and FaceMesh/template ROIs are recomputed. The eligibility mask is for retrospective benchmark alignment; it is not available/required for inference on a new unlabelled video. Numeric implementation preserves R1 float16 crop storage boundaries, timestamp merging, filtered CNN input, unfiltered classical means and the fixed public periodogram grid. The inherited moving-average implementation uses `ceil(fps/10−0.001)`.

## S1 A1 result

| Preregistered manifest selection | Held-out fold | Reference/matched windows | Maximum HR difference |
|---|---:|---:|---:|
| First UBFC2 DEV recording, valid fraction ≥90% | 0 | 4 / 4 | 0.000 bpm |
| Next eligible UBFC2 DEV recording | 1 | 5 / 5 | 0.000 bpm |
| First MCD DEV recording | 1 | 17 / 17 | 0.000 bpm |

All three passed the ≤0.01 bpm gate with no missing or extra accepted windows. Selection was written to private `OUT/CHOICE.md` before scoring. This is a three-recording parity check, not a rerun of the full public cohort. No parameters were fitted or adjusted using these results.

## Private baseline package

The baseline is the first qualifying UBFC2 DEV recording in manifest order (not chosen by error). The official file/folder requirements are in [the competition document, pages 10–11](https://signalprocessingsociety.org/sites/default/files/2026-10/2027-SP-Cup-Official-Document-FaceVitals.pdf).

The builder requires `A1_COMPLETE.json` with all three PASS, the preregistered DEV selection and its validated raw-video outputs:

```bash
# In a packaging environment with numpy/scipy/OpenCV + h5py/hdf5storage/mat73:
python submission/build_public_baseline.py --work /path/submission_round1_20261124 --git-commit TESTED_COMMIT
```

It creates `Round1_Baseline/`: lossless raw RGB MP4, acquisition timestamps, PPG CSV and separate HR-reference windows, estimated HR CSV, six-variable MAT, Python and real MATLAB/Octave wrappers. `mat_writer.py` performs full mat73 readback and frame-hash verification. The builder refuses an A1 failure. ZIP sealing follows wrapper validation, not merely successful serialization.

`reference_signal.csv` deliberately stores raw contact PPG with timestamps as specified in S1; `reference_hr_windows.csv` provides the HR counterpart. The official text calls the corresponding CSV an HR reference, so both definitions are explicit. ROI coordinates describe the actual whole oval; no separate forehead/cheek region outputs are fabricated. Unknown lux/distance/age/skin type are NA. Bland–Altman population limits for a single subject are NA.

**The private MAT, media, labels, waveforms and ZIP are never committed or attached to the public PR.** The baseline demonstrates format; the team's BTC reader_v4 and constraint results belong in the technical report. A wrapper passing in Octave does not constitute real MATLAB validation.

## Holdout preparation — CLOSED

`submission/run_holdout.py` predicts only, in fresh processes with DEPLOY parameters and no fold. It seals every result plus code/model/manifest hashes before `submission/score_holdout.py` may open labels. Output directories and score locks are exclusive. A 3-DEV-video dry-run is independent of holdout evaluation.

Real holdout requires the exact user authorization message enforced by `submission/run_holdout.py`, its recorded authorization file, and the previously locked ID/activity file with SHA256 `a0f58cc5b87888aa591ec07de6957d197a954909f85a7db6dfd95d78a664cd76`. No authorization file is supplied by this repository. Dates alone never authorize a run. Prediction manifests must contain metadata only; the separate score file contains only the matching person/PR pairs.

After authorization, person-first MAE/RMSE, person-mean Pearson and Bland–Altman, paired cohort counts/coverage and 10,000 seed-0 person bootstrap are reported overall and by original/recompressed file type. A missing/abstained prediction remains in coverage. No retry, refit or selection after outcomes.

Historical Haar36 packaging is preserved in [submission/historical](../submission/historical/); top-level compatibility modules remain for existing tests. It is not the P4m estimator. Current private artifact completion, wrapper checks and hashes are recorded in the S1 delivery receipt, not assumed from this documentation.

## Completed validation — 24 November 2026

The private baseline contains all **1,547 raw RGB frames**, with MAT shape **480 × 640 × 3 × 1,547**, `uint8`. Full `mat73` readback verified all six required variables. Octave independently loaded the video array and reproduced its decoded-frame SHA256; the source video, lossless MP4 and MAT have identical decoded frames. The real `.m` wrapper ran the Python pipeline from the bundled MP4: all five window decisions (four accepted, one abstained) matched, with **0 bpm maximum difference** on accepted windows. Real MATLAB has not been tested.

The prediction/seal/scoring workflow completed on **three DEV videos from two people** with no holdout access. This checks execution and sealing only: DEPLOY models were fitted on dev, so these dry-run scores are not OOF or generalization estimates. The score file is private. All 15 submission/scoring tests, all 18 maintained tools tests, and all 1,161 source hashes passed. See [the aggregate validation receipt](validation/s1_20261124.json).

Two reporting limitations remain explicit: C5 logs do not certify every second or an exact ≤10 fps bound (maximum rounded BTC average is 10.002 fps); the available C3 identity-guard experiment predates the R1 model-state repair. Neither is relabelled as a stronger R1 compliance result. The private ZIP is not a public repository asset and has not been submitted. Holdout is still CLOSED.
