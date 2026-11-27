# FaceVitals Round 1 — proposed technical schema

Official source: [FaceVitals 2027, pages 10–11](https://signalprocessingsociety.org/sites/default/files/2026-10/2027-SP-Cup-Official-Document-FaceVitals.pdf). Page 10 describes the reproducibility package; page 11 lists six MAT variables and an example folder tree. Dtypes, axis order, nested structures, and MAT version are not specified; the choices below are the team's implementation.

| Field | Team implementation layout |
|---|---|
| facial_video | uint8 `[H,W,3,T]`, RGB, all frames accepted by the adapter; embedded directly in v7.3 by default |
| roi_data | Struct: forehead/cheek_left/cheek_right/periocular `[T,K,2]`, pixel x,y; NaN padding, polygon_lengths, valid, and timestamps_s. Geometry does not guarantee skin pixels, especially with bangs. Placeholders are explicitly marked bbox_placeholder |
| reference_signal | Struct: ppg, timestamps_s, valid, source, effective fs_hz; preserves PPG values and adapter timing |
| processed_signal | struct: rppg, timestamps_s, fs_hz=30, hr_bpm_per_window, window_start_s, abstain, window_index_per_sample |
| metrics | MAE/RMSE/Pearson, bias, LoA, coverage, n_windows/n_evaluated/n_subjects; string NA when not estimable |
| params | resolution `[W,H]`, fps_nominal/effective, lux/distance/Fitzpatrick/age/ITA as NA when unavailable, constraints_applied, dataset/subject, codec/bitrate, pipeline/git/checkpoint hashes, and video timestamps |

HDF5 stores reversed axes `[T,3,W,H]` and MATLAB_class uint8 so MATLAB/mat73 displays `[H,W,3,T]`. Structs are written by hdf5storage with MATLAB compatibility. Read the full file using mat73 and verify dtypes/shapes and per-frame RGB hashes. Check Octave only when available; mat73 does not establish Octave compatibility.

The predicted waveform joins 10 s segments, each integrating differences from zero; `window_index_per_sample` identifies segments, so connecting lines must not be interpreted as a continuous physiological signal. Rejected windows remain in CSV rows and the coverage denominator. Reference HR is computed from PPG with a fixed decoder and is not used to select predicted peaks. Model input remains Haar36; T3 polygons are stored alongside it. Missing metadata must not be guessed.

`target_video.mp4` is a lossless RGB packaging copy, checked bit-exact against the source and MAT data. It uses nominal CFR; true sensor timing is stored in `video_timestamps.csv` and params.video_timestamps_s. Bitrate in params is the actual packaged-file bitrate and **does not** represent the C5 operating point; the T5 manifest preserves constraint settings for the input video. Native MCD is already compressed; raw RGB in MAT contains decoded pixels and must not be called uncompressed sensor data.

## Repeated measurements and limits of agreement

For errors `d_ij=HR_est-HR_ref`, use the model `d_ij=mu+b_i+e_ij`. Let k be the number of subjects, N the number of valid windows, and n_i the windows per subject. Compute `MSW=sum_ij(d_ij-mean_i)^2/(N-k)`, `MSB=sum_i n_i(mean_i-mean_all)^2/(k-1)`, and `n0=(N-sum_i n_i^2/N)/(k-1)`. Between-subject variance is `max(0,(MSB-MSW)/n0)`; within-subject variance is `MSW`. Bias is mean_all; LoA are bias ±1.96√(variance_between+MSW). Flag any clipping of negative variance separately; use NA if k<2 or N≤k. These formulas follow the changing-quantity, simultaneous-pair case in [Bland–Altman 2007, pages 575–576](https://www-users.york.ac.uk/~mb55/meas/bland2007.pdf). Within-subject residual independence and equal variances have not been automatically verified; LoA are not confidence intervals for bias.

## Standalone execution

Create a fresh Python 3.12 venv, install requirements with the CPU PyTorch extra index, and preserve the `submission/` tree and accompanying `ops/deep_hr/` modules. Place the checkpoint with the locked SHA at `submission/checkpoints/baseline_seed2027.pt` or pass `--checkpoint`. CLI:

```
python submission/process_round1.py --video INPUT --reference_csv REF.csv --out_dir NEW_OUTPUT --reference-clock-shared
```

Generic CSV input requires `timestamps_s,ppg` on the video clock; optionally add `--video-timestamps` with an NPY vector. The shared-clock flag is a caller assertion, not automatic synchronization. For UBFC2/MCD, use `--dataset`, `--root`, and `--recording`, plus `--audit-dir` for MCD receipts. `--roi-cache` accepts only the matching T3 recording/timestamps. `--constraints C4,C5 --constraint-manifest FILE` records inputs already simulated by T5; it does not transform them again. Do not pass constraint flags for native videos.

## Questions for the organizers

1. Are v7.3, this layout, and these nested structs accepted? Is there a sample evaluator/schema?
2. Can an external video path plus hash replace raw RGB? This is currently opt-in and not confirmed as valid.
3. Page 10 calls the reference CSV ground-truth HR; page 11 uses the broader reference_signal term. Are both PPG waveforms and window HR required, and is there a specific window definition?
4. The example filename `Round1_Baseline_None.mat` differs from the directory name `Round1_Baseline/`; how should constraint prefixes be named? The page 11 tree also includes Round4 although three rounds are described.
5. Page 10 names process_round1.m but describes MATLAB/Python support; is process_round1.py with its libraries and checkpoint accepted?
6. Can unknown metadata be NA? Should coordinate order, masks, padding, and left/right cheek conventions follow the subject or the image?
7. Are decoded RGB and a lossless packaged MP4 accepted with separate timestamps, especially for constraint videos with VFR/dropped frames?
8. What repeated-measure LoA, coverage/abstention, and denominator definitions do the organizers require? One video/one subject is insufficient to estimate between-subject components.

## S1 public P4m representation (24 November)

The six top-level variables are unchanged. S1 uses `roi_type=oval36_actual_frontend`, `oval_points [T,36,2]`, boxes, valid flags and acquisition times, because P4m uses the whole oval. No distinct forehead/cheek/periocular masks are claimed. `processed_signal.rppg` is the CNN cumsum waveform per accepted window with an explicit source label; HR is the P4m spectral-fusion result. Separate NPZ files preserve all source waveforms privately. `params.pipeline=P4m R1`, matching fold/checkpoint/reader/code SHA, and source/target SHA replace the historical Haar36 identity. No change is made to the six-variable contract or MATLAB axis order.
