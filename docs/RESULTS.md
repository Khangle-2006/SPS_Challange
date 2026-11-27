# R1 — rerun after open-rppg state isolation repair

**MODE: selfwarm**. The decision is preserved from z18: `ADOPT self-warm`, not reselected from R1 results.

All three models retain their weights. Initial state is copied immediately after Model construction and restored before every video/recording/condition. Selfwarm, when selected, uses only pass 2 of that same video. No CNN training, holdout access, or parameter-grid expansion.

## Checks before the full run

| Model | Reset: 2 videos | Selfwarm: 2 videos | A → B → A (both MODEs) | Self-warm gate, time (s) |
|---|---|---|---|---:|
| FacePhys.rlap | difference 0.0 | difference 0.0 | difference 0.0 | 91.8 |
| ME-chunk.pure | difference 0.0 | difference 0.0 | difference 0.0 | 51.6 |
| ME-flow.pure | difference 0.0 | difference 0.0 | difference 0.0 | 111.8 |

## Interpreting gates and confidence intervals

Preserve script gates: `a` = coverage ≥90%; `b` = lower CI95 bound for ΔMAE versus constant ≤0 (not demonstrated worse); `c` = upper bound <0 (better than constant). `-` means not passed. Gate `b` alone is not an equivalence or margin-based noninferiority test. These gates differ from preregistered BTC native success ≤10 bpm. Deltas/CIs are paired by person, with 10,000 bootstraps, seed 0.

## BTC dev

The old 8.350 result is invalidated; independent reset is 9.130. Old columns are audit records, not valid performance baselines. z18 reports 8.390 as historical because it replaced 17 heavy rows with reset spectra; this is not the original 8.350 run.

| Condition | Videos | New V4 [CI95] | Old V4 | MAE shift | Constant | Old native calibration | V3, same condition | Δ V4−V3 [CI95] | Δ V4−constant [CI95] | Gates |
|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| native | 218 | 8.333 [7.090; 9.659] | 8.350 | -0.017 | 12.071 | 11.374 | 9.587 | -1.254 [-2.282; -0.283] | -3.737 [-5.095; -2.406] | abc |
| split10 | 218 | 9.480 [8.041; 11.019] | 9.815 | -0.335 | 12.071 | 11.374 | 10.053 | -0.572 [-1.601; +0.426] | -2.590 [-4.002; -1.185] | abc |
| track_c1_g25p | 200 | 9.672 [8.173; 11.252] | 9.739 | -0.067 | 12.501 | 11.580 | 10.417 | -0.745 [-1.731; +0.112] | -2.829 [-4.102; -1.615] | abc |
| aidtrack_c1_g25gv01 | 197 | 12.258 [10.547; 14.129] | 12.274 | -0.016 | 12.614 | 11.727 | 12.206 | +0.052 [-0.957; +1.025] | -0.356 [-1.542; +0.821] | ab- |
| c2_vit | 199 | 8.832 [7.335; 10.506] | 9.141 | -0.310 | 12.426 | 11.459 | 9.523 | -0.692 [-1.403; +0.029] | -3.594 [-5.040; -2.172] | abc |
| c4_vga | 200 | 9.101 [7.529; 10.780] | 9.289 | -0.188 | 12.501 | 11.580 | 9.616 | -0.515 [-1.244; +0.248] | -3.401 [-4.859; -1.932] | abc |
| c4_qvga | 200 | 9.443 [7.877; 11.139] | 9.242 | +0.202 | 12.501 | 11.580 | 9.690 | -0.247 [-1.145; +0.653] | -3.058 [-4.559; -1.545] | abc |
| c6_us1 | 197 | 8.444 [6.892; 10.134] | 8.461 | -0.017 | 12.534 | 11.554 | 10.133 | -1.688 [-2.830; -0.745] | -4.090 [-5.641; -2.554] | abc |
| track_c7_blue | 197 | 10.457 [8.785; 12.236] | 9.959 | +0.498 | 12.687 | 11.739 | 10.506 | -0.050 [-1.147; +1.020] | -2.231 [-3.510; -0.967] | abc |

Preserve y11 logic: for each condition, fit native videos from its paired cohort within training folds, then apply condition spectra without fitting condition labels. The calibrated comparator is the historical native configuration reapplied to identical videos, not separately fitted per condition. Heavy contains only 17 existing native videos: reused, not rerun.

## Public — P4m and P4

| Condition | Dataset | Windows | New P4 [CI95] / old | New P4m [CI95] / old | Δ new−old P4m [CI95] | Constant | P4 / P4m gates |
|---|---|---:|---|---|---|---:|---|
| native | UBFC2 | 193 | 0.722 [+0.494; +1.024] / 0.710 | 0.746 [0.516; 1.051] / 0.754 | -0.007 [-0.036; +0.021] | 14.592 | abc / abc |
| native | MCD | 1959 | 1.908 [+1.481; +2.547] / 1.835 | 1.511 [1.247; 1.820] / 1.586 | -0.075 [-0.122; -0.029] | 11.684 | abc / abc |
| c4_qvga | UBFC2 | 193 | 0.728 [+0.501; +1.028] / 0.717 | 0.727 [0.494; 1.038] / 0.743 | -0.016 [-0.041; +0.011] | 14.592 | abc / abc |
| c4_qvga | MCD | 618 | 2.282 [+1.424; +3.676] / 2.017 | 1.520 [1.158; 1.975] / 1.544 | -0.024 [-0.073; +0.028] | 10.962 | abc / abc |
| c6_us1 | UBFC2 | 193 | 0.730 [+0.495; +1.045] / 0.731 | 0.732 [0.507; 1.024] / 0.756 | -0.024 [-0.055; +0.005] | 14.592 | abc / abc |
| c6_us1 | MCD | 618 | 2.146 [+1.425; +3.308] / 1.965 | 1.641 [1.202; 2.198] / 1.597 | +0.043 [-0.039; +0.146] | 10.962 | abc / abc |
| aidtrack_c1_g25gv01 | UBFC2 | 184 | 15.124 [+11.956; +18.575] / 15.093 | 13.450 [10.776; 16.320] / 14.055 | -0.605 [-1.483; +0.124] | 14.613 | ab- / ab- |
| aidtrack_c1_g25gv01 | MCD | 592 | 9.887 [+7.589; +12.508] / 10.141 | 10.435 [8.187; 12.915] / 10.306 | +0.128 [-0.791; +1.019] | 10.983 | ab- / ab- |
| c2_vit | UBFC2 | 193 | 0.745 [+0.513; +1.047] / 0.724 | 0.750 [0.524; 1.045] / 0.748 | +0.002 [-0.025; +0.028] | 14.592 | abc / abc |
| c2_vit | MCD | 618 | 2.183 [+1.490; +3.218] / 2.232 | 1.734 [1.301; 2.245] / 1.741 | -0.007 [-0.086; +0.067] | 10.962 | abc / abc |
| track_c1_g25p | UBFC2 | 193 | 2.172 [+1.305; +3.183] / 2.241 | 1.532 [0.997; 2.264] / 1.623 | -0.091 [-0.229; +0.015] | 14.592 | abc / abc |
| track_c1_g25p | MCD | 618 | 3.046 [+2.203; +4.066] / 3.006 | 2.443 [1.790; 3.182] / 2.526 | -0.083 [-0.157; -0.009] | 10.962 | abc / abc |
| track_c7_blue | UBFC2 | 192 | 1.506 [+0.919; +2.323] / 1.506 | 1.447 [0.839; 2.286] / 1.711 | -0.264 [-0.554; -0.076] | 14.587 | abc / abc |
| track_c7_blue | MCD | 618 | 4.712 [+3.086; +6.900] / 5.068 | 2.972 [2.283; 3.768] / 3.217 | -0.245 [-0.550; +0.028] | 10.962 | abc / abc |
| c1_g50p | UBFC2 | 193 | 1.263 [+0.898; +1.703] / 1.239 | 0.982 [0.741; 1.272] / 1.005 | -0.023 [-0.073; +0.018] | 14.592 | abc / abc |
| c1_g50p | MCD | 618 | 2.917 [+1.898; +4.552] / 3.027 | 1.869 [1.509; 2.274] / 1.886 | -0.016 [-0.073; +0.042] | 10.962 | abc / abc |
| c1_g12p | UBFC2 | 186 | 4.661 [+3.002; +6.686] / 4.797 | 2.564 [1.723; 3.744] / 2.997 | -0.433 [-0.916; -0.058] | 14.844 | abc / abc |
| c1_g12p | MCD | 616 | 4.130 [+2.889; +5.674] / 4.142 | 3.056 [2.335; 3.868] / 2.945 | +0.111 [-0.067; +0.309] | 10.969 | abc / abc |
| c1_g25gv001 | UBFC2 | 192 | 4.126 [+2.715; +5.855] / 4.162 | 3.082 [2.012; 4.348] / 3.129 | -0.047 [-0.552; +0.552] | 14.583 | abc / abc |
| c1_g25gv001 | MCD | 616 | 4.925 [+3.563; +6.586] / 4.866 | 4.139 [2.899; 5.768] / 4.341 | -0.202 [-0.464; +0.017] | 10.969 | abc / abc |
| c7_white | UBFC2 | 176 | 1.757 [+0.880; +3.155] / 1.808 | 1.154 [0.787; 1.579] / 1.211 | -0.057 [-0.147; +0.033] | 14.854 | abc / abc |
| c7_white | MCD | 614 | 4.474 [+3.132; +6.160] / 4.399 | 3.602 [2.517; 4.972] / 3.711 | -0.109 [-0.986; +0.605] | 10.943 | abc / abc |
| c5_split10 | UBFC2 | 179 | 0.659 [+0.515; +0.849] / 0.637 | 0.646 [0.496; 0.845] / 0.655 | -0.009 [-0.038; +0.018] | 14.656 | abc / abc |
| c5_split10 | MCD | 1929 | 1.761 [+1.486; +2.083] / 1.765 | 1.704 [1.435; 2.022] / 1.768 | -0.064 [-0.112; -0.023] | 11.690 | abc / abc |

All old/new comparisons use the same window intersection and person pairing; 10,000 bootstraps, seed 0. Gate a: coverage ≥0.9; b: CI does not establish worse-than-constant performance; c: CI establishes better-than-constant performance. Do not claim improvement when the CI includes 0.

## P4m parameters

| Fold | New weights (CNN, POS, CHROM, experts) | Old weights |
|---|---|---|
| fold0 | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| fold1 | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| fold2 | (1.0, 0.0, 0.25, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| fold3 | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| fold4 | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |
| deploy | (1.0, 0.5, 0.0, 1.0) | (1.0, 0.0, 0.25, 1.0) |

Complete readout parameters (gamma, sigma, tau, estimator, ref, training constant):

```text
P4m public reader, frozen 2026-11-23 21:49:16 (team decision 22 Nov 2026: P4m, documented deviation from the z13 rule)
  channels: CNN (team padding-fixed EfficientPhys+TSM, out-of-fold checkpoint) on crops pre-filtered by a centred moving average of k = ceil(fps/10)
            frames weighted by the valid flags (x13 nat_ma3, x20 under conditions; C5 split keeps its binned 10 fps stream); POS and CHROM from the
            UNFILTERED 6x6 block means; experts = mean of the standardised log spectra of FacePhys.rlap, ME-chunk.pure, ME-flow.pure (y12/y13).
  fusion  : E = w . Z (Z = per-window standardised log spectra on the 42-180 bpm grid), w fitted on the training folds over {0,.5,1,2}^4 (normalised).
  readout : forward-backward with quality weight (y07 marg/point, y02 qscale), (gamma, sigma, tau, estimator) per data set per fold.
  pooled  : training windows = native + c4_qvga, c6_us1, aidtrack_c1_g25gv01, c2_vit, track_c1_g25p, track_c7_blue on the common recordings.
  fold 0 : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 4.995398, 96.680; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 4.383784, 83.496
  fold 1 : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 5.016544, 93.604; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 0.8, median, 4.199450, 84.375
  fold 2 : w (CNN, POS, CHROM, experts) = (1.0, 0.0, 0.25, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 4.475250, 95.361; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 3.811568, 86.572
  fold 3 : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 5.021681, 97.119; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 4.214000, 87.012
  fold 4 : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 5.044070, 97.119; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 4.219725, 82.617
  deploy : w (CNN, POS, CHROM, experts) = (1.0, 0.5, 0.0, 1.0); UBFC2 (gamma, sigma, tau, est, ref, const) = 2.0, 15.0, 1.6, median, 5.016605, 96.680; MCD (gamma, sigma, tau, est, ref, const) = 4.0, 6.0, 1.6, median, 4.258152, 84.814
  check   : max |MAE - z13 P4m| over conditions x data sets = 0.00e+00
  sha256  : a75d4deb6199f7016a5a8736c27a84ebcfccd4352e6a51345a2be9eb68b4c787  out/z14_p4m_frozen.npz
```

## Limitations and timing

**The three-of-17 model selection used old-style spectra; selection/nested selection was not repeated in R1.** All results remain dev outcomes after many research decisions, not independent holdout validation. Do not report OOF using deployment parameters.

- Reset reference gate: 91.48 seconds.
- Self-warm + order-independence gates: 129.75 seconds.
- inference: 56687.7 seconds.
- score: 265.3 seconds.
- freeze_public: 104.3 seconds.

CLI packaging checks are documented separately in [VIDEO_CLI_VALIDATION.md](VIDEO_CLI_VALIDATION.md). The R1 PR did not merge automatically.

## Condition-code definitions

Read `constraint_ops.py`, `x01_extract.py`, and `x09_followup.py` directly; do not infer operations from condition names:

| Code | Transformation and frontend |
|---|---|
| `native` | Original video without simulated degradation. |
| `c1_g50p`, `c1_g12p` | RGB uint8 ×0.50 / ×0.12, rounded, then Poisson noise with pixel-value mean. No corresponding lux measurement. |
| `track_c1_g25p` | Gain 0.25 + Poisson; if FaceMesh loses the face, retain the oval using an upper-face template, NCC≥0.6. |
| `c1_g25gv001` | Gain 0.25 + Gaussian variance 0.001 on [0,1], clipped to [0,1]; not Poisson. |
| `aidtrack_c1_g25gv01` | Gain 0.25 + Gaussian variance 0.01. Detector sees a downsampled/blurred/brightened copy; signal crops still use degraded images; template fallback. |
| `c2_vit` | Eight landmark-aligned simulated depigmentation patches, each a 25-point hull; RGB multiplied by (1.25; 1.5; 1.6), soft edges and saturation. Not a substitute for actual pathology or Fitzpatrick I–VI stratification. |
| `c4_vga`, `c4_qvga` | Antialiased bicubic resizing, preserving aspect ratio within 640×480 / 320×240; rerun frontend. |
| `c6_us1` | One unsharp kernel α=0.2, replicate border, **full frame** in x01/x09. Simulates extra detail; not evidence of robustness for people aged 60–70+. |
| `track_c7_blue` | Blue surgical-mask polygon, RGB (167,199,231), covering the lower face, standard fit; template tracking. **Not blue lighting.** |
| `c7_white` | Same lower-face polygon, white (235,235,235), standard frontend. |
| `split10`, `c5_split10` | Split-crop transmission described below; not full-video compression. |

Cohorts differ; interpret only paired deltas on the intersection of videos/people/windows. Python simulators implement the described operations but have no established bitwise MATLAB equivalence.

## C5 — transport evidence

The frontend runs on-device and retains every `k=ceil(fps/10−0.001)`-th frame as in the frozen code. Packets include a 9-byte timestamp/flag and a 36×36 lossless WebP box crop; BTC adds 6×6 RGB block means as float32 (432 bytes), public adds a 36×36 lossless WebP oval. Expert 72×72 crops are reduced with INTER_AREA to 36×36 before transmission, matching expert inputs. This is **on-device ROI transmission**, not full-frame H.264/VideoWriter compression.

Log `W/logs/z08_split_lossless.json`: BTC 218 recordings, mean **202.517 kbps**, reported peak second **252.584 kbps**; public 152 recordings with ovals, mean **311.115 kbps**, maximum **469.312 kbps**. All logged decode checks are bit-exact. See [aggregates and source SHA](C5_TRANSPORT_EVIDENCE.json).

**Evidence limits:** the code omits the final second from peak calculations; `fps_sent` is the rounded recording average, reaching **10.002 fps** on BTC. Existing logs therefore do not establish “≤10 fps and ≤500 kbps in every second.” Frame-skip and parameters are not changed to repair results. Corresponding MAE rows are `split10` (BTC) / `c5_split10` (public), not full-video compression rows.

## C3 — identity and reproduction gates

Source: `SI/RESULTS_22NOV_VI.md`, `SI/guard_v2/` (22 November, **before the R1 state repair**). Two UBFC2 layouts include 35 people and 193 common windows; each has 67,764 valid frames and **0 frames from bystanders**, based on canvas position in protocol x07.

| C3 layout | Single-face MAE | Composite MAE | Δ [CI95] bpm | Mean gate \|Δ\|≤0.05 |
|---|---:|---:|---|---|
| Side by side | 0.75502232 | 0.77511161 | +0.02008929 [−0.01046317; +0.05231585] | PASS |
| Larger bystander | 0.75502232 | 0.77113560 | +0.01611328 [−0.01715960; +0.04938616] | PASS |

CI uses 10,000 person bootstraps, seed 0. The gate checks mean differences, not whether the entire CI lies within ±0.05. These are not rerun P4m R1 results.

Native **reader_v3** also passed: 218 videos/158 people, 9.58705844 → 9.55601786 (Δ −0.03104058). All 185,446 native-branch frames match pixel-for-pixel; 18 videos use another branch, and 6 predictions change. However, **exact reader_v4 reproduction** stopped at the first 1.00 bpm mismatch; the new P4m route differs by >0.01 on 25/193 windows, maximum 1.318359375. Thus `guarded` has not replaced the R1-results frontend. C3 PASS does not establish full R1 reproduction through the guard.

[History](history/R1_STATE_BUG_20261123.md).
