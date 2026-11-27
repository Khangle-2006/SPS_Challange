# Preregistration — seven-constraint run and MAE/RMSE reduction attempts (19 November 2026)

Written around 18:40 Vietnam time on 19 November 2026, **before any results from this run existed on the server**.
Author: Claude. The thresholds below are **Claude's proposals** (Section 11 of `claude/bay-constraint-2026-11-19.md`); the team has not adjusted them.
Results known before writing: trace-level simulations from the night of November 18–19 (BTC dev) and Codex's C1/C4 report (`reports/reader_w2b_20261116`).

## 1. Evaluated pipeline (frozen, no retraining)
- Front end: `t8_oval_20261031/oval_frontend.py` (FaceMesh, 36-point oval) → 36×36 crops, rerun **on degraded frames**.
- Public datasets (UBFC2 dev: 35 subjects; MCD: 119 recordings/60 subjects): padding-fixed CNN (out-of-fold checkpoint) + POS + CHROM → standardized log-spectrum fusion
  (weights fitted on the other four folds) → recording-level aggregation (Viterbi; (σ,τ) fitted on the other four folds), exactly as in `claude_research_20261118/code/b04_public_final.py`.
- BTC dev (218 videos/158 subjects): reader_v3 (`a11_reader_v3.py`). The BTC holdout and seven UBFC test subjects are **untouched**.
- Primary estimate: **condition-specific refitting** (readout parameters fitted on four folds of the same condition). Also report **frozen** results (parameters fitted on native data, applied to each condition).
- Statistics: subject-level MAE; 10,000 paired subject bootstrap samples, seed 0. Public datasets: 10 s windows from `inference_repair_20261113/MANIFEST.json`; BTC: one prediction per video.
- "Constant without video": median label from the four training folds (public datasets: median window HR per dataset; BTC: posterior median with β=0).

## 2. Conditions (parameters fixed in advance)
| Code | Degradation (MATLAB semantics from the documentation) |
|---|---|
| native | Unchanged (reproduction check against the old cache) |
| c1_g50p, **c1_g25p**, c1_g12p | uint8(gain·I), gain 0.5 / 0.25 / 0.125; then `imnoise(·,'poisson')` |
| c1_g25gv001, c1_g25gv01 | Gain 0.25, then `imnoise(·,'gaussian',0,v)`, v = 0.001 / 0.01 |
| **c4_vga**, c4_qvga | Antialiased bicubic `imresize`, fit within 640×480 / 320×240, preserve aspect ratio (native UBFC2/MCD are already 640×480 → only c4_qvga) |
| c6_us1 | `imfilter` with an unsharp kernel (alpha 0.2), one pass, full frame |
| **c7_blue**, c7_white, c7_black_high | Solid lower-face polygon from clean-frame landmarks; 'standard' (medical blue, white), 'high' (black) |
| c5_lit | 10 fps, ≤500 kbps, default encoding (existing team data: BTC `c5`, public `c5prime_default`) |
| **c5_tuned** | 10 fps, x264 bf=0, no periodic keyframes, maxrate ≤500 kbps (public: existing `c5prime_bf0`) |
| c5_vga | Resize to ≤640×480, then encode as c5_tuned |
| c5_face | "Device mode": crop the face region (face box expanded by 30%) to ≤192×240, then encode at ≤500 kbps |
| c3_side, c3_big | Composite a second person into the frame (side by side at the same size; distractor 1.3× larger) — UBFC2 |
Bold = **primary** condition for that constraint; other levels form dose–response/stress tests.
C1 noise: 640×480 frames → full frame; HD frames (BTC) → gain on the full frame, noise only within the native-cache face box expanded by 50% on each side (compute limitation; disclose in reporting).

## 3. Pass/fail gates
**Degradation gate (C1, C4, C5, C7)** — evaluated on the primary condition and primary estimate:
 (a) Coverage: ≥90% of windows (public) / videos (BTC) with native output still produce output;
 (b) MAE is no worse than the constant without video: the paired 95% CI of (condition MAE − constant MAE) is **not entirely above 0**;
 (c) Also report the stronger result: improvement over the constant with a CI excluding 0. Always report the paired difference from native.
 C7: the "occlusion-aware" branch (detect occluded rows on a 6×6 grid and exclude them) is primary; report the unchanged branch as a control, and report false occlusion detection rates per frame/video on unoccluded faces.
**Group gate (C2, C6)**: every group with ≥20 subjects has MAE within ±2 bpm of overall MAE, and a subject-level group-label permutation test (10,000 permutations, statistic = maximum − minimum group MAE) does not reject at 5%.
 C6: report the ≥60-year-old group separately; if it has <20 subjects, report results and CIs only, without a pass/fail conclusion.
**C3 gate**: zero valid frames read from the distractor across all composites; coverage ≥90% of single-face windows; the paired CI of (composite MAE − single-face MAE) contains 0.
A constraint is marked "passed" only if it passes on **every dataset where that condition was evaluated**; otherwise report each dataset separately.

## 4. MAE/RMSE improvement trials (predefined list and selection rule)
Candidates: (1) point estimate = posterior mean / median–mean blend (RMSE target); (2) harmonic-comb evidence (add the log-spectrum at 2f);
(3) add classical experts to fusion (GREEN, PBV, LGI, OMIT) using existing traces; (4) sub-grid peak interpolation; (5) quality-based window weighting (spectral SNR);
(6) aggregate recordings with a posterior product instead of Viterbi. Fit all new parameters on the other four folds.
Acceptance rule: accept a change only if the paired subject-level 95% CI of its difference from the current version **excludes 0** for the target metric, with no significant deterioration in the other metric.
Otherwise record "tested, not accepted". Required control for any accepted change: shuffle labels between subjects. Do not open the holdout.
