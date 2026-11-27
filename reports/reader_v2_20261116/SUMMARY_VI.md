# Reader_v2: two activity-specific offsets

Results on **218 dev videos / 158 subjects**: MAE **10.25048 bpm**, bias **+0.18085 bpm**, Spearman **0.49133**. MAE is averaged within each subject, then across subjects; lower is better. Positive bias means predictions exceed labels. Spearman describes agreement in rank between subject-average predictions and labels, not absolute accuracy.

| Control on the same videos | MAE (bpm) | Reader_v2 minus control [CI95] |
|---|---:|---|
| Training-fold constant | 12.07068 | −1.82020 [−3.03373; −0.59080] |
| Saved calibration | 11.37385 | −1.12337 [−2.08414; −0.18328] |
|Reader_v1|10,57015|−0,31967 [−0,85733; +0,22855]|
|Reader_v2|10,25048|—|

CIs use 10,000 paired subject bootstrap samples, seed 0. Reader_v2 beats the first two baselines with CIs excluding 0. Against reader_v1, the CI still includes 0: the 0.320 bpm reduction is not conclusive evidence of improvement. The MAE ≤10 bpm gate remains unmet. Do not change the candidate or freeze a configuration automatically; await the user's decision alongside C1/C4.

## Annotation table and fitting procedure

The available organizers' table has a **PROTOCOL** column in both Normal and Abnormal. Valid activity metadata was confirmed for **72/72 holdout videos**, reading only ID and activity columns. No PR labels, video access, or holdout evaluation occurred. Its hash differs from the original locked file; this check confirms metadata availability only, without replacing dev labels or confirming the hidden-test interface.

For each fold, the reader and both offsets are learned from the four training folds. First, the fitted reader predicts those same training videos; then the weighted mean of `(PR − prediction)` is computed separately for Task and non-Task groups. Each subject has equal total weight within a group. Offsets are added to validation-fold predictions only after fitting. Offsets are not fitted on OOF results; predictions are not clipped and no new variants are searched. These are offsets from in-sample training residuals, as requested, not inner-OOF residuals.

| Validation fold | Beta | Task offset (bpm) | Non-Task offset (bpm) |
|---:|---:|---:|---:|
|0|3|+13,42157|+3,28409|
|1|2|+12,09375|+2,49773|
|2|2|+11,54167|+2,54318|
|3|3|+11,88095|+3,05752|
|4|3|+7,71795|+3,99779|

The code name `offset_rest` means **non-Task**, covering both Resting and Walking. Task1/2/3 share one Task flag. REPORT.md contains Normal-Resting / Abnormal-Resting / Task / Walking tables with full MAE, bias, Spearman, and CIs.

## Controls

- **Subject-level label shuffling, 20 runs:** MAE 12.03634 in all runs; beta was always 0. The subject-block shuffle policy within equal video counts is unchanged from reader_v1. Each run retained 13–25 unchanged subject-fold blocks. This null does not test every permutation scheme; zero dispersion is not strong independent evidence. The code was not adjusted to force a null result near 12.
- **Task-flag shuffling, 20 runs:** mean MAE 10.60673; 5th–95th percentiles [10.42577; 10.72989], full range [10.37530; 10.74075]. The extra benefit over reader_v1 disappears on average, but not every shuffle performs worse than reader_v1. Flags are shuffled separately in training and validation within each fold; the base reader is unchanged.
- **Static images: 20/20 PASS.** With no spectrum, return the prior median plus the fitted flag-specific offset and `abstain=true`. This is an activity-based fallback, not a pulse measurement from a static image. See STATIC.json.
- **Octave: 5/5 real ROIs PASS**, checking both flags per ROI (10 comparisons), with Python differences below 0.5 bpm. Actual MATLAB has not been run. See OCTAVE_PARITY.json.

C1/C4 continues with the unchanged reader_v1 configuration and locked beta gate; this run does not modify workers or add reader_v2 C1/C4 runs. These results do not authorize opening the holdout.

## Rerunning and deployment

`python reports/reader_v2_20261116/run_reader_v2.py` uses native_spectra.npz and PREDICTIONS.json saved under reader_v1_20261116. The code does not access holdout media or labels. `check_schema.py` checks only the two metadata columns within the approved scope.

`process_round1.m` is stored separately under reader_v2_20261116, without overwriting v1. Call `process_round1(roiPath, outputPrefix, modelPath, isTask)` with an explicit activity flag. `reader_model_dev218_v2.json`/MAT are fitted on all dev data for packaging and are not used for the OOF metrics above. Octave checks five real ROIs with both flags per ROI; this does not replace validation in actual MATLAB.
