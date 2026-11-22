# reader_v2 — fixed protocol before scores

Only native 218 development videos/158 people, unchanged five person folds. Schema check is limited to IDs and activity column; no physiological holdout labels/media/evaluation. Current workbook snapshot supplies availability evidence only, never replaces frozen dev labels.

For each outer fold, fit reader_v1 prior/beta (including >=0.5 bpm training-gain gate) using only the other four folds. Predict those training videos with this fitted reader. For Task and non-Task separately, compute weighted mean residual (PR - prediction), weighting each video by inverse count of that person's videos within the respective category. Add its category offset to each validation prediction. No clipping, bandwidth/window/beta change, extra calibration, OOF residual fitting or inner-loop parameter search. This is train-only outer cross-validation; offset residuals are in-sample training residuals as requested, not inner-OOF predictions.

20 training-person label block permutations, seed0, within identical video-count strata, same policy as reader_v1. Validation labels remain original. Refit reader and offsets on each permutation; report fixed blocks and actual results without forcing MAE to12.

20 Task-flag permutations (seed0), separately inside each outer train and validation partition, preserving flag counts; refit offsets with permuted train flags and use permuted validation flags. Keep labels, spectra and reader unchanged. Report residual benefit vs reader_v1 and loss of benefit vs true flags; do not force expected null.

Static: repeat a real ROI frame's RGB for20dev videos; no spectrum must result, return prior median plus the prefit category offset and set abstain=true. The returned number is a conditional fallback, not a physiological measurement. No calibration added for static signals.

Metrics: person-first MAE/bias/Spearman, paired10000bootstraps seed0; four groups; constant, saved calibrated11.37, reader_v1 on identical videos. OOF configuration selection remains exploratory. Holdout not evaluated. Native candidate stays reader_v1 pending user's final choice with C1/C4.

Octave: same frozen reader frontend and numerical operators, add task flag and two stored offsets only; five existing real ROI inputs, Python difference<=0.5 bpm. No real-MATLAB claim. C1/C4 workers remain untouched. New work local CPU and low-priority CPU Octave only, no GPU/model training. Output small artifacts, no raw copies.
