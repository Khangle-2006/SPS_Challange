# reader_v1

No CNN. Input: precomputed `meta.npz` containing t (seconds), valid, rgb_mean (RGB, not BGR). The reference code and supplied model remain unchanged. OOF evaluation fits each fold using the other four person folds; the all-dev model must not be used to report OOF performance.

Single-video inference, from this directory:

```sh
python reader_v1.py --meta /path/to/meta.npz --model reader_model_dev218.json --out prediction.json
```

Output includes prediction and abstain. No valid spectrum means prior median with abstain=true; it is not a measured pulse. The saved all-dev model is for a future chosen-configuration evaluation; holdout has not been opened.

Reproduction:

```sh
python btc_posterior_reader.py --data ../w2_btc_20261116/export_extracted --out oof_check.csv
```

MAE10.5701476793; fold beta[3,2,2,3,3]; all218 predictions exactly match the supplied CSV. See REPRODUCTION.json, REPORT.md/METRICS.json (four conditions and groups), SHUFFLE20.json and STATIC.json.

## Octave / MATLAB-shaped entry point

```matlab
process_round1('/path/to/roi_data.mat','/path/to/output','/path/to/reader_model.mat');
```

`roi_data` has fields t, valid, rgb (N×3). `reader_model.mat` is exported from the supplied JSON, with fixed filter coefficients. No toolbox or Python call inside process_round1.m. Five real public ROI videos pass Octave --traditional with zero bpm difference; real MATLAB remains unverified. Historical POS packaging baseline is preserved in its original directory.

Initial port failure was caused by integer beta in MAT arithmetic, not a spectral operator discrepancy; beta is now explicitly double. Earlier failure outputs are preserved remotely in octave_attempt1. No scientific thresholds were altered.

## Pending W2 fusion

fuse_w2.py waits for five valid final W2 folds, then performs CPU inference at rate1 with5s hop, exports normalized window-averaged spectra, and compares two vs three spectra with paired person bootstrap. It does not modify training. Read FUSION_RESULTS/FUSION_REPORT only after FUSION_COMPLETE; if any fold is invalid, FUSION_BLOCKED records the reason. Meta cross-fitting over learned OOF features is exploratory, not a fully nested evaluation of the whole model-selection procedure.

No holdout, D3 or closed UBFC test used. C5-prime remains an ffmpeg proxy, not verified MATLAB VideoWriter parity. Compressed-input deltas compare the saved native baseline on identical videos; coverage differs by condition.
