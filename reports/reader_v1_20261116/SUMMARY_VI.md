# Reader v1 — November 16 results

All 218 predictions reproduced exactly. The gates of MAE≤10 and ≥1 bpm improvement over calibration with a CI excluding 0 were not met. The holdout remains unopened.

| Condition | Videos/subjects | MAE | Bias | Spearman | Δ calibration [CI95%] |
|---|---:|---:|---:|---:|---|
|native|218/158|10.570|-3.796|0.491|-0.804 [-1.829;+0.223]|
|c5|218/158|12.140|-0.908|-0.114|+0.766 [-0.023;+1.520]|
|c5prime_default|161/109|13.041|-0.372|-0.258|+1.621 [+0.618;+2.595]|
|c5prime_bf0|158/106|13.777|-2.487|-0.135|+2.203 [+1.078;+3.318]|

Differences use the saved native baseline on the same videos, not a calibration baseline recomputed from compressed videos. C5′ default and bf0 have different coverage, so comparisons between them do not establish causation. C5′ is an ffmpeg proxy, not verified against MATLAB.

Static images: 20/20 PASS, returning prior + abstention. Twenty label shuffles: mean MAE 11.873, range 11.684–12.076.
Octave passed 5/5 real ROI videos with a 0 bpm difference; actual MATLAB has not been run. W2 spectral fusion awaits five valid folds.
