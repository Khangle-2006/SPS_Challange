# W2b and reader — November 16, 2026 update

**The old W2 run has stopped. The native reader at 10.570 bpm is the BTC candidate selected by the user.** This is a dev result, not a holdout result. The ≤10 bpm success gate remains unmet.

## 1. W2 tests

Completed and partial runs were retained; the old W2 queue and W2 spectral-fusion step were canceled.

T1 only tests whether the model can memorize 16 fixed training clips. It is not evidence of generalization. The effective batch size is 16, split into two microbatches of eight to remain within the VRAM limit.

| Step | Eval-mode weak CE on 16 clips | Training weak CE from the latest step | Readout MAE on 16 clips (bpm) |
|---:|---:|---:|---:|
|0|5,266863|—|19,403198|
|100|3,423835|3,805221|0,876465|
|200|3,260529|3,578791|0,618042|
|400|3,198261|3,365320|0,389648|

A flat spectrum has CE 5.056246; the ideal target distribution has CE 3.157451. Lower CE is better. T1 passes <4.0 with both training and evaluation loss. The W2b fold 2 true/shuffled-label pair is complete; the W2b branch was closed because it failed the gate.

T2, using the same initial weights/BN on one batch:

| Model component | Phase-loss gradient L2 | Weak-CE gradient L2 | Weak/phase ratio |
|---|---:|---:|---:|
| Full model | 3.179056 | 16.737888 | 5.265050 |
|final_dense_2|2,189098|15,949274|7,285775|

The weak-CE gradient is not smaller than the phase-loss gradient on the measured batch. This alone does not identify why W2 failed to learn. W2b updates two separate optimizers as requested, without changing other parameters.

**Results verified at 09:05 ICT on November 16:**

| Fold 2 — 42 validation videos/32 subjects | True labels | Shuffled labels |
|---|---:|---:|
|MAE validation (bpm)|17,432851|17,672521|
|Weak-CE train epoch 1|5,308940|5,485891|
|Weak-CE train epoch 10|5,262399|5,423419|
| MAE on 40 fixed training videos, epoch 1 | 18.594813 | 20.178596 |
| MAE on 40 fixed training videos, epoch 10 | 18.727266 | 18.714513 |
| Static-image check | 10/10 PASS | 10/10 PASS |

True labels outperform shuffled labels by only **0.239670 bpm**, below the 1 bpm threshold. True minus shuffled is −0.239670, CI95 [−2.044661; +1.458253]. True-label weak CE decreases by only **0.046541**, below the 0.3 threshold. Both expansion gates fail. The controller has exited; W2b folds 0/1/3/4 were not launched. There is no W2b result for all 218 videos. See [W2B_REPORT.md](W2B_REPORT.md) for four groups, baselines on the same videos, and per-epoch losses.

## 2. Reader with a beta gate

**Prior and beta are refitted within each condition on the four training folds.** Native beta is not reused for C5. Calibrated rows in the detailed report are historical native predictions on the same videos, not calibration fitted specifically for compression.

Positive beta is retained only if subject-level training MAE improves by at least 0.5 bpm over beta 0. Windowing, bandwidth, beta grid, and prior are unchanged. All 218 native predictions are preserved; fold betas remain 3, 2, 2, 3, 3. Training reductions are 1.473; 1.130; 1.819; 1.719; 1.380 bpm, respectively.

| Condition | Videos/subjects | Gated reader MAE, bpm | Δ constant [CI95], bpm | Beta across five folds |
|---|---:|---:|---|---|
|Native|218/158|10,570148|−1,500527 [−2,792748; −0,191443]|3 / 2 / 2 / 3 / 3|
| Historical C5 | 218/158 | 12.073840 | +0.003165 [−0.280617; +0.290084] | 0 / 0 / 0 / 0 / 0 |
|C5′ default|161/109|12,951070|+0,146789 [−0,162080; +0,449541]|0 / 0 / 0 / 0 / 0|
|C5′ bf0|158/106|13,278302|+0,205975 [−0,169851; +0,578616]|0 / 0 / 0 / 0 / 0|

MAE is averaged within subjects, then across subjects. CIs use 10,000 paired subject bootstrap samples, seed 0. Negative Δ favors the reader. All compression conditions use the prior and no longer rely on spectra. The KDE prior median differs slightly from the raw PR median used by the constant predictor, so the two rows need not match. No new algorithm change is introduced to force equality.

This shows that the current reader does not exploit spectra effectively under the tested compression conditions; it does not prove that all pulse information is lost. Rows with different coverage cannot be directly compared to attribute codec effects. C5′ is an ffmpeg simulation, not verified as equivalent to MATLAB VideoWriter.

MAE, bias, Spearman, and four-group results are in [GATED_REPORT.md](GATED_REPORT.md). Predictions, beta, and training reductions per fold are in GATED_RESULTS.json.

## 3. C1/C4 in progress

There are 218 dev videos and five conditions: C1 gains 0.5 / 0.25 / 0.125; C4 QVGA / VGA. Four CPU shards transform full frames, rerun face detection, preserve source timing, and use the beta gate above. At 09:05 ICT, 24/1,090 video–condition items were complete; all four workers were running with no recorded errors. This is processing progress, not a complete evaluation table.

C1 in simulate.py also includes the x264 CRF18 storage codec and must not be interpreted as an isolated noise effect. C1/C4 results are not yet available. The final C_REVIEWED_REPORT.md table will report coverage, bias, Spearman, and paired differences against the constant/native reader on the same videos.

The 72-video/41-subject holdout remains locked, with zero evaluations. No further reader tuning.
