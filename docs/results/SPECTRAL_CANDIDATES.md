> Archived aggregate report. Historical pending/status sentences describe its creation date, not current status. See the root README for the 15 November disposition. Private artifacts named here are not bundled.

# Three-spectral-peak test on BTC dev

## Question and interpretation

Could a peak near true PR remain as the second or third peak, missed by strongest-peak selection? Retain up to three peaks per video, chosen from signals alone and separated by at least 6 bpm. Compare against PR only after freezing this list.

**Oracle** uses reference PR to select the closest candidate. This tests the candidate set, not an inference method without labels. MAE is mean absolute error in bpm, averaged within people first so those with more videos do not dominate.

**Label shuffling** exchanges entire PR blocks among people with equal video counts, repeated 100 times. Brackets show the 5th–95th percentiles of those results, not a model's 95% confidence interval. True and shuffled labels use identical peak lists.

## 1. Oracle error

|Group|Videos / people|Videos with candidates|True oracle (bpm)|Shuffled oracle: mean [P5–P95] (bpm)|
|---|---:|---:|---:|---|
|All|218 / 158|218|7.55|8.23 [7.74; 8.83]|
|Normal — resting|62 / 62|62|6.83|7.52 [6.52; 8.79]|
|Abnormal — resting|76 / 76|76|7.53|7.90 [6.78; 8.97]|
|Task|75 / 19|75|10.71|12.01 [10.64; 13.41]|
|Walking|5 / 5|5|2.00|6.75 [2.50; 11.81]|

## 2. How many videos have a peak near PR?

Coverage ±5 bpm means at least one candidate within 5 bpm of PR. Videos without candidates count as uncovered. Values below are percentages of videos.

|Group|True ±5 coverage (%)|Shuffled ±5: mean [P5–P95] (%)|True ±10 coverage (%)|Shuffled ±10: mean [P5–P95] (%)|
|---|---:|---|---:|---|
|All|47.71|40.79 [35.78; 46.38]|75.69|69.59 [65.11; 73.90]|
|Normal — resting|46.77|43.81 [32.26; 54.84]|85.48|75.90 [69.27; 83.87]|
|Abnormal — resting|53.95|43.89 [35.53; 52.63]|77.63|74.46 [65.79; 81.58]|
|Task|38.67|34.92 [26.67; 42.67]|64.00|58.52 [52.00; 65.33]|
|Walking|100.00|44.40 [19.00; 100.00]|100.00|83.40 [60.00; 100.00]|

## 3. Two practical baselines on the same videos with candidates

The constant is median PR from four training folds, excluding the evaluated person's PR. The calibrated median is the retained, previously saved method. The oracle sees labels and cannot be claimed as an inference-method win over these baselines.

|Group|Method|Videos / people|MAE (bpm)|Person-bootstrap CI95|
|---|---|---:|---:|---|
|All|Training-fold constant|218 / 158|12.07|[10.58; 13.68]|
|All|Calibrated median|218 / 158|11.37|[9.92; 12.89]|
|Normal — resting|Training-fold constant|62 / 62|8.66|[7.05; 10.39]|
|Normal — resting|Calibrated median|62 / 62|9.18|[7.51; 10.90]|
|Abnormal — resting|Training-fold constant|76 / 76|14.20|[11.83; 16.75]|
|Abnormal — resting|Calibrated median|76 / 76|12.71|[10.41; 15.12]|
|Task|Training-fold constant|75 / 19|14.80|[10.23; 20.29]|
|Task|Calibrated median|75 / 19|12.95|[8.74; 18.10]|
|Walking|Training-fold constant|5 / 5|12.40|[6.00; 19.20]|
|Walking|Calibrated median|5 / 5|10.76|[4.66; 18.13]|

## 4. Preregistered gates

|Condition|Requirement|Measured|
|---|---:|---:|
|Oracle improvement over shuffled mean|At least 4 bpm|0.69 bpm|
|±5 coverage improvement over shuffled|At least 15 percentage points|6.91 percentage points|

**Result: FAIL**. Both conditions must pass.

Do not implement a reader under this conditional branch. Retain the frozen selector/thresholds and wait for V5's final disposition before freezing.

## 5. Limitations to retain in reporting

Videos-per-person strata: {'1': 135, '2': 4, '6': 6, '3': 13}. Mean proportion of people whose label blocks change: 97.34%. Singleton strata cannot exchange blocks; do not claim everyone was shuffled.

Spectra use saved multi-ROI RGB traces, without video rereading or holdout. This extractor differs from the public canonical oval repair. Conclusions apply only to the frozen three-peak limit and spectral aggregation. They neither establish loss of all pulse information nor identify labels/codecs as the cause. Low oracle error does not establish a learnable peak selector.

V5 and P100 backup continue independently. No additional GPU or holdout access. Definitions, hashes, candidates, and 100 permutations are in GATES.md, PREREGISTRATION.json, CANDIDATES_FROZEN.json, RESULTS.json, and PERMUTATIONS.json.

## 6. Interpretation and independent checks

All 218 videos from 158 people have three candidates. True oracle MAE is 7.55 bpm (person-bootstrap CI95%: 6.44–8.75), versus shuffled 8.23 bpm. Coverage ±5: 47.71% versus 40.79%; ±10: 75.69% versus 69.59%. This is a small advantage over chance matching, below the required effect size for developing a reader.

Descriptive permutation p-values are 0.0396 for oracle and 0.0297 for ±5 coverage. They do not replace frozen effect-size thresholds or reopen the gate after results. Do not claim zero signal or loss of all pulse information.

Walking has only 5 videos/5 people. Do not select the best subgroup to replace the overall conclusion. People occurring in multiple activities count in each corresponding group; do not sum group person counts.

Only saved BTC dev traces were read. Preparation read PR to reconcile cohorts and retain two baselines; the separate peak extractor received no PR. Candidate SHA256 was frozen before scoring against PR. EXECUTION_CLARIFICATION.md distinguishes this from the stronger claim that no process opened labels before extraction.

The numerical spectrum grid is 0.5 bpm; this does not provide physical resolution of 0.5 bpm. A 10-second window has fundamental frequency resolution of about 6 bpm.

Result-package SHA256 and frozen source/gate hashes were rechecked; person-level MAE and both coverage rates were independently recomputed from candidate lists. All match. Spectra/peak selection were not recomputed or changed after viewing labels.

Completed at 07:32:56 ICT on 12 November 2026. The reader branch stopped at its gate; V5 and P100 backup continued independently. Holdout evaluations: 0.

Reproduction files: core.py (extraction/statistics), run.py (orchestration), GATES.md (rules), PREREGISTRATION.json (pre-run hashes), CANDIDATES.json (three peaks/video), CANDIDATES_FROZEN.json (pre-scoring hash), PERMUTATIONS.json (100 permutations), RESULTS.json, and COMPLETION_REVIEW.json. render_report.py only reformats saved values; it does not rerun the test.
