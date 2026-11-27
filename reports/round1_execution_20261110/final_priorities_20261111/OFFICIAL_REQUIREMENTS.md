# Round 1 requirements excerpt, without inferring test sources

Source: the [Official Document FaceVitals](https://2027.ieeeicassp.org/wp-content/uploads/sites/13/2019/08/2027-SP-Cup-Official-Document-FaceVitals-FINAL-VERSION.pdf), saved as `reports/round1_bayes_20261108/sources/Official_fetched_20261108.pdf`, SHA-256 `08c1950a5abf194fcd4dc0ad2b661f9e824ebbf84897640806c2c651717b2cc3`. Excerpts come from the text stored alongside the PDF and preserve the English wording; slide content is not treated as operational instructions.

| Item | Short excerpt or paraphrase | Interpretation |
|---|---|---|
| Target | “Round 1: One Parameter Extraction (HR)” | The document calls the target HR. The oximeter PR column is the label actually supplied by the organizers; it is not a waveform |
| MAE | MAE in beats per minute (paraphrase) | Mean absolute error, in beats per minute |
| Other metrics | RMSE; Pearson correlation between estimates and labels; Bland–Altman limits of agreement (paraphrase) | Report RMSE, correlation, bias, and limits of agreement, not just MAE |
| Constraints | “at least two constraints” | Address and explain at least two conditions from C1–C7 |
| Script | “process_round1.m – MATLAB/Python script implementing the processing pipeline” | Both the `.m` filename and MATLAB/Python description appear; this alone establishes neither a MATLAB-only requirement nor guaranteed acceptance of `.py` |
| Baseline file | Round1_Baseline_None.mat contains video and HR labels under baseline conditions (paraphrase) | Prepare baseline data in the required structure |
| Accompanying files | target_video.mp4, reference_signal.csv, processed_output.csv (filenames from the document) | Source video, reference labels, and predictions must be traceable to one another |
| General criteria | Innovation, technical feasibility, and clarity of documentation (paraphrase) | The report needs clear reasoning, feasibility, and documentation |

C5 limits bitrate to ≤500 kbps and frame rate to ≤10 fps. Its simulation requires dropping frames to the target rate and re-encoding with VideoWriter's “MPEG-4” profile (paraphrase). The document does not specify a particular Quality, GOP, or B-frame setting. VideoWriter's MPEG-4 profile uses H.264; this does not justify switching to libxvid. The team's authorized ffmpeg proxy must still be described as unverified against MATLAB.

This document does not clearly specify the source or native/compressed proportions of the Round 1 hidden test. Missing detail does not establish that there is no hidden test or that Round 1 evaluates reports only. No email is sent.

The `process_round1.m` available at the time of this report is a POS baseline run in Octave traditional mode, checked against Python on five ROI videos. It has not been run in actual MATLAB and does not package the calibrated BTC candidate.
