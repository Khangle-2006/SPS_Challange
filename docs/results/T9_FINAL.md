> Archived aggregate report. Historical pending/status sentences describe its creation date, not current status. See the root README for the 15 November disposition. Private artifacts named here are not bundled.

T9 v2 final —31October2026 10:25UTC
Decision: raw10 is healthy in this fixed diagnostic; no >15bpm investigation
trigger. Strong codec-at-lowFPS sensitivity, not proof of a specific physical
mechanism or of absence of every possible implementation issue. Do not retune
T8 or revive QA-lite. T8 proceeds under existing authorization and resource gates.

35/35 UBFCdevelopment subjects completed,0worker errors. Original7test untouched.
207 independent canonical10s candidates;193 eligible and accepted in EVERY cell;
14 source/reference-invalid remain in denominator. Eligiblecoverage100%; total
referencecoverage93.2367%. No missing predictions and no selection of a favorable
intersection. All193 windows are common for paired estimates.

Cell                         subject-macroMAE  pooledP95  >20bpm/193 pooledbias
native                       1.238421          1.757813   2           -0.462223
raw10(no added compression)   1.027693          1.757813   1           -0.214034
nativeFPS+H264500cap          12.940639         52.470703  50          -9.713508
10fps+H264500cap(C5)          25.541853         67.500000  111         -1.750982
Unitsbpm. Bias=prediction-reference. P95above is pooled-window95th percentile,
not mean per-subjectP95 (also retained in summary.json). All failure denominators
are reported separately; no135bpm penalty. Reference/decoder band0.7-3Hz.

Paired changes in subject-mean absolute error;10000subject bootstrap draws,
seed20261031;35subjects,193common windows:
FPS-only raw10-native:       -0.210728 CI95[-0.715477,0.060064]
Codec at nativeFPS:         +11.702218 CI95[8.374873,15.157115]
Codec at10FPS:              +24.514160 CI95[20.994972,27.941330]
Interaction difference:     +12.811942 CI95[9.841647,15.935287]
Interaction=(h26410-raw10)-(nativeH264-native), using absolute HR errors.
Positive interaction shows combined intervention worsens error more than the
additive comparison on this cohort. It does not isolate codec pixel effects
from variant-specific oval tracking or identify an encoder mechanism.

Timing and bitrate:
Actual sourceFPS native/purecodec23.17833-29.97303; raw10/C5 9.82715-10.00249.
The legacy field h26430 means nativeFPS/allframes here, NOT exactly30FPS.
Purecodec35/35 preserve all sourceframes; sourceframe count=encodedcount in all
receipts. Whole-file bitrate450.789-455.630kbps,35/35<=500. C5 already verified
451.783-455.779kbps35/35. Command target450k leaves headroom for500kcap.
fs_contract.csv includes140rows: source/muxFPS,30Hzinterpolation,48samplePOS
window,300sampleHRwindows,decoder30Hz. Real runtime evidence in
reports/timing_contract_20261031/result.json. No16samplePOS path in this protocol.
RGB interpolation happens BEFORE POS; independent T1rawPPG sensor timestamps.

Provenance:
Native/raw10/codec traces use same frozen36point oval with independent causal
tracking on each actualstream. raw10 retains exactT5C5sourceindices and original
pixels. Purecodec preserves ALLframes; v1superseded because oldT5target30drops
some sourceframes. C5control predictions reused byte-identically from T6oval,
withoutQAweights. Reference and reusedprediction hashes verified for35records;
T6receipt hashes verified by fs_export. Same207canonicalcandidates acrosscells.
Native1.24 is this matchedprotocol result, not historical~2.3 or learnedD1.
No deep-model inference/learning and noRound1S is reported here. Development
cohort, not untouchedtest. Results do not predict how multiFPS-trained T8 fares.

Artifacts:
final/summary.json; *_metrics.json; reference_signal.csv; four*_output.csv;
final/fs_contract.csv; final/comparison.png; final_receipts with per-record
complete.json and CSVs. Frozen report_t9.py used unchanged. Plot visually checked:
bars matchnumbers, histogram includes35subjectinteraction values, labelslegible.
Encodedvideo/raw remain onserver; only smallJSON/CSV/logs downloaded.
