# Limitations and release status

1. README results come from completed experiments; this source release did not rerun training, participant inference, or holdout. Release checks cover syntax, hashes, scorer behavior, and separately documented synthetic smoke tests.
2. Raw videos, individual PR/PPG, person splits, physiological traces, face images, checkpoints, and participant-bearing manifests are excluded. Exact reproduction requires these authorized assets; see the guides and schemas. Cloning alone does not supply them.
3. Many research scripts are Linux-server snapshots tied to hashed receipts. They are preserved for traceability rather than having all paths rewritten. Configure an isolated environment/output before running historical launchers/controllers on a real machine.
4. Rejected configurations remain in source: V2 was not launched; pre-polarity joint runs are excluded from harmonized comparisons; original V4 padding failed static input; SiNC/weak-label was rejected; V5 did not win. File presence is not a recommendation.
5. C5′ has no MATLAB VideoWriter validation; `process_round1.m` accepts precomputed ROI and is a POS baseline. Octave parity is neither MATLAB parity nor parity with the calibrated BTC estimator.
6. Dev was used for many experiments; paired confidence intervals do not remove repeated-selection bias. Holdout remains closed. The repository contains no hidden-test scores or official competition results.
7. Codec/HLG/rotation conclusions apply only where support exists. The stage4 ladder loses tracking, so later stages cannot isolate HR effects. Without simultaneous webcam–phone pairs, acquisition is not established as the cause.
8. Failure of the three-spectral-peak test limits only that fixed extractor; it does not show that all pulse information is absent.
9. The full calibrated estimator lacks confirmed end-to-end static-input controls. The V3 component has static-input issues; retaining 11.37 is a research disposition, not deployment certification.
