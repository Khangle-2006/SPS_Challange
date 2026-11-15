"""Small verified report export manifest; media and PSD arrays remain on P100."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
names='''FINDINGS.md WALKING_PTS_CHECK.json SUMMARY.md REPORT.md REPORT_STATUS.json FINAL_VALIDATION.json VERIFIED_ARTIFACTS.csv PLAN.json PROTOCOL.md READOUT_AUDIT.json PREFLIGHT.json TEST_V2.json AMENDMENT_V2.json AMENDMENT_V3.json AMENDMENT_V4.json MAPPING_CHECK_V3.json MCD_VALIDITY_INSPECTION.json REAL_COMPLETE.json SYNTH_COMPLETE.json WORKER_V3.json WORKER_V4.json A100_LAUNCH_INVENTORY.json A100_FINAL_INVENTORY.json ALL_REAL_SPECTRA.csv ALL_SYNTHETIC_SPECTRA.csv ENCODER_SCHEDULES.csv PAIRED_CLOCK_MOVEMENT.csv PAIRED_FPS_MOVEMENT.csv BEFORE_AFTER.csv ROI_COMPARISON.csv COVERAGE.csv REFERENCE_COVERAGE.csv WINDOW_PREDICTIONS.csv VIDEO_PREDICTIONS.csv BTC_BASELINES_MATCHED.csv BTC_METRICS.json MOVEMENT_SUMMARY.json audit_clock.py run_v2.py run_v3.py run_v4.py report_clock.py final_checks.py summarize_final.py export_reports.py EXCLUDED_SOURCE.json'''.split()
assert json.loads((P/'FINAL_VALIDATION.json').read_text())['status']=='PASS'
(P/'EXCLUDED_SOURCE.json').write_bytes((P/'real/MCD/1474_FullHDwebcam_after/EXCLUDED.json').read_bytes())
receipt={n:dict(bytes=(P/n).stat().st_size,sha256=hashlib.sha256((P/n).read_bytes()).hexdigest()) for n in names}
(P/'REPORT_EXPORT_SHA256.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(files=len(receipt),bytes=sum(x['bytes'] for x in receipt.values()))))
