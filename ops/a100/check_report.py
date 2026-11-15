"""Independently recompute aggregate metrics using Python standard library."""
import csv, json, math
from pathlib import Path
ROOT = Path(__file__).parent / 'run_20261015/results/ubfc_pos_green_v1'
summary = json.loads((ROOT / 'summary.json').read_text())
rows = list(csv.DictReader((ROOT / 'recordings.csv').open()))
windows = list(csv.DictReader((ROOT / 'windows.csv').open()))
checks = []
for key, items, pred, truth in [
    ('record_pos_vs_ppg', rows, 'pos_bpm', 'gt_ppg_bpm'),
    ('record_green_vs_ppg', rows, 'green_bpm', 'gt_ppg_bpm'),
    ('record_pos_vs_sensor', rows, 'pos_bpm', 'sensor_hr_bpm'),
    ('window_pos_vs_ppg', windows, 'pos_bpm', 'gt_ppg_bpm'),
    ('window_green_vs_ppg', windows, 'green_bpm', 'gt_ppg_bpm'),
    ('test_window_pos_vs_ppg', [r for r in windows if r['split']=='test'], 'pos_bpm', 'gt_ppg_bpm')]:
    e = [float(r[pred]) - float(r[truth]) for r in items]
    measured = {'n':len(e), 'MAE_bpm':sum(map(abs,e))/len(e), 'RMSE_bpm':math.sqrt(sum(x*x for x in e)/len(e)), 'bias_bpm':sum(e)/len(e)}
    assert all(abs(measured[k]-summary[key][k])<1e-10 for k in measured), key
    checks.append({'metric':key, **measured})
def worst(items, pred, truth):
    ordered = sorted(items, key=lambda r: abs(float(r[pred])-float(r[truth])), reverse=True)
    return [{k:r[k] for k in ('subject','split',pred,truth)} for r in ordered[:5]]
report = {'independent_metric_recalculation':'passed', 'checks': checks,
          'decoded_frames':sum(int(r['frames']) for r in rows),
          'subjects_with_duplicate_timestamps':sum(int(r['duplicate_timestamps'])>0 for r in rows),
          'duplicate_timestamp_count':sum(int(r['duplicate_timestamps']) for r in rows),
          'minimum_face_coverage':min(float(r['face_coverage']) for r in rows),
          'measured_fps_range':[min(float(r['measured_fs']) for r in rows),max(float(r['measured_fs']) for r in rows)],
          'max_timestamp_gap_s':max(float(r['max_timestamp_gap_s']) for r in rows),
          'worst_pos_recordings':worst(rows,'pos_bpm','gt_ppg_bpm'),
          'largest_reference_disagreements':worst(rows,'gt_ppg_bpm','sensor_hr_bpm')}
(ROOT/'independent_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
