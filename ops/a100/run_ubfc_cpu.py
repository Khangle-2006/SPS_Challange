"""Audit raw UBFC files and run fixed POS/GREEN protocol, without training."""
import concurrent.futures as cf
import hashlib
import json
import os
import math
import time
from pathlib import Path
import numpy as np
import cv2
from hr_baseline import read_rgb, align_timestamps, pos, estimate_hr, metrics, save, selftest

ROOT = Path('/data/khangle/facevitals')
DATA = ROOT / 'datasets/UBFC-rPPG/DATASET_2'
OUT = ROOT / 'results/ubfc_pos_green_v1'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}

def run_subject(task):
    subject, split = task
    result = {'subject': subject, 'split': split, 'files': {}, 'ok': False}
    try:
        folder = DATA / subject
        for name in ('vid.avi', 'ground_truth.txt'):
            result['files'][name] = digest(folder / name)
        with (folder / 'vid.avi').open('rb') as f:
            header = f.read(12)
        if header[:4] != b'RIFF' or header[8:12] != b'AVI ':
            raise ValueError('Invalid AVI signature')
        # Streaming full decode; core reader checks decoded versus declared frames.
        rgb, nominal_fs, info = read_rgb(folder / 'vid.avi', fullface=True)
        result['video'] = info
        np.savez_compressed(OUT / (subject + '_raw_rgb.npz'), rgb=rgb, nominal_fs=nominal_fs)
        gt = np.loadtxt(folder / 'ground_truth.txt')
        if gt.ndim != 2 or gt.shape[0] != 3 or not np.isfinite(gt).all():
            raise ValueError('Ground truth must contain three finite rows')
        result['label_samples'] = gt.shape[1]
        rgb, ppg, sensor, t, fs, timing = align_timestamps(rgb, gt)
        result['timing'] = timing
        pred = pos(rgb, fs)
        trim = math.ceil(1.6 * fs)
        keep = slice(trim, len(t) - trim)
        row = {'subject': subject, 'split': split,
               'pos_bpm': estimate_hr(pred[keep], fs),
               'green_bpm': estimate_hr(rgb[keep, 1], fs),
               'gt_ppg_bpm': estimate_hr(ppg[keep], fs),
               'sensor_hr_bpm': float(sensor[keep].mean()),
               'measured_fs': fs, 'nominal_fs': nominal_fs,
               'duration_s': float(t[-1]), **info, **timing}
        windows = []
        step = round(10 * fs)
        for start in range(trim, len(t) - trim - step + 1, step):
            end = start + step
            windows.append({'subject': subject, 'split': split, 'start_s': start / fs,
                            'pos_bpm': estimate_hr(pred[start:end], fs),
                            'green_bpm': estimate_hr(rgb[start:end, 1], fs),
                            'gt_ppg_bpm': estimate_hr(ppg[start:end], fs),
                            'sensor_hr_bpm': float(sensor[start:end].mean())})
        if not windows:
            raise ValueError('No complete evaluation windows')
        np.savez_compressed(OUT / (subject + '_signals.npz'), rgb=rgb, ppg=ppg,
                            pos=pred, sensor_hr=sensor, t=t)
        result.update(ok=True, recording=row, windows=windows)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result

def report(results, split, finished=False):
    good = sorted([r for r in results if r['ok']], key=lambda r: int(r['subject'][7:]))
    rows = [r['recording'] for r in good]
    windows = [w for r in good for w in r['windows']]
    summary = {'dataset': 'UBFC-rPPG Dataset 2', 'expected_subjects': 42,
               'attempted': len(results), 'processed': len(good),
               'finished': finished, 'complete': len(good) == 42,
               'failures': [{'subject': r['subject'], 'error': r['error']} for r in results if not r['ok']],
               'protocol': 'POS 1.6s; GREEN; actual GT timestamps; duplicate timestamps merged consistently; trim 1.6s both ends; fixed 45-180 bpm; no label-driven tuning',
               'split': split,
               'source_checksum_status': 'Local SHA256 inventory only; source checksum comparison is separate.',
               'record_pos_vs_ppg': metrics(rows),
               'record_green_vs_ppg': metrics(rows, 'green_bpm'),
               'record_pos_vs_sensor': metrics(rows, truth='sensor_hr_bpm'),
               'record_green_vs_sensor': metrics(rows, 'green_bpm', 'sensor_hr_bpm'),
               'window_pos_vs_ppg': metrics(windows),
               'window_green_vs_ppg': metrics(windows, 'green_bpm')}
    for tag in ('train', 'val', 'test'):
        for method in ('pos', 'green'):
            for scale, items in (('record', rows), ('window', windows)):
                summary[f'{tag}_{scale}_{method}_vs_ppg'] = metrics([r for r in items if r['split'] == tag], method + '_bpm')
    save(rows, OUT / 'recordings.csv')
    save(windows, OUT / 'windows.csv')
    (OUT / 'audit.json').write_text(json.dumps([{k: v for k, v in r.items() if k not in ('recording', 'windows')} for r in results], indent=2))
    temp = OUT / 'summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2))
    temp.replace(OUT / 'summary.json')
    return summary

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    selftest()
    manifest = json.loads((DATA.parent / 'drive_manifest.json').read_text())
    subjects = sorted({r['path'].split('/')[0] for r in manifest}, key=lambda s: int(s[7:]))
    assert len(manifest) == 84 and len(subjects) == 42
    expected = {r['path'] for r in manifest}
    actual = {str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file() and p.name in ('vid.avi', 'ground_truth.txt')}
    assert actual == expected, {'missing': sorted(expected - actual), 'extra': sorted(actual - expected)}
    shuffled = np.random.default_rng(2027).permutation(subjects).tolist()
    split = {'seed': 2027, 'train': shuffled[:28], 'val': shuffled[28:35], 'test': shuffled[35:]}
    (OUT / 'subject_split.json').write_text(json.dumps(split, indent=2))
    (OUT / 'run_provenance.json').write_text(json.dumps({'started_unix': time.time(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'numpy': np.__version__, 'opencv': cv2.__version__, 'code': {f: digest(Path(__file__).parent / f) for f in ('run_ubfc_cpu.py', 'hr_baseline.py')}}, indent=2))
    results = []
    tasks = [(s, next(k for k in ('train', 'val', 'test') if s in split[k])) for s in subjects]
    with cf.ProcessPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))) as pool:
        futures = [pool.submit(run_subject, task) for task in tasks]
        for f in cf.as_completed(futures):
            result = f.result()
            results.append(result)
            report(results, split)
            print(json.dumps({'done': len(results), 'subject': result['subject'], 'ok': result['ok'], 'error': result.get('error')}), flush=True)
    summary = report(results, split, finished=True)
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(0 if summary['complete'] else 2)
