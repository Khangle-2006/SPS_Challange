"""Portable CLI around the archived POS/Welch numerical baseline (no labels)."""
import argparse
import csv
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def baseline():
    path = ROOT / 'reports/round1_execution_20261110/portable_python.py'
    spec = importlib.util.spec_from_file_location('portable_pos', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create_demo(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(901) / 30
    pulse = np.sin(2 * np.pi * 1.5 * t)
    rgb = np.column_stack((100 + .1*pulse, 120 + pulse, 80 + .2*pulse))
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_s', 'red', 'green', 'blue', 'valid'])
        w.writerows([*row, 1] for row in np.column_stack((t, rgb)))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--make-demo', type=Path)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.make_demo:
        if a.input or a.output:
            p.error('--make-demo cannot be combined with inference')
        create_demo(a.make_demo)
        print('Synthetic 90 bpm RGB demo written; no participant data.')
        return
    if not a.input or not a.output:
        p.error('--input and --output are required')
    with a.input.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = ['timestamp_s', 'red', 'green', 'blue', 'valid']
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError('Expected CSV columns: ' + ','.join(fields))
        data = np.asarray([[float(row[k]) for k in fields] for row in reader])
    if data.ndim != 2 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError('Need finite, nonempty RGB trace')
    t, rgb, valid = data[:, 0], data[:, 1:4], data[:, 4]
    if t[0] != 0 or not np.all(np.diff(t) > 0):
        raise ValueError('Timestamp must start at zero and strictly increase')
    if not np.isin(valid, [0, 1]).all():
        raise ValueError('valid must be 0 or 1')
    if t[-1] < 10:
        raise ValueError('Less than 10 seconds; no padding/extrapolation')
    rows = baseline().predict(t, rgb, valid.astype(bool))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['window_start_s', 'window_end_s', 'PR_bpm', 'valid'])
        w.writerows(rows)
    accepted = rows[np.isfinite(rows[:, 2]), 2]
    print(f'POS baseline: {len(accepted)}/{len(rows)} valid windows; '
          f'median PR = {float(np.median(accepted)) if len(accepted) else None}')

if __name__ == '__main__':
    main()
