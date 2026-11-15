"""Export timing evidence from completed T9/T6 receipts without changing workers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main(a):
    paths = sorted(a.t9.glob('*/complete.json'))
    assert len(paths) == a.expected_records, (len(paths), a.expected_records)
    rows = []
    for path in paths:
        r = json.loads(path.read_text())
        prior = a.t6 / r['recording_id'] / 'complete.json'
        assert hashlib.sha256(prior.read_bytes()).hexdigest() == r['h26410_receipt_sha256']
        old = json.loads(prior.read_text())
        for cell in ('native', 'raw10', 'h26430', 'h26410'):
            if cell == 'h26410':
                fps, n, mux = old['effective_sensor_fps'], old['diagnostics']['frames'], old['mux_fps']
            else:
                fps, n = r['counts'][cell]['effective_fps'], r['counts'][cell]['frames']
                mux = r['h26430_manifest']['mux_fps'] if cell == 'h26430' else ''
            rows.append(dict(recording_id=r['recording_id'], cell=cell,
                             effective_source_fps=fps, source_samples=n, mux_fps=mux,
                             interpolated_fs_hz=30, POS_window_samples=48,
                             POS_window_seconds=1.6, HR_window_samples=300,
                             HR_window_seconds=10, decoder_fs_hz=30,
                             decoder_band_hz='0.7-3.0',
                             runtime_contract_evidence='timing_contract_20261031/result.json',
                             source='reused_T6_control' if cell == 'h26410' else 'T9_v2'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(dict(records=len(paths), timing_rows=len(rows), output=str(a.output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--t9', type=Path, required=True)
    p.add_argument('--t6', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-records', type=int, default=35)
    main(p.parse_args())
