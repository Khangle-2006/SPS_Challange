"""Read-only CPU check on the exact frozen development index, never D3.

Run explicitly where native caches exist. Writes only small reports to --output.
The policy is already fixed; this script tests it and cannot change its signs.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from ppg_polarity import adapt_cached_meta, RAW, POLICY_SHA256
from morphology_contract import assess_dataset, extract_cycles

INDEX_SHA = '12a87b829ab5e158b430c67701fa0dd978c9a3d592e0060da3e876069108d505'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    if sha(args.index) != INDEX_SHA:
        raise ValueError('Only frozen development index is authorized by this check')
    index = json.loads(args.index.read_text())
    grouped = {'MCD': [], 'UBFC2': []}
    rows = []
    metadata_hashes = {}
    for item in index['records']:
        dataset, recording = item['dataset'], item['recording_id']
        if dataset not in grouped:
            raise ValueError('Unexpected dataset in frozen development index')
        path = Path(index['roots'][dataset])/recording/'native'/'crop'/'meta.npz'
        metadata_hashes[dataset+':'+recording] = sha(path)
        with np.load(path) as z:
            # No image, raw video, or additional partition access.
            meta = {k: z[k] for k in ('ppg', 'ppg_t', 'ppg_valid')}
        adapted, receipt = adapt_cached_meta(meta, dataset, source_convention=RAW)
        allowed = np.zeros(len(meta['ppg_t']), bool)
        for start in item['eligible_starts']:
            allowed |= (meta['ppg_t'] >= start) & (meta['ppg_t'] <= start+10)
        adapted['ppg_valid'] = meta['ppg_valid'].astype(bool) & allowed
        adapted['ppg_polarity'] = receipt
        grouped[dataset].append(adapted)
        cycles = extract_cycles(adapted['ppg'], adapted['ppg_t'], adapted['ppg_valid'])
        peak = float(np.argmax(cycles.mean(axis=0))/100) if len(cycles) else None
        rows.append(dict(dataset=dataset, recording=recording, subject=item['subject_id'],
                         valid_cycles=len(cycles), mean_rise_phase=peak,
                         mean_fall_phase=1-peak if peak is not None else None))
    report = dict(evidence_cohort='frozen_development_index_only_no_D3',
                  index_sha256=INDEX_SHA, policy_sha256=POLICY_SHA256,
                  source_metadata_sha256=metadata_hashes,
                  code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                  selection='native PPG supported by frozen training-eligible windows',
                  alignment='sign applied BEFORE cycle detection and trough alignment',
                  datasets={ds: assess_dataset(records, ds) for ds, records in grouped.items()})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'morphology_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output/'recording_morphology.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({ds: {k: v for k, v in value.items() if k != 'mean_cycle'}
                      for ds, value in report['datasets'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args())
