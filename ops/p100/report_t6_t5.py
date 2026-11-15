"""Full-cohort canonical T6 C5 scoring. Never score a ready-only cohort as complete."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from score_round1 import score, key


def read(path):
    return json.loads(Path(path).read_text())


def csv_read(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    assert rows
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(control, qa):
    assert set(control['per_subject']) == set(qa['per_subject'])
    # Paired window effects require both outputs, then average within person.
    cq = {tuple(r[k] for k in ('subject_id', 'recording_id', 'window_start_s')): r for r in qa['windows']}
    per_person = {}
    paired = 0
    for r in control['windows']:
        other = cq[tuple(r[k] for k in ('subject_id', 'recording_id', 'window_start_s'))]
        assert r['reference_hr_bpm'] == other['reference_hr_bpm']
        if r['accepted'] and other['accepted']:
            ref = r['reference_hr_bpm']
            delta = abs(other['prediction_hr_bpm']-ref)-abs(r['prediction_hr_bpm']-ref)
            per_person.setdefault(r['subject_id'], []).append(delta)
            paired += 1
    values = np.array([np.mean(per_person[s]) for s in sorted(per_person)])
    if len(values):
        rng = np.random.default_rng(20261031)
        boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(1)
        ci, mean = np.quantile(boot, [.025, .975]).tolist(), float(values.mean())
    else:
        ci, mean = None, None
    # No hidden substitution of successful windows: same accepted identities required
    # for a performance-promotion conclusion, not merely equal total coverage.
    caccepted = {tuple(r[k] for k in ('subject_id','recording_id','window_start_s')) for r in control['windows'] if r['accepted']}
    qaccepted = {tuple(r[k] for k in ('subject_id','recording_id','window_start_s')) for r in qa['windows'] if r['accepted']}
    return dict(windows=paired, subjects=len(values), delta_MAE=mean, CI95=ci,
        control_accepted_subset_of_QA=caccepted <= qaccepted,
        identical_accepted_windows=caccepted == qaccepted,
        per_subject_delta={s: float(np.mean(v)) for s,v in per_person.items()})


def main(a):
    cohort, status = read(a.input/'cohort.json'), read(a.input/'status.json')
    expected = 35 if cohort['dataset'] == 'UBFC2' else 119
    assert status['state'] == 'complete' and not status['errors']
    assert status['completed'] == cohort['expected_records'] == len(cohort['recordings']) == expected
    records = [read(a.input/rid/'complete.json') for rid in cohort['recordings']]
    assert len({json.dumps(r['fingerprint'], sort_keys=True) for r in records}) == 1
    refs, predictions = [], {'control': [], 'qa': []}
    for rid in cohort['recordings']:
        refs.extend(csv_read(a.input/rid/'reference_signal.csv'))
        for method in predictions:
            predictions[method].extend(csv_read(a.input/rid/(method+'_output.csv')))
    assert len(refs) == cohort['expected_windows'] and len({key(r) for r in refs}) == len(refs)
    a.output.mkdir(parents=True, exist_ok=False)
    csv_write(a.output/'reference_signal.csv', refs)
    metrics = {}
    for method in predictions:
        path = a.output/(method+'_output.csv')
        csv_write(path, predictions[method])
        metrics[method] = score(a.output/'reference_signal.csv', path)
        (a.output/(method+'_metrics.json')).write_text(json.dumps(metrics[method], indent=2))
    pair = paired_statistics(metrics['control'], metrics['qa'])
    compliant = [r for r in records if r['C5_compliant']]
    summary = dict(dataset=cohort['dataset'], records=len(records), expected_windows=len(refs),
        C5_compliant_records=len(compliant), all_records_C5_compliant=len(compliant)==expected,
        bitrate_kbps=[r['bitrate_kbps'] for r in records],
        effective_sensor_fps=[r['effective_sensor_fps'] for r in records],
        decoder_and_reference_hz=[.7,3.], resampling_hz=30, offsets_changed=False,
        no_test_access=True, paired=pair,
        metrics={m: {k:metrics[m][k] for k in ('counts','subject_macro','window_level','reference_coverage')} for m in metrics},
        diagnostics={r['recording_id']: r['diagnostics'] for r in records},
        overall_promotion='pending_other_dataset_and_E0_nonharm_check',
        interpretation='Development POS diagnostic, not EfficientPhys or Round1 S')
    summary['before_after'] = {}
    if cohort['dataset'] == 'MCD':
        for condition in ('before', 'after'):
            ids = {r['recording_id'] for r in records if r['rows'] and r['rows'][0]['step'] == condition}
            subset = a.output/condition
            subset.mkdir()
            csv_write(subset/'reference_signal.csv', [r for r in refs if r['recording_id'] in ids])
            summary['before_after'][condition] = {}
            for method in predictions:
                csv_write(subset/(method+'_output.csv'), [r for r in predictions[method] if r['recording_id'] in ids])
                detail = score(subset/'reference_signal.csv', subset/(method+'_output.csv'))
                summary['before_after'][condition][method] = {k:detail[k] for k in ('counts','subject_macro','window_level')}
    summary['this_dataset_promotion_gate_passed'] = bool(len(compliant)==expected and pair['delta_MAE'] is not None
        and pair['delta_MAE'] <= -.5 and pair['CI95'][1] < 0 and pair['control_accepted_subset_of_QA']
        and pair['subjects'] == metrics['control']['counts']['subjects']
        and metrics['control']['subject_macro']['mae'] is not None and metrics['qa']['subject_macro']['mae'] is not None)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(list(pair['per_subject_delta'].values()), bins=20, color='#28749b')
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set(xlabel='QA minus control MAE (bpm)', ylabel='Subjects', title='Paired subject effects')
    axes[1].hist([r['diagnostics']['gate_fraction'] for r in records], bins=20, color='#aa7b38')
    axes[1].set(xlabel='Gate-open fraction', ylabel='Videos', title='Fixed image-only gate')
    fig.suptitle('T6 '+cohort['dataset']+':10fps,500kbps cap;35UBFCdev or60MCDdev only')
    fig.savefig(a.output/'comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
