"""Concatenate disjoint outer folds, score OOF and pair subjects across arms/seeds."""
import argparse
import csv
from pathlib import Path
import numpy as np
from t8_plan import read, write, SEEDS
from t8_evaluate import csv_write
from score_round1 import score, key


def load_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def aggregate(root, out):
    if out.exists():
        raise FileExistsError(out)
    # Check all runs first; no partial protocol score masquerades as full S.
    for fold in range(5):
        for seed in SEEDS:
            for arm in ('ubfc_only', 'joint'):
                run = root/f'fold{fold}'/f'seed{seed}'/arm
                assert read(run/'status.json')['state'] == 'complete'
    out.mkdir()
    all_metrics, scores = {}, {}
    conditions = ('UBFC2_native', 'UBFC2_f10_b500', 'MCD_native', 'MCD_f10_b500')
    for seed in SEEDS:
        all_metrics[str(seed)], scores[str(seed)] = {}, {}
        for arm in ('ubfc_only', 'joint'):
            all_metrics[str(seed)][arm] = {}
            for condition in conditions:
                refs, predictions = [], []
                seen_subjects = set()
                for fold in range(5):
                    directory = root/f'fold{fold}'/f'seed{seed}'/arm/'evaluation'/condition
                    fold_refs = load_csv(directory/'reference_signal.csv')
                    subjects = {r['subject_id'] for r in fold_refs}
                    assert seen_subjects.isdisjoint(subjects), 'Outer subject leakage'
                    seen_subjects |= subjects
                    refs.extend(fold_refs)
                    predictions.extend(load_csv(directory/'processed_output.csv'))
                assert len(seen_subjects) == (35 if condition.startswith('UBFC2') else 60)
                assert len({key(r) for r in refs}) == len(refs)
                dest = out/f'seed{seed}_{arm}_{condition}'
                dest.mkdir()
                csv_write(dest/'reference_signal.csv', refs)
                csv_write(dest/'processed_output.csv', predictions)
                metric = score(dest/'reference_signal.csv', dest/'processed_output.csv')
                write(dest/'metrics.json', metric)
                all_metrics[str(seed)][arm][condition] = metric
            values = [all_metrics[str(seed)][arm][c]['subject_macro']['mae'] for c in conditions[:3]]
            scores[str(seed)][arm] = None if None in values else float(np.mean(values))
    # Compare only identical canonical references; keep missing-subject MAE undefined.
    paired = {}
    for condition in conditions:
        by_seed = []
        for seed in SEEDS:
            u, j = (all_metrics[str(seed)][arm][condition] for arm in ('ubfc_only', 'joint'))
            assert u['reference_sha256'] == j['reference_sha256']
            subjects = sorted(u['per_subject'])
            assert subjects == sorted(j['per_subject'])
            by_seed.append([None if u['per_subject'][s]['mae'] is None or j['per_subject'][s]['mae'] is None
                            else j['per_subject'][s]['mae']-u['per_subject'][s]['mae'] for s in subjects])
        if any(x is None for row in by_seed for x in row):
            paired[condition] = dict(status='undefined_missing_subject_prediction')
            continue
        values = np.asarray(by_seed)
        rng = np.random.default_rng(20261031)
        # One resampled person carries all three seeds; seeds are not independent people.
        samples = rng.integers(values.shape[1], size=(10000, values.shape[1]))
        boot = values[:, samples].mean(axis=(0, 2))
        paired[condition] = dict(subjects=subjects, joint_minus_control_by_seed=values.tolist(),
            mean_delta=float(values.mean()), subject_bootstrap_CI95=np.quantile(boot, [.025, .975]).tolist(),
            note='All seeds kept together within resampled subject; development OOF only')
    summary = dict(S_by_seed=scores, paired_conditions=paired,
        scope='35 UBFC development and MCD dev60 OOF; 7 UBFC test and D3 closed',
        S_definition='mean(UBFC2 native, UBFC2 compliant C5 10fps500cap, MCD native) subject-macro MAE',
        caution='C5 MCD diagnostic is excluded from S; coverage and invalid references retained in full metrics')
    write(out/'SUMMARY.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    aggregate(a.runs, a.output)
