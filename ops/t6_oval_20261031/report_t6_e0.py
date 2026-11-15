"""Summarize completed QA-lite E0 receipts without changing any processing."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def metrics(rows, method, subjects):
    by_subject = defaultdict(list)
    errors = []
    for row in rows:
        pred = row['predictions'][method]
        ref = row['reference_hr']
        if pred is not None and ref is not None:
            error = pred-ref
            by_subject[row['subject']].append(error)
            errors.append(error)
    e = np.asarray(errors)
    per_subject = {s: float(np.mean(np.abs(by_subject[s]))) if by_subject[s] else None for s in subjects}
    return dict(accepted=len(e), expected=len(rows), coverage=len(e)/len(rows) if rows else None,
        subject_macro_MAE=float(np.mean(list(per_subject.values()))) if all(v is not None for v in per_subject.values()) else None,
        window_MAE=float(np.abs(e).mean()) if len(e) else None,
        P95=float(np.quantile(abs(e), .95)) if len(e) else None,
        error_gt20=int(np.sum(abs(e)>20)), bias=float(e.mean()) if len(e) else None,
        per_subject_MAE=per_subject)


def summarize(records):
    rows = [w for r in records for w in r['rows']]
    primary = [w for w in rows if w['primary_E0_paired']]
    if len(records) != 119 or len(primary) != 1860:
        raise ValueError('Require all119records and original1860primary windows; no partial result as fullcohort')
    subjects = sorted({w['subject'] for w in primary})
    assert len(subjects) == 60
    result = dict(source='E0 H264/nativeFPS, large compressed-video oval v2; not compliantC5 or Round1S',
        records=len(records), subjects=len(subjects), all_candidate_rows=len(rows), primary_windows=len(primary),
        metrics={method: metrics(primary, method, subjects) for method in ('control', 'qa')},
        before_after={}, paired={}, T5_promotion_decision='pending_T5_compliant_10fps_both_datasets')
    for condition in ('before', 'after'):
        group = [w for w in primary if w['step'] == condition]
        ss = sorted({w['subject'] for w in group})
        result['before_after'][condition] = {m: metrics(group, m, ss) for m in ('control', 'qa')}
    deltas = defaultdict(list)
    for w in primary:
        if all(w['predictions'][m] is not None for m in ('control', 'qa')):
            deltas[w['subject']].append(abs(w['predictions']['qa']-w['reference_hr'])-
                                        abs(w['predictions']['control']-w['reference_hr']))
    person = np.array([np.mean(deltas[s]) for s in subjects if deltas[s]])
    rng = np.random.default_rng(20261031)
    boot = rng.choice(person, size=(10000, len(person)), replace=True).mean(1) if len(person) else np.array([])
    result['paired'] = dict(windows=sum(map(len, deltas.values())), subjects=len(person),
        delta_MAE=float(person.mean()) if len(person) else None,
        CI95=np.quantile(boot, [.025, .975]).tolist() if len(boot) else None,
        per_subject_delta={s: float(np.mean(v)) for s, v in deltas.items()},
        coverage_equal=result['metrics']['qa']['accepted'] == result['metrics']['control']['accepted'])
    result['diagnostics'] = {r['recording_id']: r['diagnostics'] for r in records}
    result['control_max_trace_error_DN'] = max(r['max_trace_error_DN'] for r in records)
    result['E0_nonharm_check'] = bool(len(person) == 60 and result['paired']['coverage_equal'] and person.mean() <= .5)
    return result, rows


def main(a):
    cohort = json.loads(a.cohort.read_text())
    records = [json.loads((a.input/(Path(r['video']).stem+'.json')).read_text()) for r in cohort]
    fingerprints = {json.dumps(r['fingerprint'], sort_keys=True) for r in records}
    assert len(fingerprints) == 1
    result, rows = summarize(records)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    (a.output/'all_windows.json').write_text(json.dumps(rows, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].hist(list(result['paired']['per_subject_delta'].values()), bins=20, color='#3874a8')
    axes[0].axvline(0, color='black', linestyle='--')
    axes[0].set(xlabel='QA minus control MAE (bpm)', ylabel='Subjects', title='Paired subject effects')
    subjects = sorted(result['metrics']['control']['per_subject_MAE'])
    x = [result['metrics']['control']['per_subject_MAE'][s] for s in subjects]
    y = [result['metrics']['qa']['per_subject_MAE'][s] for s in subjects]
    axes[1].scatter(x, y, alpha=.7, s=25)
    maximum = max([v for v in x+y if v is not None], default=1)
    axes[1].plot([0, maximum], [0, maximum], '--', color='black')
    axes[1].set(xlabel='Control MAE (bpm)', ylabel='QA MAE (bpm)', title='Same60 development subjects')
    axes[2].hist([r['diagnostics']['gate_fraction'] for r in records], bins=20, color='#ad6b38')
    axes[2].set(xlabel='Fraction of frames with gate open', ylabel='Videos', title='Image-only gate activation')
    fig.suptitle('T6 E0: H264 at native FPS; not official C5 operating point')
    fig.savefig(a.output/'comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('input', 'cohort', 'output'):
        p.add_argument('--'+k, type=Path, required=True)
    main(p.parse_args())
