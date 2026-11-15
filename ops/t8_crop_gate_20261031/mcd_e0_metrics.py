"""Fixed E0 window analysis and paired subject-level comparisons."""
from collections import Counter, defaultdict
import numpy as np
from hr_baseline import pos, estimate_hr

ARMS = ['P0R0', 'P0R1', 'P1R0', 'P1R1']


def sample_grid(t, values, valid, grid):
    right = np.searchsorted(t, grid).clip(0, len(t)-1)
    left = (right-1).clip(0, len(t)-1)
    exact = np.isclose(grid, t[right], rtol=0, atol=1e-10)
    left = np.where(exact, right, left)
    ok = ((grid >= t[0]) & (grid <= t[-1]) & valid[left] & valid[right]
          & ((t[right]-t[left]) <= .2))
    values = np.asarray(values)
    if values.ndim == 1:
        interp = np.interp(grid, t, values)
        ok &= np.isfinite(interp)
    else:
        interp = np.stack([np.interp(grid, t, values[:, j]) for j in range(values.shape[1])], axis=1)
        ok &= np.isfinite(interp).all(axis=1)
    return interp, ok


def analyze_windows(t, rgb, ppg, reference_support):
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('Nonmonotonic/short metadata time axis')
    grid = np.arange(int(np.floor(t[-1]*30))+1)/30
    pg, pok = sample_grid(t, ppg, reference_support & np.isfinite(ppg), grid)
    sampled = []
    for a in range(4):
        values = rgb[:, a].mean(axis=1)
        sampled.append(sample_grid(t, values, np.isfinite(values).all(axis=1), grid))
    rows = []
    for k in range(int(np.floor(t[-1]/10))):
        sel = (grid >= k*10) & (grid < (k+1)*10)
        row = dict(index=k, start_s=k*10, reference_hr=None, reference_failure=None,
                   predictions={}, failures={})
        try:
            if sel.sum() != 300 or not pok[sel].all():
                raise ValueError('Incomplete reference support')
            row['reference_hr'] = estimate_hr(pg[sel], 30)
        except ValueError as e:
            row['reference_failure'] = str(e)
        for arm, (rg, valid) in zip(ARMS, sampled):
            try:
                if sel.sum() != 300 or not valid[sel].all():
                    raise ValueError('Incomplete ROI/interpolation support')
                row['predictions'][arm] = estimate_hr(pos(rg[sel], 30), 30)
                row['failures'][arm] = None
            except ValueError as e:
                row['predictions'][arm] = None
                row['failures'][arm] = str(e)
        rows.append(row)
    return rows


def error_metrics(rows, arm):
    err = np.asarray([r['predictions'][arm]-r['reference_hr'] for r in rows])
    subjects = defaultdict(list)
    for r, e in zip(rows, err):
        subjects[r['subject']].append(abs(e))
    return dict(windows=len(rows), subjects=len(subjects),
                window_MAE=float(np.abs(err).mean()) if len(err) else None,
                window_RMSE=float(np.sqrt(np.mean(err**2))) if len(err) else None,
                bias=float(err.mean()) if len(err) else None,
                subject_macro_MAE=float(np.mean([np.mean(v) for v in subjects.values()])) if subjects else None,
                errors_over20=int(np.sum(np.abs(err)>20)))


def summarize(records, expected=119):
    rows = [{**w, 'subject':r['subject'], 'video':r['video'], 'step':r['step']}
            for r in records for w in r.get('windows', [])]
    reference = [r for r in rows if r['reference_hr'] is not None]
    matched = [r for r in reference if all(r['predictions'][a] is not None for a in ARMS)]
    effects = defaultdict(lambda: defaultdict(list))
    for r in matched:
        e = {a:abs(r['predictions'][a]-r['reference_hr']) for a in ARMS}
        ds = dict(pixel_at_native_ROI=e['P1R0']-e['P0R0'],
                  ROI_at_H264_pixels=e['P1R1']-e['P1R0'],
                  ROI_at_native_pixels=e['P0R1']-e['P0R0'],
                  interaction=e['P1R1']-e['P1R0']-e['P0R1']+e['P0R0'],
                  diagonal_total=e['P1R1']-e['P0R0'])
        for key, v in ds.items():
            effects[key][r['subject']].append(v)
    effect_summary = {}
    for key, bysubject in effects.items():
        values = np.array([np.mean(bysubject[s]) for s in sorted(bysubject)])
        rng = np.random.default_rng(20261030)
        boot = rng.choice(values, size=(1000,len(values)), replace=True).mean(axis=1)
        effect_summary[key] = dict(subjects=len(values), mean_bpm=float(values.mean()),
                                  ci95_bpm=np.quantile(boot,[.025,.975]).tolist())
    coverage = []
    for step in ['all','before','after']:
        selected = [r for r in rows if step=='all' or r['step']==step]
        coverage.append(dict(condition=step, windows=len(selected),
            reference_valid=sum(r['reference_hr'] is not None for r in selected),
            predictions_available={a:sum(r['predictions'][a] is not None for r in selected) for a in ARMS},
            paired_valid=sum(r['reference_hr'] is not None and all(r['predictions'][a] is not None for a in ARMS) for r in selected)))
    return dict(expected_records=expected, processed=len(records),
                processing_errors=sum('error' in r for r in records),
                known_candidate_windows=sum(r.get('candidate_windows',0) for r in records),
                analyzed_windows=len(rows), reference_windows=len(reference), matched_windows=len(matched),
                coverage=coverage, matched={a:error_metrics(matched,a) for a in ARMS},
                per_cell_available={a:error_metrics([r for r in reference if r['predictions'][a] is not None],a) for a in ARMS},
                paired_effects=effect_summary,
                reference_failures=dict(Counter(r['reference_failure'] for r in rows if r['reference_failure'])),
                source_labels_unchanged=True, offset_corrected=False)
