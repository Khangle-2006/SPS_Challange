"""Postprocessing criteria fixed on 2026-10-30; does not change the lag worker."""
READING_CRITERIA_SHA256 = 'f53a01127ef58cccc8283a6e0f76b865a3a2f29f46f69a27a0db9de8a94ade55'


def endpoint_drift(record):
    windows = sorted(record.get('windows', []), key=lambda w: w['index'])
    out = dict(endpoint_evaluable=False, endpoint_delta_frames=None,
               endpoint_drift_flag=None, endpoint_reason=None,
               endpoint_weak=None, endpoint_ambiguous=None, endpoint_boundary=None)
    if len(windows) < 2 or windows[0]['index'] != 0:
        out['endpoint_reason'] = 'fewer_than_two_complete_windows_or_missing_first'
        return out
    first, last = windows[0].get('result'), windows[-1].get('result')
    if first is None or last is None:
        out['endpoint_reason'] = 'first_or_last_complete_window_invalid'
        return out
    delta = last['lag_frames'] - first['lag_frames']
    out.update(endpoint_evaluable=True, endpoint_delta_frames=delta,
               endpoint_drift_flag=abs(delta) > 3,
               endpoint_weak=bool(first['weak_peak'] or last['weak_peak']),
               endpoint_ambiguous=bool(first['ambiguous_peak'] or last['ambiguous_peak']),
               endpoint_boundary=bool(first['search_boundary'] or last['search_boundary']))
    return out


def drift_counts(records, expected):
    if len(records) > expected:
        raise ValueError('More records than the fixed cohort denominator')
    ds = [endpoint_drift(r) for r in records]
    evaluable = sum(d['endpoint_evaluable'] for d in ds)
    flagged = sum(d['endpoint_drift_flag'] is True for d in ds)
    unknown = expected - evaluable
    lower = flagged / expected if expected else None
    upper = (flagged + unknown) / expected if expected else None
    decision = ('below_5_percent' if upper is not None and upper < .05 else
                'at_or_above_5_percent' if lower is not None and lower >= .05 else
                'unresolved_missing_endpoints')
    return dict(expected_records=expected, processed_records=len(records),
                endpoint_evaluable=evaluable, flagged=flagged,
                unassessable_processed=len(records)-evaluable,
                unprocessed=expected-len(records), unknown_total=unknown,
                flagged_fraction_all=lower,
                flagged_fraction_evaluable=flagged/evaluable if evaluable else None,
                possible_flagged_fraction_bounds=[lower, upper], decision=decision)
