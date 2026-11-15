"""CPU-only post-adapter mean-cycle contract; never selects or changes a sign."""
import numpy as np
from scipy import signal
from ppg_polarity import CONVENTION, POLICY_SHA256, dataset_name, SIGNS


def extract_cycles(ppg, timestamps, valid=None):
    """Orient PPG BEFORE this function. Do not negate previously aligned cycles.

    Invalid samples and >.2-second clock gaps split support. Each supported span is
    resampled to30Hz, detrended, and low-passed8Hz. Individual noisy beats are not
    treated as convention assertions; aggregate sufficiency is checked separately.
    """
    y, t = np.asarray(ppg, float), np.asarray(timestamps, float)
    if y.ndim != 1 or t.shape != y.shape or len(y) < 2:
        raise ValueError('Expected equal-length PPG/time vectors')
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError('Invalid PPG timestamps')
    good = np.isfinite(y)
    if valid is not None:
        v = np.asarray(valid, bool)
        if v.shape != y.shape:
            raise ValueError('Invalid PPG validity shape')
        good &= v
    # Match adapter duplicate-time averaging, conservatively invalidating a group.
    u, inv, n = np.unique(t, return_inverse=True, return_counts=True)
    merged = np.zeros(len(u)); np.add.at(merged, inv, np.where(good, y, 0))
    merged /= n
    merged_good = np.ones(len(u), bool); np.logical_and.at(merged_good, inv, good)
    t, y, good = u, merged, merged_good
    indices = np.flatnonzero(good)
    if not len(indices):
        return np.empty((0, 101))
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(t[indices]) > .2)) + 1
    cycles = []
    for ix in np.split(indices, cuts):
        if len(ix) < 2 or t[ix[-1]] - t[ix[0]] < 2:
            continue
        q = t[ix[0]] + np.arange(int(np.floor((t[ix[-1]]-t[ix[0]])*30))+1)/30
        sampled = np.interp(q, t[ix], y[ix])
        smooth = signal.sosfiltfilt(signal.butter(2, 8, fs=30, output='sos'),
                                   signal.detrend(sampled))
        if smooth.std() < 1e-10:
            continue
        troughs = signal.find_peaks(-smooth, distance=10, prominence=.25*smooth.std())[0]
        for left, right in zip(troughs[:-1], troughs[1:]):
            if 10 <= right-left <= 43:
                cycle = np.interp(np.linspace(left, right, 101), np.arange(len(smooth)), smooth)
                if cycle.std() > 1e-10:
                    cycles.append((cycle-cycle.mean())/cycle.std())
    return np.asarray(cycles).reshape(-1, 101)


def assess_dataset(records, dataset, *, min_recordings=3, min_cycles_per_recording=3,
                   min_total_cycles=12, phase_margin=.04):
    """records: post-adapter dicts with ppg/ppg_t/ppg_valid/ppg_polarity.

    Recording means have equal weight. Returns pass/fail/unresolved; never raises
    merely because an individual waveform is noisy, and never auto-flips labels.
    A caller must explicitly require status='pass' to admit a dataset/release.
    """
    dataset = dataset_name(dataset)
    report = dict(dataset=dataset, convention=CONVENTION, policy_sha256=POLICY_SHA256,
                  status='unresolved', reason=None, recordings_seen=0,
                  adequate_recordings=0, cycles=0, rise_phase=None, fall_phase=None,
                  min_recordings=min_recordings, min_cycles_per_recording=min_cycles_per_recording,
                  min_total_cycles=min_total_cycles, phase_margin=phase_margin)
    means = []
    for record in records:
        report['recordings_seen'] += 1
        receipt = record.get('ppg_polarity', {})
        expected = {'status': 'canonical', 'convention': CONVENTION, 'dataset': dataset,
                    'policy_sha256': POLICY_SHA256, 'raw_to_canonical_sign': SIGNS.get(dataset)}
        if dataset not in SIGNS or any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('Post-adapter canonical provenance required for morphology')
        c = extract_cycles(record['ppg'], record['ppg_t'], record.get('ppg_valid'))
        if len(c) >= min_cycles_per_recording:
            means.append(c.mean(axis=0)); report['cycles'] += len(c)
    report['adequate_recordings'] = len(means)
    if len(means) < min_recordings or report['cycles'] < min_total_cycles:
        report['reason'] = 'insufficient_valid_average_cycles'
        return report
    average = np.mean(means, axis=0)
    peak = int(np.argmax(average))
    rise, fall = peak/100., 1-peak/100.
    report.update(rise_phase=rise, fall_phase=fall, mean_cycle=average.tolist())
    if peak <= 5 or peak >= 95 or abs(fall-rise) <= phase_margin:
        report['reason'] = 'ambiguous_average_morphology'
    elif rise < fall:
        report.update(status='pass', reason='average_rise_shorter_than_fall')
    else:
        report.update(status='fail', reason='average_rise_longer_than_fall')
    return report
