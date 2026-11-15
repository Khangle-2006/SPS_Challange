"""Frozen T2-v1 spectral HR decoder; no fitted/label-dependent parameters.

Window-local filtering is noncausal within the observed window. ``causal``
means no future *window*, and outputs arrive at the end of a complete window.
Use decode_sequence for genuine offline forward-backward smoothing. A lone
decode(...,mode='offline') is a one-window smoother, without invented context.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class DecoderConfig:
    band_low: float = .7
    regional: bool = True
    harmonic: bool = True
    hmm: bool = True
    abstention: bool = True
    beta: float = .5
    kappa: float = 10.
    tau: float = 2.
    transition_scale: float = 10.


def ablation_config(stage):
    if stage not in ['legacy', 'a', 'b', 'c', 'd', 'e']:
        raise ValueError(stage)
    n = ['legacy', 'a', 'b', 'c', 'd', 'e'].index(stage)
    return DecoderConfig(.75 if n == 0 else .7, n >= 2, n >= 3,
                         n >= 4, n >= 5)


def fold_frequency(f, fs):
    """Alias arbitrary frequency onto [0, fs/2], including fs=10 harmonic."""
    return np.abs((np.asarray(f) + fs / 2) % fs - fs / 2)


def _spectra(waves, fs, mask, cfg):
    if waves.ndim != 2 or mask.shape != waves.shape:
        raise ValueError('waveforms and valid_mask must be [R,N]')
    if not 0 < cfg.band_low < 3 < fs/2:
        raise ValueError('HR band must be strictly inside Nyquist')
    n = waves.shape[1]
    if n < 5*fs or n > 4096:
        raise ValueError('T2 windows need 5s minimum and <=4096 samples')
    d = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
                     [0, 1, 2], shape=(n-2, n), format='csc')
    a = sparse.eye(n, format='csc') + 100**2*(d.T@d)
    sos = signal.butter(1, [cfg.band_low, 3], btype='bandpass', fs=fs, output='sos')
    ps, valid_regions = [], []
    f = np.fft.rfftfreq(4096, 1/fs)
    band = (f >= cfg.band_low) & (f <= 3)
    for j, w in enumerate(waves):
        if not mask[j].all() or not np.isfinite(w).all() or np.std(w) < 1e-10:
            continue
        x = signal.sosfiltfilt(sos, w-spsolve(a, w))
        _, p = signal.periodogram(x, fs=fs, window='hann', nfft=4096, detrend=False)
        total = p[band].sum()
        if np.isfinite(p).all() and total > 1e-30:
            ps.append(p / total)
            valid_regions.append(j)
    if not ps:
        return None
    ps = np.asarray(ps)
    weights = ps[:, band].max(axis=1) if cfg.regional else np.ones(len(ps))
    weights /= weights.sum()
    full = weights @ ps
    fundamental = full[band]
    likelihood = fundamental.copy()
    if cfg.harmonic:
        likelihood += cfg.beta*np.interp(fold_frequency(2*f[band], fs), f, full)
    return f[band]*60, fundamental, likelihood, weights, valid_regions


def _transition(cfg, dt):
    grid = np.arange(42., 181.)
    scale = cfg.transition_scale * max(float(dt), 1e-6)/10
    t = np.exp(-np.abs(grid[:, None]-grid[None, :])/scale)
    return t/t.sum(axis=1, keepdims=True)


def _result(spectra, state, cfg, dt):
    grid = np.arange(42., 181.)
    if spectra is None:
        return dict(hr_bpm=float('nan'), confidence=0., abstain=True,
                    top2_peaks_bpm=[], psd_fused=np.array([]), state=state,
                    invalid=True, snr=0., posterior=np.zeros(139))
    bpm, p, like, weights, valid_regions = spectra
    l = np.interp(grid, bpm, like, left=0, right=0)
    emission = np.exp(cfg.kappa*(l/max(l.max(), 1e-30)-1))
    emission /= emission.sum()
    streak = 0
    reset = False
    if cfg.hmm:
        if state is None:
            posterior = emission.copy()
        else:
            previous = np.asarray(state['posterior'], float)
            if previous.shape != (139,) or not np.isfinite(previous).all() or previous.sum() <= 0:
                raise ValueError('invalid HMM state')
            old = int(np.argmax(previous)); new = int(np.argmax(l))
            challenge = abs(new-old) > 10 and l[new] > l[old]
            streak = state.get('challenger_streak', 0)+1 if challenge else 0
            reset = streak >= 3
            prior = np.ones(139)/139 if reset else previous@_transition(cfg, dt)
            posterior = prior*emission
            posterior /= posterior.sum()
            if reset:
                streak = 0
        hr = float(grid[np.argmax(posterior)])
    else:
        posterior = emission
        hr = float(bpm[np.argmax(like)])
    noise = p[np.abs(bpm-hr) > 6]
    snr = float(np.interp(hr, bpm, p)/max(float(np.median(noise)), 1e-12))
    peaks = signal.find_peaks(like)[0].tolist()
    if not peaks:
        peaks = [int(np.argmax(like))]
    peaks.sort(key=lambda i: like[i], reverse=True)
    outstate = dict(posterior=posterior.tolist(), challenger_streak=streak)
    return dict(hr_bpm=hr, confidence=snr, abstain=bool(cfg.abstention and snr < cfg.tau),
                top2_peaks_bpm=[float(bpm[i]) for i in peaks[:2]], psd_fused=p,
                frequencies_bpm=bpm, likelihood=like, state=outstate,
                posterior=posterior, emission=emission, snr=snr, invalid=False,
                region_weights=weights, valid_regions=valid_regions, reset=reset)


def decode(waveforms, fs, valid_mask=None, state=None, mode='causal', *,
           config=None, step_seconds=10.):
    if mode not in ('causal', 'offline'):
        raise ValueError('use decode_sequence for fixedlag1')
    if mode == 'offline' and state is not None:
        raise ValueError('offline sequence context must use decode_sequence')
    cfg = config or DecoderConfig()
    waves = np.asarray(waveforms, dtype=float)
    mask = np.ones_like(waves, dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
    out = _result(_spectra(waves, float(fs), mask, cfg), state, cfg, step_seconds)
    out['mode'] = mode
    out['lookahead_windows'] = 0
    return out


def decode_sequence(windows, fs, *, masks=None, config=None, mode='causal', starts=None):
    """Windows [W,R,N] of ONE recording, chronologically ordered.

    offline smooths each valid consecutive segment; resets break smoothing.
    fixedlag1 outputs the preceding window with one-window latency; final
    output is forward-only, explicitly tagged lookahead_windows=0.
    """
    if mode not in ('causal', 'offline', 'fixedlag1'):
        raise ValueError(mode)
    cfg = config or DecoderConfig()
    starts = np.arange(len(windows))*10. if starts is None else np.asarray(starts)
    if len(starts) != len(windows) or np.any(np.diff(starts) <= 0):
        raise ValueError('starts must strictly increase')
    results, state = [], None
    for i, w in enumerate(windows):
        dt = 10 if i == 0 else starts[i]-starts[i-1]
        r = decode(w, fs, None if masks is None else masks[i], state,
                   config=cfg, step_seconds=dt)
        results.append(r)
        state = None if r['invalid'] else r['state']
    if mode == 'causal' or not cfg.hmm:
        return results
    beta = np.ones(139)
    for i in range(len(results)-1, -1, -1):
        r = results[i]
        if r['invalid']:
            beta = np.ones(139)
            continue
        future = i+1 < len(results) and not results[i+1]['invalid'] and not results[i+1]['reset']
        if future:
            bnext = beta if mode == 'offline' else np.ones(139)
            beta = _transition(cfg, starts[i+1]-starts[i]) @ (results[i+1]['emission']*bnext)
            beta /= max(beta.sum(), 1e-30)
        else:
            beta = np.ones(139)
        post = r['posterior']*beta; post /= post.sum()
        hr = float(42+np.argmax(post))
        noise = r['psd_fused'][np.abs(r['frequencies_bpm']-hr)>6]
        snr = float(np.interp(hr,r['frequencies_bpm'],r['psd_fused'])/max(float(np.median(noise)),1e-12))
        r.update(hr_bpm=hr, posterior=post, confidence=snr, snr=snr,
                 abstain=bool(cfg.abstention and snr<cfg.tau), mode=mode,
                 lookahead_windows=(len(results)-1-i if mode=='offline' else int(future)))
    return results
