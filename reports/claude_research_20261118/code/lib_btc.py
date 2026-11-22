"""Shared helpers for the BTC dev analyses in this folder: loader for the numeric export, the team's POS/CHROM and frozen readout
operator (re-implemented, checked against the saved readouts), person-first metrics with paired bootstrap. Dev videos only."""
import csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import factorized
ROOT = Path('/mnt/samsung_t5/khangle/facevitals'); W = ROOT / 'reports/claude_research_20261118'; OUT = W / 'out'
NUM = ROOT / 'reports/w2_btc_20261116/numeric_dev'
GRID = np.arange(42., 180.01, .5)
def videos(): return json.loads((NUM / 'VIDEOS.json').read_text())
def meta(): return {r['recording']: r for r in csv.DictReader((OUT / 'dev_video_meta.csv').open())}
def load_blocks(rec):
    with np.load(NUM / 'blocks' / (rec + '.npz'), allow_pickle=False) as z:
        return dict(t=z['t'].copy(), valid=z['valid'].copy(), rgb_mean=z['rgb_mean'].copy(), boxes=z['boxes'].copy(), fill=z['fill'].astype(float) / 255.)
def supported(t, valid, start, end, gap=.2):
    t = np.asarray(t, float); valid = np.asarray(valid, bool)
    if len(t) < 2 or np.any(np.diff(t) <= 0): return False
    if start < t[0] - 1e-9 or end > t[-1] + 1e-9: return False
    for i in np.flatnonzero(~valid):
        a, b = t[max(0, i - 1)], t[min(len(t) - 1, i + 1)]
        if a < end - 1e-10 and b > start + 1e-10: return False
    for i in np.flatnonzero(np.diff(t) > gap + 1e-12):
        if t[i] < end - 1e-10 and t[i + 1] > start + 1e-10: return False
    return not np.any((~valid) & (t >= start - 1e-10) & (t <= end + 1e-10))
def pos_team(rgb, fs=30.):
    rgb = np.asarray(rgb, float); L = math.ceil(1.6 * fs); out = np.zeros(len(rgb))
    for e in range(L, len(rgb) + 1):
        x = rgb[e - L:e]; n = x / np.maximum(x.mean(0), 1e-8); s = np.array([[0, 1, -1], [-2, 1, 1]]) @ n.T
        h = s[0] + np.std(s[0]) / max(np.std(s[1]), 1e-8) * s[1]; out[e - L:e] += h - h.mean()
    return out
def chrom_team(rgb, fs=30.):
    rgb = np.asarray(rgb, float); n = len(rgb); w = math.ceil(1.6 * fs); w += w % 2
    b, a = signal.butter(3, [.7, 3], fs=fs, btype='bandpass'); starts = list(range(0, n - w + 1, w // 2))
    if starts[-1] != n - w: starts.append(n - w)
    out = np.zeros(n)
    for st in starts:
        v = rgb[st:st + w]; r, g, bl = (v / v.mean(0)).T
        x = signal.filtfilt(b, a, 3 * r - 2 * g); y = signal.filtfilt(b, a, 1.5 * r + g - 1.5 * bl)
        out[st:st + w] += (x - np.std(x) / np.std(y) * y) * signal.windows.hann(w)
    return out
_T = {}
def trend(n, lam=100.):
    if (n, lam) not in _T:
        d = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format='csc')
        _T[(n, lam)] = factorized(sparse.eye(n, format='csc') + lam ** 2 * (d.T @ d))
    return _T[(n, lam)]
SOS1 = signal.butter(1, [.7, 3.], btype='bandpass', fs=30, output='sos')
def window_spectrum(w):
    """frozen readout operator, returned on GRID (42-180 bpm, 0.5 bpm) and normalised to unit sum."""
    w = np.asarray(w, float); y = signal.sosfiltfilt(SOS1, w - trend(len(w))(w)); f, p = signal.periodogram(y, fs=30, window='hann', nfft=4096, detrend=False)
    p = np.interp(GRID, 60 * f, p); return p / p.sum()
def avg_spectra(t, valid, rgb, hop=5.):
    """window-averaged CHROM and POS spectra of one video (10 s windows every `hop` s on a 30 Hz grid); None if no window is supported."""
    C, P = [], []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), hop):
        if not supported(t, valid, s, s + 10): continue
        q = s + np.arange(300) / 30.; x = np.stack([np.interp(q, t, rgb[:, c]) for c in range(3)], 1)
        if not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        wc, wp = chrom_team(x), pos_team(x)
        if not (np.isfinite(wc).all() and np.isfinite(wp).all()) or wc.std() < 1e-12 or wp.std() < 1e-12: continue
        C.append(window_spectrum(wc)); P.append(window_spectrum(wp))
    return (np.mean(C, 0), np.mean(P, 0), len(C)) if C else None
def zlog(s):
    l = np.log(np.asarray(s, float) + 1e-30); return (l - l.mean(-1, keepdims=True)) / l.std(-1, keepdims=True)
def yuv_to_rgb(yuv, m, deep):
    """mean Y,U,V (any leading shape, last axis 3) -> R'G'B' on a 0-255 scale, using the matrix and range named in the container."""
    yuv = np.asarray(yuv, float) / (4. if deep else 1.)
    if (m.get('color_range') or 'tv') == 'pc': y = yuv[..., 0] / 255.; cb = (yuv[..., 1] - 128) / 255.; cr = (yuv[..., 2] - 128) / 255.
    else: y = (yuv[..., 0] - 16) / 219.; cb = (yuv[..., 1] - 128) / 224.; cr = (yuv[..., 2] - 128) / 224.
    cs = m.get('color_space') or ''
    if cs in ('smpte170m', 'bt470bg'): k = (1.402, .344136, .714136, 1.772)
    elif cs.startswith('bt2020'): k = (1.4746, .16455, .57135, 1.8814)
    else: k = (1.5748, .1873, .4681, 1.8556)
    return 255. * np.stack([y + k[0] * cr, y - k[1] * cb - k[2] * cr, y + k[3] * cb], -1)
def person_weights(per):
    c = Counter(per); return np.array([1. / c[p] for p in per])
def summarize(pred, truth, per, base=None, n_boot=10000):
    """person-first MAE, bias, Spearman of person means, bootstrap CI over people (seed 0); paired difference to `base` if given."""
    from scipy.stats import spearmanr
    d = defaultdict(list)
    for i, p in enumerate(per):
        row = [abs(pred[i] - truth[i]), pred[i] - truth[i], pred[i], truth[i]]
        if base is not None: row.append(abs(base[i] - truth[i]))
        d[p].append(row)
    v = np.array([np.mean(d[p], 0) for p in sorted(d)]); ix = np.random.default_rng(0).integers(0, len(v), (n_boot, len(v)))
    out = dict(MAE=float(v[:, 0].mean()), bias=float(v[:, 1].mean()), rho=float(spearmanr(v[:, 2], v[:, 3]).statistic) if np.ptp(v[:, 2]) > 0 else float('nan'),
               CI=[float(x) for x in np.quantile(v[ix, 0].mean(1), [.025, .975])], RMSE=float(np.sqrt(np.mean((v[:, 2] - v[:, 3]) ** 2))),
               r=float(np.corrcoef(v[:, 2], v[:, 3])[0, 1]) if np.ptp(v[:, 2]) > 0 else float('nan'), people=len(v))
    if base is not None:
        dd = v[:, 0] - v[:, 4]; out['delta'] = float(dd.mean()); out['dCI'] = [float(x) for x in np.quantile(dd[ix].mean(1), [.025, .975])]
    return out
