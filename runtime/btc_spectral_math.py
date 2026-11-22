"""Unchanged numerical functions from frozen lib_btc.py; no dataset paths."""
import math
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import factorized
GRID=np.arange(42.,180.01,.5)
_T={}
SOS1=signal.butter(1,[.7,3.],btype="bandpass",fs=30,output="sos")

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

def trend(n, lam=100.):
    if (n, lam) not in _T:
        d = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format='csc')
        _T[(n, lam)] = factorized(sparse.eye(n, format='csc') + lam ** 2 * (d.T @ d))
    return _T[(n, lam)]

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
