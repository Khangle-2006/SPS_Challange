"""Rate-generic versions of the team's operators (same maths, sampling rate as a parameter)."""
import math
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import factorized
GRID = np.arange(42., 180.01, .5)
_tr = {}
def trend_fs(n, lam):
    k = (n, round(lam, 4))
    if k not in _tr:
        d = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format='csc')
        _tr[k] = factorized(sparse.eye(n, format='csc') + lam ** 2 * (d.T @ d))
    return _tr[k]
def pos_fs(rgb, fs):
    L = math.ceil(1.6 * fs); out = np.zeros(len(rgb))
    for e in range(L, len(rgb) + 1):
        x = rgb[e - L:e]; n = x / np.maximum(x.mean(0), 1e-8)
        s = np.array([[0, 1, -1], [-2, 1, 1]]) @ n.T
        h = s[0] + np.std(s[0]) / max(np.std(s[1]), 1e-8) * s[1]
        out[e - L:e] += h - h.mean()
    return out
def chrom_fs(rgb, fs):
    n = len(rgb); w = math.ceil(1.6 * fs); w += w % 2
    b, a = signal.butter(3, [.7, min(3., .45 * fs)], fs=fs, btype='bandpass')
    pad = min(3 * max(len(a), len(b)), w - 1)
    st = list(range(0, n - w + 1, w // 2))
    if st[-1] != n - w: st.append(n - w)
    out = np.zeros(n)
    for s in st:
        v = rgb[s:s + w]; r, g, bl = (v / v.mean(0)).T
        x = signal.filtfilt(b, a, 3 * r - 2 * g, padlen=pad); y = signal.filtfilt(b, a, 1.5 * r + g - 1.5 * bl, padlen=pad)
        out[s:s + w] += (x - np.std(x) / max(np.std(y), 1e-12) * y) * signal.windows.hann(w)
    return out
def spec_fs(w, fs, nfft=8192):
    lam = 100. * (fs / 30.) ** 2          # keeps the detrender cut-off (~0.48 Hz) at the same frequency
    sos = signal.butter(1, [.7, min(3., .45 * fs)], btype='bandpass', fs=fs, output='sos')
    y = signal.sosfiltfilt(sos, w - trend_fs(len(w), lam)(w))
    f, p = signal.periodogram(y, fs=fs, window='hann', nfft=nfft, detrend=False)
    return np.interp(GRID, 60 * f, p, left=0., right=0.)
