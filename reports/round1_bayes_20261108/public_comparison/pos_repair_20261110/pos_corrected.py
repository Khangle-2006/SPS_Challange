"""Versioned canonical POS path on the established oval-36 frontend.

No dataset branch, prior, fitted threshold or label input. This restores the
previously validated public baseline, rather than modifying frozen experiments.
"""
import math
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

FRONTEND = 't8_frozen_oval36_inter_area'

def mean_oval_crops(crops):
    x = np.asarray(crops)
    if x.ndim != 4 or x.shape[1:] != (36, 36, 3):
        raise ValueError('Expected frozen RGB oval crops [T,36,36,3]')
    return x.astype(np.float64).mean(axis=(1, 2))

def pos_wave(rgb, fs=30.):
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite RGB trace')
    length = math.ceil(1.6 * fs)
    if len(rgb) < length:
        raise ValueError('Short POS trace')
    if np.max(np.std(rgb, axis=0)) < 1e-10:
        return np.zeros(len(rgb))
    out = np.zeros(len(rgb))
    for end in range(length, len(rgb) + 1):
        x = rgb[end-length:end]
        normalized = x / np.maximum(x.mean(axis=0), 1e-8)
        s = np.array([[0, 1, -1], [-2, 1, 1]]) @ normalized.T
        h = s[0] + np.std(s[0]) / max(np.std(s[1]), 1e-8) * s[1]
        out[end-length:end] += h - h.mean()
    return out

def decode_wave(wave):
    w = np.asarray(wave, float)
    if w.shape != (300,) or not np.isfinite(w).all() or w.std() < 1e-10:
        return dict(hr=None, snr_db=None, reason='flat_or_invalid_wave')
    n = len(w)
    d = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n), format='csc')
    trend = spsolve(sparse.eye(n, format='csc') + 100**2 * (d.T @ d), w)
    y = signal.sosfiltfilt(signal.butter(1, [.7, 3.], btype='bandpass', fs=30, output='sos'), w-trend)
    f, power = signal.periodogram(y, fs=30, window='hann', nfft=4096, detrend=False)
    selected = (f >= .7) & (f <= 3.)
    f, power = f[selected], power[selected]
    if not np.isfinite(power).all() or power.sum() <= 1e-30:
        return dict(hr=None, snr_db=None, reason='invalid_spectrum')
    hz = f[np.argmax(power)]
    near = np.abs(f-hz) <= .1 + 1e-12
    ratio = power[near].sum() / max(power[~near].sum(), np.finfo(float).tiny)
    return dict(hr=float(60*hz), snr_db=float(10*np.log10(ratio)), reason=None)

def estimate_window(t, rgb, valid, start, *, frontend):
    if frontend != FRONTEND:
        raise ValueError('Canonical POS requires the frozen oval frontend; small skin ROIs are not interchangeable')
    t, rgb, valid = np.asarray(t, float), np.asarray(rgb, float), np.asarray(valid, bool)
    if t.ndim != 1 or rgb.shape != (len(t), 3) or valid.shape != t.shape or len(t) < 2:
        raise ValueError('Invalid trace shapes')
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Canonical clock must be finite, unique and increasing')
    end = start+10
    left = np.searchsorted(t, start, side='right')-1
    right = np.searchsorted(t, end, side='left')
    if left < 0 or right >= len(t):
        raise ValueError('No extrapolation')
    sl = slice(left, right+1)
    if not valid[sl].all() or not np.isfinite(rgb[sl]).all() or np.any(np.diff(t[sl]) > .2+1e-12):
        raise ValueError('Invalid support or timestamp gap')
    q = start + np.arange(300)/30
    x = np.stack([np.interp(q, t, rgb[:, c]) for c in range(3)], axis=1)
    return decode_wave(pos_wave(x))
