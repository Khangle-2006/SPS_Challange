"""Classical spectra on the frozen shifted BTC window grid."""
import numpy as np
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view

from btc_spectral_math import supported, window_spectrum

B, A = signal.butter(3, [.7, 3], fs=30, btype='bandpass')


def waves(rgb):
    width = 48
    blocks = sliding_window_view(rgb, width, axis=0).transpose(0, 2, 1)
    normalized = blocks / np.maximum(blocks.mean(axis=1, keepdims=True), 1e-8)
    red, green, blue = normalized.transpose(2, 0, 1)
    x, y = green - blue, -2 * red + green + blue
    h = x + (x.std(axis=1) / np.maximum(y.std(axis=1), 1e-8))[:, None] * y
    centered = h - h.mean(axis=1, keepdims=True)
    pos = np.zeros(len(rgb))
    for j in range(width):
        pos[j:j+len(blocks)] += centered[:, j]
    starts = np.unique(np.r_[np.arange(0, len(blocks), 24), len(blocks)-1])
    v = blocks[starts] / blocks[starts].mean(1, keepdims=True)
    r, g, b = v.transpose(2, 0, 1)
    cx = signal.filtfilt(B, A, 3*r-2*g, axis=1)
    cy = signal.filtfilt(B, A, 1.5*r+g-1.5*b, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        parts = (cx - (cx.std(1)/cy.std(1))[:, None]*cy) * signal.windows.hann(width)
    chrom = np.zeros(len(rgb))
    for start, part in zip(starts, parts):
        chrom[start:start+width] += part
    return chrom, pos


def windows(t, valid, rgb, width, hop):
    result = []
    for start in np.arange(0, max(0., t[-1]-width+1e-8), hop):
        if not supported(t, valid, start, start+width):
            continue
        q = start + np.arange(width*30)/30.
        clip = np.column_stack([np.interp(q,t,rgb[:,i]) for i in range(3)])
        if not np.isfinite(clip).all() or clip.mean(0).min() <= 0:
            continue
        c, p = waves(clip)
        if not np.isfinite(c).all() or not np.isfinite(p).all() or min(c.std(), p.std()) < 1e-12:
            continue
        result.append((start, np.stack([window_spectrum(c), window_spectrum(p)])))
    return result
