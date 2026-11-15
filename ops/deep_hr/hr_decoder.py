"""Versioned FFT decoder. Historical reports retain their frozen code."""
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

BAND_HZ = (0.7, 3.0)
LEGACY_BAND_HZ = (0.75, 3.0)


def spectrum(wave, fs=30., band=BAND_HZ):
    w = np.asarray(wave, dtype=float)
    low, high = band
    if not 0 < low < high < fs / 2:
        raise ValueError("Band must be inside Nyquist")
    if w.ndim != 1 or len(w) < 5 * fs or not np.isfinite(w).all() or w.std() < 1e-10:
        raise ValueError("Invalid, short or flat waveform")
    n = len(w)
    d = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)],
                     [0, 1, 2], shape=(n-2, n), format="csc")
    trend = spsolve(sparse.eye(n, format="csc") + 100**2 * (d.T @ d), w)
    filt = signal.sosfiltfilt(signal.butter(1, band, btype="bandpass", fs=fs, output="sos"), w-trend)
    f, p = signal.periodogram(filt, fs=fs, window="hann",
                              nfft=max(4096, 2**int(np.ceil(np.log2(n)))), detrend=False)
    mask = (f >= low) & (f <= high)
    p = p[mask]
    if not np.isfinite(p).all() or p.sum() <= 1e-30:
        raise ValueError("Invalid spectral power")
    return f[mask]*60, p/p.sum()


def estimate_hr(wave, fs=30., band=BAND_HZ):
    f, p = spectrum(wave, fs, band)
    return float(f[np.argmax(p)])
