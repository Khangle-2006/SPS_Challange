"""Versioned raw-cache metadata adapter, before frozen interpolation/differencing."""
from pathlib import Path
import numpy as np
from ppg_polarity import adapt_cached_meta
from t8_t1_loader import clip_arrays


def load_canonical_clip(folder, start, dataset):
    folder = Path(folder)
    with np.load(folder/'meta.npz') as stored:
        meta, receipt = adapt_cached_meta(stored, dataset, source_convention='raw_device')
        x, dy, waveform = clip_arrays(np.load(folder/'rgb.npy', mmap_mode='r'),
            meta['t'], meta['valid'], meta['ppg'], meta['ppg_t'], meta['ppg_valid'], start)
    return x, dy, waveform, receipt
