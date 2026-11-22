"""f03b - the three open-rppg experts' BVP on the same window (venv_orppg). Writes out/f01/f03_waves_<model>.npz."""
import os, sys
os.environ.setdefault('KERAS_BACKEND', 'jax')
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '2')
import numpy as np, rppg
from pathlib import Path
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); KEY = sys.argv[1]; s = float(sys.argv[2])
with np.load(W / 'out/x10' / (KEY + '.npz')) as z: t, crops = z['t'], z['crop72']
fps = (len(t) - 1) / (t[-1] - t[0]); lo = np.searchsorted(t, s - 20); hi = np.searchsorted(t, s + 30)   # 20 s margin either side
for name in ('FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure'):
    m = rppg.Model(name); m.process_faces_tensor(np.ascontiguousarray(crops[lo:hi]), fps=float(fps)); bvp, ts = m.bvp(raw=True)
    bvp = np.asarray(bvp, float); ts = np.asarray(ts, float); ts = ts - ts[0] + t[lo]
    np.savez(W / 'out/f01' / f'f03_waves_{name}.npz', wave=np.interp(s + np.arange(300) / 30., ts, bvp)); print(name, 'ok', len(bvp))
