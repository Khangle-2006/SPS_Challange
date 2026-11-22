"""u17 (22 Nov, Claude; to be RUN BY CODEX - open-rppg is third-party code): save the raw BVP WAVEFORM (not only the spectrum) of the
open-rppg models on the BTC dev face-box crops (x10), so that the inter-beat intervals can be measured (u18: is the noise of the hard camera
originals an irregular pulse?). Same model call as y05 (process_faces_tensor + bvp(raw=True)). CPU, no training, no holdout.
Usage: venv_orppg/bin/python code/u17_openrppg_bvp.py <model> <keys-file>   -> out/u17/<model>/<key>.npz (bvp, ts)"""
import os, sys, time
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '2')
os.environ.setdefault('KERAS_BACKEND', 'jax'); os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
if __name__ == '__main__':
    import rppg
    name, kf = sys.argv[1], sys.argv[2]; keys = (W / kf).read_text().strip().split(','); m = rppg.Model(name); out = W / 'out/u17' / name; out.mkdir(parents=True, exist_ok=True); secs = []
    for i, key in enumerate(keys):
        dst = out / (key + '.npz')
        if dst.exists(): continue
        p = W / 'out/x10' / (key + '.npz')
        if not p.exists(): continue
        with np.load(p) as z: t, crops = z['t'], z['crop72']
        fps = (len(t) - 1) / (t[-1] - t[0]); t0 = time.time(); m.process_faces_tensor(np.ascontiguousarray(crops), fps=float(fps)); bvp, ts = m.bvp(raw=True); secs.append(time.time() - t0)
        np.savez_compressed(dst, bvp=np.asarray(bvp, np.float32), ts=np.asarray(ts, float), t=np.asarray(t, float))
        if i % 20 == 0: print(name, i, len(keys), 'mean s/video %.1f' % np.mean(secs), flush=True)
    print(name, 'done; files', len(list(out.glob('*.npz'))), flush=True)
