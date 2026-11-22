"""b01 - cost of the pipeline on ONE CPU core, no GPU (evidence for the on-device and the split mode of C5).
Front end per frame: decode | face mesh | oval crop (36x36).  Back end per 10 s window: POS + CHROM spectra | padding-fixed CNN | readout.
The process (and the decoder it starts) is pinned to one hardware thread.
Usage: $AV code/b01_cpu_cost.py <core> front [frames=300]      $PY code/b01_cpu_cost.py <core> back"""
import os, sys, time, json
core = int(sys.argv[1]); os.sched_setaffinity(0, {core}); MODE = sys.argv[2]
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
if MODE == 'back':
    import x02_infer as X2
    rng = np.random.default_rng(0); raw = (120 + 10 * rng.standard_normal((301, 36, 36, 3))).astype(np.float32); tu = np.arange(330) / 30.; tr = raw.mean((1, 2)); tr = np.concatenate([tr, tr[:29]])
    X2.cnn(0, raw); t0 = time.perf_counter()
    for _ in range(10): wv = X2.cnn(0, raw)
    tcnn = (time.perf_counter() - t0) / 10; t0 = time.perf_counter()
    for _ in range(20): X2.classical(tu, tr, 0.)
    tcl = (time.perf_counter() - t0) / 20; t0 = time.perf_counter()
    for _ in range(20): X2.spectrum(np.cumsum(wv))
    tsp = (time.perf_counter() - t0) / 20
    print(f'pinned to hardware thread {core} | back end per 10 s window: POS + CHROM spectra {1000 * tcl:.1f} ms | CNN (EfficientPhys, 301 frames 36x36) {1000 * tcnn:.0f} ms | readout of the CNN wave {1000 * tsp:.1f} ms -> {100 * (tcl + tcnn + tsp) / 10:.2f}% of real time')
    sys.exit()
import x01_extract as X
import cv2
cv2.setNumThreads(1); NF = int(sys.argv[3]) if len(sys.argv) > 3 else 300
jobs = json.loads((X.R / 'MANIFEST.json').read_text())
def pick(ds, pred=lambda j: True, n=2): return [j for j in jobs if j['dataset'] == ds and pred(j)][:n]
def media(j): return (j.get('metadata') or {}).get('media', {}) if isinstance(j.get('metadata'), dict) else {}
SEL = pick('UBFC2') + pick('MCD') + pick('BTC', lambda j: media(j).get('codec') == 'hevc' and int(media(j).get('width', 0)) * int(media(j).get('height', 0)) == 1920 * 1080 and not X.is_heavy(j)) \
    + pick('BTC', lambda j: int(media(j).get('width', 0)) * int(media(j).get('height', 0)) < 500000, 1)
print('pinned to hardware thread', core, '| frames per video', NF, flush=True)
for j in SEL:
    if j['dataset'] == 'BTC':
        with np.load(Path(j['crop']) / 'meta.npz') as z: ref_t = z['t']
        src = iter(X.BTCVideo(j['source'], ref_t))
    else:
        with np.load(j['meta']) as z: n = len(z['original_t'])
        src = X.frames_of(j, n)
    frames = []; t0 = time.perf_counter()
    for f in src:
        frames.append(np.ascontiguousarray(f))
        if len(frames) >= NF: break
    dec = time.perf_counter() - t0; src.close()
    mesh = X.Mesh(); mesh.step(frames[0]); t0 = time.perf_counter(); pts = [mesh.step(f) for f in frames]; tm = time.perf_counter() - t0; mesh.close()
    t0 = time.perf_counter(); out = [X.fast_crop(f, p) for f, p in zip(frames, pts)]; tc = time.perf_counter() - t0; n = len(frames); tot = 1000 * (dec + tm + tc) / n
    print(f"{j['dataset']:5s} {frames[0].shape[1]}x{frames[0].shape[0]} {media(j).get('codec', ''):5s} frames {n} | per frame: decode {1000 * dec / n:6.2f} ms  face mesh {1000 * tm / n:6.2f} ms  crop {1000 * tc / n:5.2f} ms  total {tot:6.2f} ms"
          f" -> {1000 / tot:5.1f} frames/s on one thread ({1000 / tot / 10:.1f}x the 10 fps of the split mode; {1000 / tot / 30:.2f}x real time at 30 fps) | detected {np.mean([p is not None for p in pts]):.3f}", flush=True)
    del frames
