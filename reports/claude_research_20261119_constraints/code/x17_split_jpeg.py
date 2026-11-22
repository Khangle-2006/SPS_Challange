"""x17 - C5 split stream WITH the inputs of the pretrained experts of reader_v4 (21 Nov). The device keeps every k-th frame (k = ceil(fps/10))
and sends, per kept frame: time stamp + valid flag (9 B), 6x6 block means of the masked 36x36 oval crop (float32, 432 B; classical branch,
exact), the 72x72 unmasked face-box crop as JPEG (quality Q; experts), and - for the public estimator only - the 36x36 masked crop
(uint8, 3888 B; team CNN). Writes the JPEG round-tripped, decimated box crops (what the server decodes) and the measured stream sizes.
  BTC (218, out/x10)        -> out/x15/c5_split10_jpg<Q>/<key>.npz   (scored by y14 with reader_v4)
  UBFC2 + MCD (154, x10)    -> out/x16/c5_split10_jpg<Q>/<key>.npz   (scored by y13)
  bit rates                 -> logs/x17_split_jpeg_Q<Q>.json  (kbps of both stream variants per recording: mean and max over 1 s)
Usage: nice -n 19 $AV code/x17_split_jpeg.py <workers> <Q>"""
import os, sys, json, glob
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
def run(a):
    f, Q = a; key = Path(f).stem; out = W / ('out/x15' if key.startswith('BTC') else 'out/x16') / f'c5_split10_jpg{Q}' / (key + '.npz')
    with np.load(f) as z: t, valid, crops = z['t'].astype(float), z['valid'].astype(bool), z['crop72']
    fps = (len(t) - 1) / (t[-1] - t[0]); k = max(1, int(np.ceil(fps / 10 - 1e-3))); t, valid, crops = t[::k], valid[::k], crops[::k]
    dec = np.zeros_like(crops); nb = np.zeros(len(t), int)
    for i in range(len(t)):
        if not valid[i]: continue
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(crops[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, Q]); nb[i] = len(buf)
        dec[i] = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix('.tmp.npz'); np.savez_compressed(tmp, crop72=dec, t=t, valid=valid); tmp.replace(out)
    fr = (len(t) - 1) / (t[-1] - t[0]); base = 9 + 432; sec = np.floor(t - t[0]).astype(int)
    def kbps(extra):
        per = base + nb + extra; s = np.bincount(sec, weights=per) * 8 / 1000; return float(per.mean() * fr * 8 / 1000), float(s[:-1].max() if len(s) > 1 else s.max())
    return key, dict(fps_sent=round(fr, 3), k=k, jpeg_bytes_mean=float(nb[valid].mean()) if valid.any() else 0., btc_stream_kbps=kbps(0), public_stream_kbps=kbps(3888))
if __name__ == '__main__':
    from multiprocessing import Pool
    n, Q = int(sys.argv[1]), int(sys.argv[2]); fs = sorted(glob.glob(str(W / 'out/x10/*.npz'))); fs = [f for f in fs if not f.endswith('.tmp.npz')]
    with Pool(n) as p: R = dict(p.map(run, [(f, Q) for f in fs], chunksize=2))
    (W / f'logs/x17_split_jpeg_Q{Q}.json').write_text(json.dumps(R, indent=0))
    for grp, var in (('BTC', 'btc_stream_kbps'), ('UBFC2', 'public_stream_kbps'), ('MCD', 'public_stream_kbps')):
        v = np.array([r[var] for k, r in R.items() if k.startswith(grp)]); print(grp, len(v), var, 'mean %.0f, max of 1-s maxima %.0f kbps' % (v[:, 0].mean(), v[:, 1].max()), flush=True)
