"""Native-resolution colour traces for the 218 BTC dev videos (holdout never opened; labels not read). Supersedes a04.
One decode pass per video with PyAV; region means are taken on the decoded Y, U, V planes (no colour conversion of whole frames).
Three ways of placing the face box, all derived from the frozen per-frame boxes (reports/w2_btc_20261116/numeric_dev):
  mode 0 'frame'  : the frozen box of that frame (what the current frontend follows)
  mode 1 'smooth' : running median of the box coordinates over 2 s
  mode 2 'static' : one box for the whole video (median of all frames)
Six regions relative to the box (see REG), a 10 x 10 grid of cells for mode 1, and two background corners.
Rotation metadata is applied so that coordinates match the display orientation used by the frozen frontend.
Output: out/native/<video>.npz with yuv[T, mode, region, 3], grid[T, 10, 10, 3], bg[T, 2, 3], t, ptype (0 I, 1 P, 2 B), size, key.
Usage: python a05_native_traces.py [n_workers]"""
import csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import av, numpy as np
from scipy.ndimage import median_filter
ROOT = Path('/mnt/samsung_t5/khangle/facevitals'); W = ROOT / 'reports/claude_research_20261118'; OUTD = W / 'out/native'
REG = {'c60': (.2, .8, .2, .8), 'c40': (.3, .7, .3, .7), 'box': (0., 1., 0., 1.), 'forehead': (.25, .75, .08, .28), 'cheeks': (.15, .85, .45, .72), 'lower': (.2, .8, .5, .9)}
def picture_type(fr):
    p = fr.pict_type; name = getattr(p, 'name', None)
    if name is None:
        try: return {1: 0, 2: 1, 3: 2}.get(int(p), 3)
        except Exception: name = str(p)
    return {'I': 0, 'P': 1, 'B': 2}.get(name.split('.')[-1], 3)
def rmean(pl, xa, xb, ya, yb, sub):
    H, Wd = pl.shape; xa, xb, ya, yb = int(xa / sub), int(np.ceil(xb / sub)), int(ya / sub), int(np.ceil(yb / sub))
    xa, xb, ya, yb = max(0, xa), min(Wd, xb), max(0, ya), min(H, yb)
    return float(pl[ya:yb, xa:xb].mean()) if (xb > xa and yb > ya) else np.nan
def work(job):
    vid, path, rot = job; out = OUTD / (vid + '.npz')
    if out.exists(): return vid, 'exists'
    try:
        with np.load(ROOT / 'reports/w2_btc_20261116/numeric_dev/blocks' / (vid + '.npz')) as z: boxes = z['boxes'].astype(float)[:, :4]; t_ref = z['t'].copy()
        good = (boxes[:, 2] > boxes[:, 0] + 8) & (boxes[:, 3] > boxes[:, 1] + 8)
        b0 = boxes.copy()
        if good.any():
            for c in range(4): b0[~good, c] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), boxes[good, c])
        B = [b0, median_filter(b0, size=(61, 1), mode='nearest'), np.repeat(np.median(b0[good] if good.any() else b0, 0)[None], len(b0), 0)]
        c = av.open(str(path)); s = c.streams.video[0]; s.thread_type = 'AUTO'; s.codec_context.thread_count = 2
        fmt = s.codec_context.pix_fmt or ''; deep = fmt.endswith('10le') or fmt.endswith('12le'); bps = 2 if deep else 1
        tb = float(s.time_base); psize = {}; rows = []; shape = None
        for pkt in c.demux(s):
            if pkt.pts is not None: psize[pkt.pts] = (pkt.size, int(pkt.is_keyframe))
            for fr in pkt.decode():
                P = []
                for pl in fr.planes[:3]:
                    a = np.frombuffer(pl, dtype='<u2' if deep else np.uint8).reshape(pl.height, pl.line_size // bps)[:, :pl.width]
                    if rot == -90: a = np.rot90(a, -1)
                    elif rot in (-180, 180): a = np.rot90(a, 2)
                    elif rot == 90: a = np.rot90(a, 1)
                    P.append(a)
                H, Wd = P[0].shape; sub = [1, Wd / P[1].shape[1], Wd / P[2].shape[1]]; shape = (H, Wd); i = len(rows)
                yuv = np.full((3, len(REG), 3), np.nan)
                for m in range(3):
                    x0, y0, x1, y1 = B[m][min(i, len(b0) - 1)]; bw, bh = x1 - x0, y1 - y0
                    for k, (fx0, fx1, fy0, fy1) in enumerate(REG.values()):
                        for ch in range(3): yuv[m, k, ch] = rmean(P[ch], x0 + fx0 * bw, x0 + fx1 * bw, y0 + fy0 * bh, y0 + fy1 * bh, sub[ch])
                x0, y0, x1, y1 = B[1][min(i, len(b0) - 1)]; xs = np.linspace(x0, x1, 11); ys = np.linspace(y0, y1, 11); grid = np.full((10, 10, 3), np.nan, np.float32)
                for gy in range(10):
                    for gx in range(10):
                        for ch in range(3): grid[gy, gx, ch] = rmean(P[ch], xs[gx], xs[gx + 1], ys[gy], ys[gy + 1], sub[ch])
                bg = np.array([[rmean(P[ch], 0, Wd // 10, 0, H // 10, sub[ch]) for ch in range(3)], [rmean(P[ch], Wd - Wd // 10, Wd, 0, H // 10, sub[ch]) for ch in range(3)]])
                rows.append((fr.pts, picture_type(fr), yuv, grid, bg))
        c.close(); rows.sort(key=lambda r: r[0])
        pts = np.array([r[0] for r in rows], float) * tb; pts -= pts[0]
        np.savez_compressed(out, t=pts, ptype=np.array([r[1] for r in rows], np.uint8), yuv=np.array([r[2] for r in rows]), grid=np.array([r[3] for r in rows]), bg=np.array([r[4] for r in rows]),
                            size=np.array([psize.get(r[0], (0, 0))[0] for r in rows]), key=np.array([psize.get(r[0], (0, 0))[1] for r in rows], np.uint8), shape=np.array(shape), n_ref=len(t_ref),
                            regions=np.array(list(REG)), modes=np.array(['frame', 'smooth', 'static']), rot=rot, deep=int(deep), pix_fmt=fmt, box_good=float(good.mean()),
                            box_jitter=np.median(np.abs(np.diff(b0, axis=0)), 0), box_size=np.median(b0[:, 2:] - b0[:, :2], 0))
        return vid, f"frames {len(rows)} ref {len(t_ref)} shape {shape} fmt {fmt} good {good.mean():.2f} types {np.bincount([r[1] for r in rows], minlength=4).tolist()}"
    except Exception as e:
        return vid, 'ERROR ' + repr(e)[:300]
if __name__ == '__main__':
    OUTD.mkdir(parents=True, exist_ok=True)
    mp = json.loads((ROOT / 'reports/w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video']; inv = {v: k for k, v in mp.items()}
    files = {p.stem: p for p in (ROOT / 'datasets/FaceVitals_BTC_20261107').rglob('*') if p.is_file() and p.suffix.lower() in ('.mov', '.mp4')}
    meta = {r['recording']: r for r in csv.DictReader((W / 'out/dev_video_meta.csv').open())}
    jobs = [(v, files[inv[v]], int(float(meta[v]['rotation'] or 0))) for v in sorted(inv)]
    jobs.sort(key=lambda j: int(meta[j[0]]['width']) * int(meta[j[0]]['height']))      # smallest first
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        for vid, msg in pool.imap_unordered(work, jobs): print(vid, msg, flush=True)
