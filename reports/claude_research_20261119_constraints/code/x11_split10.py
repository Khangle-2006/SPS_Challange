"""x11 - C5 in 'split' mode on the public sets (UBFC2 dev, MCD): the device runs the front end (face mesh -> 36x36 masked crop) and
transmits ONLY that crop, at <= 10 fps; the server runs the rest of the pipeline unchanged. Nothing is re-encoded with a video codec:
  stream per frame = 36x36x3 crop rounded to uint8 (3888 B) + 6x6 block means as float32 (432 B) + time stamp and valid flag (9 B)
                   = 4329 B -> 346 kbps at 10 fps (<= 500 kbps);   BTC analogue: code/y08_fps_sim.py (the classical branch needs the means only).
  c5_split10    : every k-th frame of the native crops (k = smallest integer giving <= 10 fps)   [frame skipping, as the C5 specification says]
  c5_split10bin : mean of k consecutive frames, then one kept                                     [device-side temporal binning]
Source = the team's native crop caches (t8_oval_20261031, float16); no video is decoded here.
Usage: $PY code/x11_split10.py <workers>   -> out/x11/<cond>/<key>.npz + out/x11c/<cond>/<key>.npz, then x02_infer.py ... x11"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
OUT = W / 'out/x11'; CROPS = W / 'out/x11c'; BYTES = 36 * 36 * 3 + 36 * 3 * 4 + 9
def stats(c):
    n = len(c); return c.reshape(n, 6, 6, 6, 6, 3).mean((2, 4), dtype=np.float32), (c.max(-1) > 0).reshape(n, 6, 6, 6, 6).sum((2, 4)).astype(np.uint8)
def run(j):
    key = j['key']
    if all((OUT / c / (key + '.npz')).exists() for c in ('c5_split10', 'c5_split10bin')): return dict(key=key, state='skip')
    crop = np.load(Path(j['crop']) / 'rgb.npy', mmap_mode='r')
    with np.load(j['meta']) as z: t, valid = z['t'], z['valid'].astype(bool)
    assert len(t) == len(crop) == len(valid); fps = (len(t) - 1) / (t[-1] - t[0]); k = max(1, int(np.ceil(fps / 10 - 1e-3))); n = len(t) // k; info = dict(key=key, dataset=j['dataset'], fps=float(fps), k=k)
    full = np.asarray(crop[:n * k], np.float32); tt = t[:n * k].reshape(n, k); vv = valid[:n * k].reshape(n, k)
    for cond in ('c5_split10', 'c5_split10bin'):
        if cond == 'c5_split10': c = full[::k]; tc = tt[:, 0]; vc = vv[:, 0]
        else:
            w = vv.astype(np.float32); c = (full.reshape(n, k, 36, 36, 3) * w[..., None, None, None]).sum(1) / np.maximum(w.sum(1), 1)[:, None, None, None]; tc = tt.mean(1); vc = vv.any(1)
        blk, fill = stats(c); c8 = np.clip(np.rint(c), 0, 255).astype(np.uint8)                     # block means before rounding (sent as float32); crop sent as uint8
        for d in (OUT / cond, CROPS / cond): d.mkdir(parents=True, exist_ok=True)
        tmp = OUT / cond / (key + '.tmp.npz'); np.savez_compressed(tmp, t=tc, valid=vc, blk=blk, fill=fill, box=np.zeros((n, 4), np.int16)); tmp.replace(OUT / cond / (key + '.npz'))
        tmp = CROPS / cond / (key + '.tmp.npz'); np.savez_compressed(tmp, crop=c8); tmp.replace(CROPS / cond / (key + '.npz'))
        fo = (n - 1) / (tc[-1] - tc[0]); info.update({'fps_' + cond: float(fo), 'kbps_' + cond: float(BYTES * 8 * fo / 1000), 'valid_' + cond: float(vc.mean())})
    info['state'] = 'done'; return info
if __name__ == '__main__':
    jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] != 'BTC']; log = (W / 'logs/x11_split10.jsonl').open('a'); res = []
    with Pool(int(sys.argv[1])) as pool:
        for info in pool.imap_unordered(run, jobs): log.write(json.dumps(info) + '\n'); log.flush(); res.append(info)
    d = [r for r in res if r['state'] == 'done']
    if d: print(len(d), 'recordings | k', sorted({r['k'] for r in d}), '| output fps %.2f-%.2f' % (min(r['fps_c5_split10'] for r in d), max(r['fps_c5_split10'] for r in d)), '| stream kbps %.0f-%.0f' % (min(r['kbps_c5_split10'] for r in d), max(r['kbps_c5_split10'] for r in d)))
    (W / 'out/x11_DONE').write_text(time.strftime('%F %T')); print('skipped', sum(r['state'] == 'skip' for r in res))
