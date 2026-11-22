"""x13 - temporal pre-filter at the NATIVE frame times, public sets (UBFC2 dev, MCD). Follow-up of x11: the 10 fps binned split stream gave a
BETTER result than the native 30 fps crops (MCD: CNN alone 6.58 -> 2.69, fused and pooled 2.32 -> 1.61), i.e. the gain comes from
removing frame-to-frame flicker, not from the lower frame rate. Here the same filter is applied without changing the time stamps, so
that every native window stays eligible and the comparison with native is window for window:
  nat_b10i : mean of k consecutive crops (k = smallest integer giving <= 10 fps), linearly interpolated back to every original frame time
  nat_ma3  : centred moving average over k crops (the simplest low-pass)
Source = the team's native crop caches (float16). Block means are computed from the filtered crops.
Usage: $PY code/x13_prefilter.py <workers>  -> out/x13/<cond>/<key>.npz + out/x13c/<cond>/<key>.npz, then x02_infer.py ... x13"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
OUT = W / 'out/x13'; CROPS = W / 'out/x13c'; CONDS = ('nat_b10i', 'nat_ma3')
def stats(c):
    n = len(c); return c.reshape(n, 6, 6, 6, 6, 3).mean((2, 4), dtype=np.float32), (c.max(-1) > 0).reshape(n, 6, 6, 6, 6).sum((2, 4)).astype(np.uint8)
def run(j):
    key = j['key']
    if all((OUT / c / (key + '.npz')).exists() for c in CONDS): return dict(key=key, state='skip')
    crop = np.asarray(np.load(Path(j['crop']) / 'rgb.npy', mmap_mode='r'), np.float32)
    with np.load(j['meta']) as z: t, valid = z['t'], z['valid'].astype(bool)
    assert len(t) == len(crop) == len(valid); N = len(t); fps = (N - 1) / (t[-1] - t[0]); k = max(1, int(np.ceil(fps / 10 - 1e-3))); w = valid.astype(np.float32); out = {}
    n = N // k; ww = w[:n * k].reshape(n, k); cb = (crop[:n * k].reshape(n, k, 36, 36, 3) * ww[..., None, None, None]).sum(1) / np.maximum(ww.sum(1), 1)[:, None, None, None]; tb = t[:n * k].reshape(n, k).mean(1)
    q = np.clip(t, tb[0], tb[-1]); ix = np.searchsorted(tb, q, side='right').clip(1, n - 1); a = ((q - tb[ix - 1]) / (tb[ix] - tb[ix - 1])).astype(np.float32)
    out['nat_b10i'] = cb[ix - 1] * (1 - a[:, None, None, None]) + cb[ix] * a[:, None, None, None]
    pad = k // 2; cs = np.cumsum(np.concatenate([np.zeros((1, 36, 36, 3), np.float64), crop * w[:, None, None, None]]), 0); cw = np.concatenate([[0.], np.cumsum(w)])
    lo = np.clip(np.arange(N) - pad, 0, N); hi = np.clip(np.arange(N) - pad + k, 0, N); out['nat_ma3'] = ((cs[hi] - cs[lo]) / np.maximum(cw[hi] - cw[lo], 1)[:, None, None, None]).astype(np.float32)
    for cond, c in out.items():
        blk, fill = stats(c)
        for d in (OUT / cond, CROPS / cond): d.mkdir(parents=True, exist_ok=True)
        tmp = OUT / cond / (key + '.tmp.npz'); np.savez_compressed(tmp, t=t, valid=valid, blk=blk, fill=fill, box=np.zeros((N, 4), np.int16)); tmp.replace(OUT / cond / (key + '.npz'))
        tmp = CROPS / cond / (key + '.tmp.npz'); np.savez_compressed(tmp, crop=c.astype(np.float16)); tmp.replace(CROPS / cond / (key + '.npz'))
    return dict(key=key, dataset=j['dataset'], fps=float(fps), k=k, state='done')
if __name__ == '__main__':
    jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] != 'BTC']; res = []
    with Pool(int(sys.argv[1])) as pool:
        for info in pool.imap_unordered(run, jobs): res.append(info)
    print('done', sum(r['state'] == 'done' for r in res), 'skipped', sum(r['state'] == 'skip' for r in res), '| k', sorted({r['k'] for r in res if 'k' in r}))
    (W / 'out/x13_DONE').write_text(time.strftime('%F %T'))
