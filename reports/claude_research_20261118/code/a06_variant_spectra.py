"""Window-averaged CHROM/POS spectra (same reader input as reader_v1) for extraction variants built from out/native/*.npz (a05):
box placement (frame / smooth / static) x region (c60, c40, box, forehead, cheeks, lower), plus the frozen oval trace of the numeric
export as the baseline, an anchors-only version (I and P pictures only) and a GOP-template-removed version of smooth:c60.
No labels are read here. Output: out/variant_spectra.npz (CHROM and POS arrays [variant, video, grid])."""
import sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import *
V = videos(); M = meta(); REC = [v['recording'] for v in V]
MODES = ['frame', 'smooth', 'static']; REGS = ['c60', 'c40', 'box', 'forehead', 'cheeks', 'lower']
NAMES = ['oval_frozen'] + [f'{m}:{r}' for m in MODES for r in REGS] + ['smooth:c60:anchors', 'smooth:c60:nogop', 'static:c60:anchors', 'smooth:fh+ck', 'static:fh+ck']
def gop_template(x, ptype):
    r = x - trend(len(x), 30.)(x); pos_in = np.zeros(len(x), int); last = 0
    for i in range(len(x)):
        if ptype[i] == 0: last = i
        pos_in[i] = i - last
    tpl = np.zeros(len(x))
    for k in np.unique(pos_in):
        m = pos_in == k
        if m.sum() >= 4: tpl[m] = np.median(r[m])
    return tpl
def one(rec):
    out = {}
    b = load_blocks(rec); a = avg_spectra(b['t'], b['valid'], b['rgb_mean']); out['oval_frozen'] = a
    p = OUT / 'native' / (rec + '.npz')
    if not p.exists(): return rec, out
    z = np.load(p); t = z['t']; rgb = yuv_to_rgb(z['yuv'], M[rec], int(z['deep'])); pt = z['ptype']; ok = np.ones(len(t), bool)
    for mi, m in enumerate(MODES):
        for ri, r in enumerate(REGS):
            x = rgb[:, mi, ri]; v = np.isfinite(x).all(1); out[f'{m}:{r}'] = avg_spectra(t, v, np.nan_to_num(x, nan=1.))
    for mi, m in [(1, 'smooth'), (2, 'static')]:
        x = rgb[:, mi, 0]; keep = pt != 2
        out[f'{m}:c60:anchors'] = avg_spectra(t[keep], np.isfinite(x[keep]).all(1), np.nan_to_num(x[keep], nan=1.)) if keep.sum() > 60 else None
        y = (rgb[:, mi, 3] + rgb[:, mi, 4]) / 2; out[f'{m}:fh+ck'] = avg_spectra(t, np.isfinite(y).all(1), np.nan_to_num(y, nan=1.))
    x = rgb[:, 1, 0]
    if np.isfinite(x).all() and len(x) > 90: out['smooth:c60:nogop'] = avg_spectra(t, ok, x - np.stack([gop_template(x[:, c], pt) for c in range(3)], 1))
    return rec, out
if __name__ == '__main__':
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool: res = dict(pool.imap_unordered(one, REC, chunksize=2))
    C = np.full((len(NAMES), len(REC), len(GRID)), np.nan, np.float32); P = C.copy(); NW = np.zeros((len(NAMES), len(REC)), int)
    for j, r in enumerate(REC):
        for i, n in enumerate(NAMES):
            a = res[r].get(n)
            if a is not None: C[i, j], P[i, j], NW[i, j] = a
    np.savez_compressed(OUT / 'variant_spectra.npz', names=np.array(NAMES), rec=np.array(REC), chrom=C, pos=P, nwin=NW, grid=GRID)
    print('saved', C.shape, '| videos with spectra per variant:', {n: int(np.isfinite(C[i]).all(1).sum()) for i, n in enumerate(NAMES)})
