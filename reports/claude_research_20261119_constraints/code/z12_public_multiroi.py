"""z12 - the teammate's E2 idea (multi-ROI POS with equal-weight spectral fusion, 'M2', from the shared ChatGPT chat of 21 Nov) tested
on OUR public native data: UBFC2 dev (35) + MCD (119), the team's masked 36x36 oval crops (t8 cache), the manifest's 10 s windows,
the window rule and frozen spectral operator of the pipeline (lib_btc). ROIs on the 36x36 crop of the oval box: forehead
(rows 2-9, cols 9-26), image-left cheek (rows 17-26, cols 4-13), image-right cheek (rows 17-26, cols 23-32); mean over non-zero pixels.
  E0      : POS (or CHROM) of the whole-crop mean (as in the pipeline)
  M2      : mean of the three normalised ROI spectra          M2z : mean of the three standardised log spectra (product of experts)
Window HR = spectral argmax; person-first MAE per data set, paired person bootstrap vs E0 (10,000, seed 0).
Usage: $PY code/z12_public_multiroi.py <workers>"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'; sys.path.insert(0, str(OLD / 'code'))
from lib_btc import supported, chrom_team, pos_team, window_spectrum, GRID, zlog
ROIS = dict(forehead=(slice(2, 10), slice(9, 27)), cheek_l=(slice(17, 27), slice(4, 14)), cheek_r=(slice(17, 27), slice(23, 33)))
def roi_mean(c, sl):
    x = c[:, sl[0], sl[1], :].astype(np.float32); m = (x.max(-1) > 0)[..., None]; n = m.sum((1, 2)); return (x * m).sum((1, 2)) / np.maximum(n, 1)
def one(j):
    c = np.load(Path(j['crop']) / 'rgb.npy', mmap_mode='r'); c = np.asarray(c, np.float32)
    with np.load(j['meta']) as z: t, valid = z['t'].astype(float), z['valid'].astype(bool)
    tr = {'whole': c.mean((1, 2)), **{k: roi_mean(c, sl) for k, sl in ROIS.items()}}; out = []
    for w in j['windows']:
        if not w['eligible']: continue
        s = float(w['start'])
        if not supported(t, valid, s, s + 10): continue
        q = s + np.arange(300) / 30.; S = {}
        for k, x in tr.items():
            xi = np.stack([np.interp(q, t, x[:, ch]) for ch in range(3)], 1)
            if not np.isfinite(xi).all() or xi.mean(0).min() <= 0: S = None; break
            for nm, f in (('pos', pos_team), ('chrom', chrom_team)):
                y = f(xi)
                if not np.isfinite(y).all() or y.std() < 1e-12: S = None; break
                S[(k, nm)] = window_spectrum(y)
            if S is None: break
        if S is None: continue
        row = dict(start=s, truth=float(w['truth']))
        for nm in ('pos', 'chrom'):
            row[f'E0_{nm}'] = GRID[S[('whole', nm)].argmax()]
            P = np.array([S[(k, nm)] for k in ROIS]); row[f'M2_{nm}'] = GRID[P.mean(0).argmax()]; row[f'M2z_{nm}'] = GRID[np.mean([zlog(p) for p in P], 0).argmax()]
            row[f'forehead_{nm}'] = GRID[S[('forehead', nm)].argmax()]
        row['E0_pos+chrom'] = GRID[((zlog(S[('whole', 'pos')]) + zlog(S[('whole', 'chrom')])) / 2).argmax()]
        row['M2z_pos+chrom'] = GRID[np.mean([zlog(S[(k, nm)]) for k in ROIS for nm in ('pos', 'chrom')], 0).argmax()]
        out.append(row)
    return j['dataset'], j['person'], out
if __name__ == '__main__':
    jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] in ('UBFC2', 'MCD')]
    with Pool(int(sys.argv[1])) as p: RES = p.map(one, jobs, chunksize=2)
    ix = {}
    def boot(v):
        if len(v) not in ix: ix[len(v)] = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
        return np.quantile(v[ix[len(v)]].mean(1), [.025, .975])
    for D in ('UBFC2', 'MCD'):
        rows = [(per, r) for ds, per, out in RES if ds == D for r in out]; ppl = sorted({p for p, _ in rows})
        def pm(key):
            e = {p: [] for p in ppl}
            for p, r in rows: e[p].append(abs(r[key] - r['truth']))
            return np.array([np.mean(e[p]) for p in ppl])
        print(f'=== {D}: {len(rows)} windows, {len(ppl)} people')
        for fam in ('pos', 'chrom', 'pos+chrom'):
            keys = [f'E0_{fam}'] + [k for k in (f'M2_{fam}', f'M2z_{fam}', f'forehead_{fam}') if k in rows[0][1]]
            ref = pm(keys[0])
            for k in keys:
                v = pm(k); d = boot(v - ref); print(f'   {k:16s} window MAE {v.mean():6.3f} | vs {keys[0]} {np.mean(v - ref):+.3f} [{d[0]:+.2f};{d[1]:+.2f}]', flush=True)
