"""z04 - quick classical-only score of per-frame traces on the PUBLIC sets (for front-end tests such as x18): for each listed
<tag>/<cond> trace set, per eligible 10 s window of the manifest (window rule of the pipeline on the merged time base), CHROM and POS
spectra (frozen operators), fused z = mean of the standardised log spectra; window HR = argmax; also the recording median of the
window HRs. Person-first MAE per data set on the windows that EVERY listed set has (paired), with paired person bootstrap.
Usage: $PY code/z04_public_classical_score.py tag/cond[,tag/cond...] [keys-prefix]"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
sys.path.insert(0, str(OLD / 'code'))
from lib_btc import supported, chrom_team, pos_team, window_spectrum, GRID, zlog
SETS = sys.argv[1].split(','); PRE = sys.argv[2] if len(sys.argv) > 2 else ''
jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] in ('UBFC2', 'MCD') and j['key'].startswith(PRE)]
def one(a):
    s_, j = a; p = W / 'out' / s_ / (j['key'] + '.npz')
    if not p.exists(): return None
    with np.load(p) as z: t, valid, blk = z['t'].astype(float), z['valid'].astype(bool), z['blk'].astype(float)
    u, inv = np.unique(t, return_inverse=True); cnt = np.bincount(inv); rgb = np.zeros((len(u), 3)); np.add.at(rgb, inv, blk.mean((1, 2))); rgb /= cnt[:, None]
    uv = np.ones(len(u), bool); np.logical_and.at(uv, inv, valid); out = {}
    for w in j['windows']:
        if not w['eligible']: continue
        s = float(w['start'])
        if not supported(u, uv, s, s + 10): continue
        q = s + np.arange(300) / 30.; x = np.stack([np.interp(q, u, rgb[:, c]) for c in range(3)], 1)
        if not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        wc, wp = chrom_team(x), pos_team(x)
        if not (np.isfinite(wc).all() and np.isfinite(wp).all()) or wc.std() < 1e-12 or wp.std() < 1e-12: continue
        zf = (zlog(window_spectrum(wc)) + zlog(window_spectrum(wp))) / 2; out[round(s, 3)] = (float(GRID[zf.argmax()]), float(w['truth']))
    return j['recording'], j['dataset'], j['person'], out
if __name__ == '__main__':
    with Pool(8) as pool: RES = {s_: pool.map(one, [(s_, j) for j in jobs], chunksize=2) for s_ in SETS}
    rows = {}
    for s_ in SETS:
        for r in RES[s_]:
            if r is None: continue
            rec, ds, per, out = r
            for st, v in out.items(): rows.setdefault((rec, st), {'ds': ds, 'per': per})[s_] = v
    common = [k for k, v in rows.items() if all(s_ in v for s_ in SETS)]
    print('windows per set:', {s_: sum(1 for v in rows.values() if s_ in v) for s_ in SETS}, '| common', len(common), flush=True)
    ix = {}
    def boot(v):
        if len(v) not in ix: ix[len(v)] = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
        return np.quantile(v[ix[len(v)]].mean(1), [.025, .975])
    for D in ('UBFC2', 'MCD'):
        ks = [k for k in common if rows[k]['ds'] == D]
        if not ks: continue
        ppl = sorted({rows[k]['per'] for k in ks}); V = {}; VR = {}
        for s_ in SETS:
            e = {p: [] for p in ppl}; er = {}
            for k in ks: e[rows[k]['per']].append(abs(rows[k][s_][0] - rows[k][s_][1])); er.setdefault((rows[k]['per'], k[0]), []).append(rows[k][s_])
            V[s_] = np.array([np.mean(e[p]) for p in ppl])
            rp = {p: [] for p in ppl}
            for (p, rec), L in er.items(): m = np.median([x[0] for x in L]); rp[p] += [abs(m - x[1]) for x in L]
            VR[s_] = np.array([np.mean(rp[p]) for p in ppl])
        for s_ in SETS:
            d1 = boot(V[s_] - V[SETS[0]]); d2 = boot(VR[s_] - VR[SETS[0]])
            print(f"{D:5s} {s_:34s} people {len(ppl):2d} windows {len(ks):4d} | window MAE {V[s_].mean():6.2f} vs first {np.mean(V[s_] - V[SETS[0]]):+.2f} [{d1[0]:+.2f};{d1[1]:+.2f}] | recording-median MAE {VR[s_].mean():6.2f} vs first {np.mean(VR[s_] - VR[SETS[0]]):+.2f} [{d2[0]:+.2f};{d2[1]:+.2f}]", flush=True)
