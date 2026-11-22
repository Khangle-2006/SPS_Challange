"""z14 - FREEZE the public reader P4m chosen by the team on 22 Nov 2026 (P4 + 0.1 s moving average of the crops before the team CNN;
documented deviation from the z13 acceptance rule: UBFC2 native +0.043 [+0.006; +0.080]). Same data, channels and fitting code as z13
(reader 'P4m' = pooled fit on native + 6 POOL conditions, channel set B). Saves everything the integration (prompt B) must reproduce:
  out/z14_p4m_frozen.npz : per fold f = 0..4 the fusion weights and, per data set, (gamma, sigma, tau, estimator, ref, constant);
                           the same for f = -1 = DEPLOY fit on all dev folds (for data outside the dev CV); and the per-window
                           out-of-fold predictions of every condition (cond, ds, rec, start, truth, pred).
  logs/P4M_DEFINITION.txt: definition, parameters, sha256 of the npz.
Check printed at the end: per condition x data set MAE vs the P4m column of out/z13_public_p4_prefilter.csv (must match to 1e-9).
Usage: $PY code/z14_p4m_freeze.py"""
import sys, json, itertools, csv
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
import x05_public_score as X
import y02_public_improve as Y2
import y07_public_prior as P7
MODELS = ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']
POOL = ['c4_qvga', 'c6_us1', 'aidtrack_c1_g25gv01', 'c2_vit', 'track_c1_g25p', 'track_c7_blue']
EVAL = ['c1_g50p', 'c1_g12p', 'c1_g25gv001', 'c7_white', 'c5_split10']
BPM = X.BPM; DS = X.DS; ZERO = np.zeros(len(BPM)); FIELDS = ('ds', 'person', 'fold', 'rec', 'start', 'truth')
jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] in DS]; J = {j['recording']: j for j in jobs}
def tag_of(c): return 'x11' if c.startswith('c5_split') else ('x09' if (W / 'out/x09_spec' / c).exists() else 'x01')
def kk(d): return [(r, round(float(s), 3)) for r, s in zip(d['rec'], d['start'])]
def experts(d, cond):
    key = kk(d); ok = np.ones(len(key), bool); ZE = []
    for m in MODELS:
        tab = {}
        for rec in sorted(set(d['rec'].tolist())):
            p = (W / 'out/y12' / m / (J[rec]['key'] + '.npz')) if cond == 'native' else (W / 'out/y13' / cond / m / (J[rec]['key'] + '.npz'))
            if not p.exists(): continue
            with np.load(p) as z:
                for s, sp in zip(z['start'], z['spec']):
                    if np.isfinite(sp).all(): tab[(rec, round(float(s), 3))] = sp
        S = np.array([tab.get(k, np.full(len(BPM), np.nan)) for k in key]); h = np.isfinite(S).all(1); ok &= h; Zm = np.zeros((len(key), len(BPM))); Zm[h] = X.zs(S[h]); ZE.append(Zm)
    return np.mean(ZE, 0), ok
NAT_ALL = X.load_old_native()
def build(c):
    """windows of condition c with both channel sets: A = (CNN, POS, CHROM, experts), B = (pre-filtered CNN, POS, CHROM, experts)."""
    if c == 'native': o = NAT_ALL; m_ = X.load('x13', ['nat_ma3']).get('nat_ma3')
    else: o = X.load(tag_of(c), [c]).get(c); m_ = None if c.startswith('c5_split') else X.load('x20', [c]).get(c)
    if o is None or (m_ is None and not c.startswith('c5_split')): return None
    E, ok = experts(o, c); ko = kk(o)
    if m_ is None: zm = o['Z'][0]; okm = np.ones(len(ko), bool)
    else:
        km = {k: i for i, k in enumerate(kk(m_))}; ix = np.array([km.get(k, -1) for k in ko]); okm = ix >= 0; zm = np.zeros_like(o['Z'][0]); zm[okm] = m_['Z'][0][ix[okm]]
    keep = ok & okm; d = {k: np.asarray(o[k])[keep] for k in FIELDS}
    d['A'] = np.concatenate([o['Z'][:, keep], E[None, keep]]); d['B'] = np.concatenate([zm[None, keep], o['Z'][1:, keep], E[None, keep]])
    recs = sorted(set(d['rec'].tolist())); base = {D: int(((NAT_ALL['ds'] == D) & np.isin(NAT_ALL['rec'], recs)).sum()) for D in DS}
    d['cov'] = {D: (int((d['ds'] == D).sum()), base[D]) for D in DS}; d['lost_prefilter'] = int((ok & ~okm).sum()); return d
def sub(d, m): return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim == 1 else (v[:, m] if isinstance(v, np.ndarray) else v)) for k, v in d.items()}
D_ = {}
for c in ['native'] + POOL + EVAL:
    d = build(c)
    if d is None: print(c, 'missing input - skipped', flush=True); continue
    D_[c] = d; print(c, 'windows', len(d['truth']), 'coverage', d['cov'], 'windows lost to missing pre-filtered CNN', d['lost_prefilter'], flush=True)
NAT = D_['native']; common = sorted(set.intersection(*[set(D_[c]['rec'].tolist()) for c in POOL])) if all(c in D_ for c in POOL) else []
assert common, 'pooled set needs every POOL condition'
parts = [sub(NAT, np.isin(NAT['rec'], common))] + [sub(D_[c], np.isin(D_[c]['rec'], common)) for c in POOL]
POOLD = {k: np.concatenate([p[k] for p in parts]) for k in ('ds', 'person', 'fold', 'start', 'truth')}
POOLD['rec'] = np.concatenate([np.char.add(n + '|', p['rec'].astype(str)) for n, p in zip(['native'] + POOL, parts)])
for ch in 'AB': POOLD[ch] = np.concatenate([p[ch] for p in parts], 1)
print('pooled', len(common), 'recordings', len(POOLD['truth']), 'windows', flush=True)
G4 = sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in itertools.product([0, .5, 1, 2], repeat=4) if sum(w) > 0})
READERS = {'P4m': ('P', 'B')}
def fit(a):
    nm, f = a; src, ch = READERS[nm]; d = NAT if src == 'N' else POOLD; Z = d[ch]; tr = {D: np.flatnonzero((d['ds'] == D) & (d['fold'] != f)) for D in DS}; best = (1e9, None)
    for w in G4:
        l = sum(X.pmae(np.abs(BPM[np.tensordot(np.array(w), Z[:, tr[D]], 1).argmax(1)] - d['truth'][tr[D]]), d['person'][tr[D]])[1].mean() for D in DS if len(tr[D]))
        if l < best[0] - 1e-9: best = (l, w)
    w = np.array(best[1]); out = dict(w=w)
    for D in DS:
        if not len(tr[D]): continue
        rtr = X.runs_of(d, tr[D]); Etr = [np.tensordot(w, Z[:, r], 1) for r in rtr]; ref = float(np.mean(np.concatenate([E.max(1) for E in Etr]))); tt = np.concatenate([d['truth'][r] for r in rtr]); pp = np.concatenate([d['person'][r] for r in rtr]); b2 = (1e9, None)
        for g in P7.GAM:
            Eg = [Y2.qscale(E, g, ref) for E in Etr]
            for s in X.SIG:
                for ta in X.TAU:
                    M_ = [P7.marg(E, s, ta, ZERO) for E in Eg]
                    for est in ('median', 'mean'):
                        l = X.pmae(np.abs(np.concatenate([P7.point(x, est) for x in M_]) - tt), pp)[1].mean()
                        if l < b2[0] - 1e-9: b2 = (l, (g, s, ta, est))
        out[D] = b2[1] + (ref, float(np.median(d['truth'][tr[D]])))
    return (nm, f), out
if __name__ == '__main__':
    import hashlib, time
    FOLDS = [0, 1, 2, 3, 4, -1]
    with Pool(6) as pool: FITS = dict(pool.map(fit, [('P4m', f) for f in FOLDS], chunksize=1))
    def apply(d):
        Z = d['B']; pred = np.full(len(d['truth']), np.nan)
        for f in range(5):
            m = FITS[('P4m', f)]
            for D in DS:
                te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
                if not len(te) or D not in m: continue
                g, s, ta, est, ref, cst = m[D]
                for r in X.runs_of(d, te): pred[r] = P7.point(P7.marg(Y2.qscale(np.tensordot(m['w'], Z[:, r], 1), g, ref), s, ta, ZERO), est)
        return pred
    rows = {k: [] for k in ('cond', 'ds', 'rec', 'start', 'truth', 'pred', 'person', 'fold')}
    nkey = {k: i for i, k in enumerate(kk(NAT))}; ref_csv = {}
    import csv as _csv
    for r_ in _csv.DictReader(open(W / 'out/z13_public_p4_prefilter.csv')): ref_csv[(r_['condition'], r_['dataset'])] = float(r_['P4m'])
    worst = 0.0
    for c in D_:
        d = D_[c]; p = apply(d); ni = np.array([nkey.get(k, -1) for k in kk(d)])
        for k, v in (('cond', np.full(len(p), c)), ('ds', d['ds']), ('rec', d['rec']), ('start', d['start']), ('truth', d['truth']), ('pred', p), ('person', d['person']), ('fold', d['fold'])): rows[k].append(np.asarray(v))
        for D in DS:
            sel = np.flatnonzero((d['ds'] == D) & (ni >= 0))
            if not len(sel): continue
            mae = X.pmae(np.abs(p[sel] - d['truth'][sel]), d['person'][sel])[1].mean(); ref = ref_csv.get((c, D)); dlt = abs(mae - ref) if ref is not None else float('nan')
            worst = max(worst, dlt if dlt == dlt else 0.0)
            print(f"{c:20s} {D:5s} windows {len(sel):4d} P4m {mae:.6f} z13 {ref if ref is not None else float('nan'):.6f} |diff| {dlt:.2e}", flush=True)
    out = {k: np.concatenate(v) for k, v in rows.items()}
    for f in FOLDS:
        m = FITS[('P4m', f)]; tag = 'deploy' if f < 0 else f'fold{f}'
        out[f'w_{tag}'] = np.asarray(m['w'], float)
        for D in DS:
            if D in m: g, s, ta, est, ref, cst = m[D]; out[f'ro_{tag}_{D}'] = np.array([g, s, ta, 1.0 if est == 'median' else 0.0, ref, cst], float)
    out['readme'] = np.array('channels B = (team CNN on ma3-prefiltered crops, POS, CHROM, mean zlog spectrum of FacePhys.rlap / ME-chunk.pure / ME-flow.pure); ro_* = (gamma, sigma, tau, median?1:mean, ref, train-fold median HR); deploy = fit on all dev folds')
    np.savez_compressed(W / 'out/z14_p4m_frozen.npz', **out)
    h = hashlib.sha256((W / 'out/z14_p4m_frozen.npz').read_bytes()).hexdigest()
    lines = ['P4m public reader, frozen ' + time.strftime('%F %T') + ' (team decision 22 Nov 2026: P4m, documented deviation from the z13 rule)',
             '  channels: CNN (team padding-fixed EfficientPhys+TSM, out-of-fold checkpoint) on crops pre-filtered by a centred moving average of k = ceil(fps/10)',
             '            frames weighted by the valid flags (x13 nat_ma3, x20 under conditions; C5 split keeps its binned 10 fps stream); POS and CHROM from the',
             '            UNFILTERED 6x6 block means; experts = mean of the standardised log spectra of FacePhys.rlap, ME-chunk.pure, ME-flow.pure (y12/y13).',
             '  fusion  : E = w . Z (Z = per-window standardised log spectra on the 42-180 bpm grid), w fitted on the training folds over {0,.5,1,2}^4 (normalised).',
             '  readout : forward-backward with quality weight (y07 marg/point, y02 qscale), (gamma, sigma, tau, estimator) per data set per fold.',
             '  pooled  : training windows = native + c4_qvga, c6_us1, aidtrack_c1_g25gv01, c2_vit, track_c1_g25p, track_c7_blue on the common recordings.']
    for f in FOLDS:
        m = FITS[('P4m', f)]; tag = 'deploy' if f < 0 else f'fold {f}'
        lines.append(f'  {tag:7s}: w (CNN, POS, CHROM, experts) = {tuple(float(x) for x in m["w"])}; ' + '; '.join(f'{D} (gamma, sigma, tau, est, ref, const) = {m[D][0]}, {m[D][1]}, {m[D][2]}, {m[D][3]}, {m[D][4]:.6f}, {m[D][5]:.3f}' for D in DS if D in m))
    lines += [f'  check   : max |MAE - z13 P4m| over conditions x data sets = {worst:.2e}', f'  sha256  : {h}  out/z14_p4m_frozen.npz']
    (W / 'logs/P4M_DEFINITION.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines), flush=True)
