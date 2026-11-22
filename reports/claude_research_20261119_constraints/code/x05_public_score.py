"""x05 - public sets (UBFC2 dev 35 people, MCD 119 recordings / 60 people) under every condition found in out/<tag>_spec (built by x02).
Estimator = the 18 Nov proposal, unchanged (b04_public_final.py): emission = weighted sum of standardised log-spectra of
{padding-fixed CNN, POS, CHROM}; Viterbi over the consecutive 10 s windows of a recording ('no tracking' is an option).
  refit  : weights and (sigma, tau) fitted on the 4 training folds of the SAME condition (primary)
  frozen : weights and (sigma, tau) fitted on the native training folds, applied to the condition's held-out fold (secondary)
  +oa    : same with the occlusion-aware spectra (rows flagged by colour removed / zeroed) - native (false-alarm cost) and C7
Gates of PREREG.md: (a) >=90% of the native windows still give an output, (b) not worse than the train-fold constant (paired CI95 not
entirely above 0), (c) better than the constant. Person-first MAE, paired bootstrap over people (10,000, seed 0).
Usage: $PY code/x05_public_score.py <workers> [tag=x01] [native_tag=x01] [datasets=UBFC2,MCD]"""
import itertools, sys, json, csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
F = np.fft.rfftfreq(4096, 1 / 30.); BPM = 60 * F[(F >= .7) & (F <= 3.)]
GW = sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in itertools.product([0, .5, 1, 2], repeat=3) if sum(w) > 0})
SIG = [2., 4., 6., 10., 15., 25.]; TAU = [.05, .1, .2, .4, .8, 1.6]; PEN = {s: -((BPM[:, None] - BPM[None]) ** 2) / (2 * s ** 2) for s in SIG}; DS = ['UBFC2', 'MCD']
def zs(S):
    l = np.log(S.astype(np.float64) + 1e-30); return (l - l.mean(1, keepdims=True)) / l.std(1, keepdims=True)
def load(tag, conds=None):
    jobs = {j['key']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] in DS}; D = {}
    for cdir in sorted((W / 'out' / (tag + '_spec')).iterdir()):
        if conds and cdir.name not in conds: continue
        files = sorted(cdir.glob('*.npz')); files = [f for f in files if not f.name.endswith('.tmp.npz') and f.stem in jobs]
        if not files: continue
        a = defaultdict(list)
        for f in files:
            j = jobs[f.stem]
            with np.load(f) as z:
                n = len(z['start'])
                for k in ('pad', 'pos', 'chrom', 'pad_oa', 'pos_oa', 'chrom_oa'): a[k].append(z[k])
                a['start'].append(z['start']); a['truth'].append(z['truth']); a['nrow'].append(z['nrow']); a['ds'] += [j['dataset']] * n; a['person'] += [j['person']] * n; a['fold'] += [j['fold']] * n; a['rec'] += [j['recording']] * n
        d = {k: (np.concatenate(v) if k in ('start', 'truth', 'nrow') else np.array(v)) for k, v in a.items() if k not in ('pad', 'pos', 'chrom', 'pad_oa', 'pos_oa', 'chrom_oa')}
        d['Z'] = np.stack([zs(np.concatenate(a[k])) for k in ('pad', 'pos', 'chrom')]); d['Zo'] = np.stack([zs(np.concatenate(a[k])) for k in ('pad_oa', 'pos_oa', 'chrom_oa')]); d['keys'] = {jobs[f.stem]['recording'] for f in files}; D[cdir.name] = d
    return D
def load_old_native():
    """native spectra of all 154 public recordings as saved on 18 Nov (same windows and operators; no occlusion-aware variant)."""
    with np.load(W.parent / 'claude_research_20261118/out/public_spectra.npz') as z:
        m = (z['cond'] == 'native') & np.isin(z['dataset'], DS); Z = np.stack([zs(z[k][m]) for k in ('pad', 'pos', 'chrom')])
        return dict(ds=z['dataset'][m], person=z['person'][m], fold=z['fold'][m], rec=z['recording'][m], start=z['start'][m], truth=z['truth'][m], nrow=np.zeros(int(m.sum()), np.int8), Z=Z, Zo=Z, keys=set(z['recording'][m].tolist()))
def restrict(d, keep):
    m = np.isin(d['rec'], sorted(keep)); out = {k: (v[m] if k in ('ds', 'person', 'fold', 'rec', 'start', 'truth', 'nrow') else v) for k, v in d.items()}
    out['Z'] = d['Z'][:, m]; out['Zo'] = d['Zo'][:, m]; out['keys'] = set(d['keys']) & set(keep); return out
def pmae(err, per):
    d = defaultdict(list)
    for e, p in zip(err, per): d[p].append(e)
    k = sorted(d); return k, np.array([np.mean(d[p]) for p in k])
def viterbi(E, sigma, tau):
    T, G = E.shape; pen = PEN[sigma]; sc = tau * E[0].copy(); back = np.zeros((T, G), int)
    for t in range(1, T):
        m = sc[:, None] + pen; back[t] = m.argmax(0); sc = m.max(0) + tau * E[t]
    path = np.zeros(T, int); path[-1] = sc.argmax()
    for t in range(T - 1, 0, -1): path[t - 1] = back[t][path[t]]
    return BPM[path]
def runs_of(d, sel):
    by = defaultdict(list)
    for i in sel: by[d['rec'][i]].append(i)
    out = []
    for v in by.values():
        v = np.array(v)[np.argsort(d['start'][v])]; cur = [v[0]]
        for a, b in zip(v[:-1], v[1:]):
            if abs(d['start'][b] - d['start'][a] - 10) < .6: cur.append(b)
            else: out.append(np.array(cur)); cur = [b]
        out.append(np.array(cur))
    return out
def fitp(d, Z, f):
    """weights (pooled over the two sets, each weighing the same) and per-set (sigma, tau), from the folds other than f."""
    best = (1e9, None); tr = {D: np.flatnonzero((d['ds'] == D) & (d['fold'] != f)) for D in DS}
    for w in GW:
        m = sum(pmae(np.abs(BPM[np.tensordot(np.array(w), Z[:, tr[D]], 1).argmax(1)] - d['truth'][tr[D]]), d['person'][tr[D]])[1].mean() for D in DS if len(tr[D]))
        if m < best[0] - 1e-9: best = (m, w)
    w = best[1]; cfg = {}
    for D in DS:
        if not len(tr[D]): cfg[D] = None; continue
        rtr = runs_of(d, tr[D]); E = [np.tensordot(np.array(w), Z[:, r], 1) for r in rtr]
        bc = (pmae(np.abs(BPM[np.tensordot(np.array(w), Z[:, tr[D]], 1).argmax(1)] - d['truth'][tr[D]]), d['person'][tr[D]])[1].mean(), None)
        for s in SIG:
            for ta in TAU:
                e = []; pp = []
                for r, Er in zip(rtr, E): e.extend(np.abs(viterbi(Er, s, ta) - d['truth'][r])); pp.extend(d['person'][r])
                m = pmae(e, pp)[1].mean()
                if m < bc[0] - 1e-9: bc = (m, (s, ta))
        cfg[D] = bc[1]
    return w, cfg
def apply(d, Z, f, w, cfg):
    fus = {}; trk = {}
    for D in DS:
        te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
        if not len(te): continue
        E = np.tensordot(np.array(w), Z[:, te], 1); fus.update(zip(te.tolist(), BPM[E.argmax(1)]))
        for r in runs_of(d, te):
            Er = np.tensordot(np.array(w), Z[:, r], 1); p = BPM[Er.argmax(1)] if cfg[D] is None else viterbi(Er, *cfg[D]); trk.update(zip(r.tolist(), p))
    return fus, trk
def task(a):
    name, var, f = a; d = DATA[name]; Z = d['Z'] if var == 'std' else d['Zo']; w, cfg = fitp(d, Z, f); fus, trk = apply(d, Z, f, w, cfg)
    dn = DATA['native']; wn, cn = NATFIT[f] if NATFIT else fitp(dn, dn['Z'], f); ffus, ftrk = apply(d, Z, f, wn, cn)
    const = {}
    for D in DS:
        tr = (d['ds'] == D) & (d['fold'] != f); te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
        if tr.any(): const.update({i: float(np.median(d['truth'][tr])) for i in te.tolist()})
    return name, var, f, (w, cfg), fus, trk, ffus, ftrk, const
def natfit(f): return fitp(DATA['native'], DATA['native']['Z'], f)
DATA = {}; NATFIT = None
if __name__ == '__main__':
    tag = sys.argv[2] if len(sys.argv) > 2 else 'x01'; ntag = sys.argv[3] if len(sys.argv) > 3 else 'x01'
    if len(sys.argv) > 4: DS[:] = sys.argv[4].split(',')          # restrict to some datasets (e.g. C3 composites exist for UBFC2 only)
    DATA = load(tag)
    if ntag == 'old': DATA['native'] = load_old_native()
    elif 'native' not in DATA: DATA['native'] = load(ntag, ['native'])['native']
    nk = DATA['native']['keys']          # every condition is scored on the recordings that also have a native extraction (paired design)
    DATA = {c: (d if c == 'native' else restrict(d, nk)) for c, d in DATA.items()}
    print('recordings per condition:', {c: len(d['keys']) for c, d in DATA.items()}, flush=True)
    names = ['native'] + [c for c in DATA if c != 'native']; jobs = [(n, v, f) for n in names for v in (['std', 'oa'] if (n == 'native' or n.startswith('c7')) else ['std']) for f in range(5)]
    with Pool(5) as pool: NATFIT = dict(zip(range(5), pool.map(natfit, range(5))))
    with Pool(int(sys.argv[1])) as pool: RES = pool.map(task, jobs, chunksize=1)
    P = defaultdict(dict); INFO = defaultdict(dict)
    for name, var, f, info, fus, trk, ffus, ftrk, const in RES:
        k = (name, var); INFO[k][f] = info
        for key, val in (('fus', fus), ('trk', trk), ('ffus', ffus), ('ftrk', ftrk), ('const', const)): P[k].setdefault(key, {}).update(val)
    rng = np.random.default_rng(0); out = []; dn = DATA['native']; nat = P[('native', 'std')]['trk']
    natkey = {(dn['rec'][i], round(float(dn['start'][i]), 3)): i for i in range(len(dn['truth']))}
    def boot(v): ix = np.random.default_rng(0).integers(0, len(v), (10000, len(v))); return np.quantile(v[ix].mean(1), [.025, .975])
    for (name, var) in P:
        d = DATA[name]; dn = DATA['native']; Z = d['Z'] if var == 'std' else d['Zo']; pr = P[(name, var)]
        print(f"\n=== {name}{'' if var == 'std' else '+oa'} | weights (CNN, POS, CHROM) per fold {[INFO[(name, var)][f][0] for f in range(5)]}")
        for D in DS:
            sel = np.flatnonzero(d['ds'] == D)
            if not len(sel): print(f'   {D:5s} no window with an output: coverage 0.000 | gates ---'); out.append(dict(condition=name, variant=var, dataset=D, windows=0, people=0, coverage=0., gates='---')); continue
            t = d['truth'][sel]; per = d['person'][sel]; cov = len(sel) / max(((dn['ds'] == D) & np.isin(dn['rec'], sorted(d['keys']))).sum(), 1)
            arr = lambda key: np.array([pr[key][i] for i in sel.tolist()])
            cand = {'CNN': BPM[Z[0][sel].argmax(1)], 'POS': BPM[Z[1][sel].argmax(1)], 'CHROM': BPM[Z[2][sel].argmax(1)], 'fusion': arr('fus'), 'fusion+pool': arr('trk'), 'frozen fusion+pool': arr('ftrk'), 'constant': arr('const')}
            ppl, cb = pmae(np.abs(cand['constant'] - t), per)
            # paired with native on the windows both have
            common = [(i, natkey[(d['rec'][i], round(float(d['start'][i]), 3))]) for i in sel.tolist() if (d['rec'][i], round(float(d['start'][i]), 3)) in natkey]
            row = dict(condition=name, variant=var, dataset=D, windows=len(sel), people=len(ppl), coverage=cov, rows_flagged=float((d['nrow'][sel] > 0).mean()), sigma_tau=str([INFO[(name, var)][f][1][D] for f in range(5)]))
            for k, q in cand.items():
                _, v = pmae(np.abs(q - t), per); ci = boot(v); dc = v - cb; c1 = boot(dc)
                row.update({k + '_MAE': float(v.mean()), k + '_lo': float(ci[0]), k + '_hi': float(ci[1]), k + '_RMSE': float(np.sqrt(np.mean((q - t) ** 2))), k + '_dconst': float(dc.mean()), k + '_dc_lo': float(c1[0]), k + '_dc_hi': float(c1[1])})
                extra = ''
                if k in ('fusion+pool', 'frozen fusion+pool') and common:
                    ci_, ni_ = zip(*common); pos = {i: n for n, i in enumerate(sel.tolist())}; qq = np.array([q[pos[i]] for i in ci_]); tt = d['truth'][list(ci_)]; pp = d['person'][list(ci_)]
                    _, v1 = pmae(np.abs(qq - tt), pp); _, v0 = pmae(np.abs(np.array([nat[i] for i in ni_]) - tt), pp); dd = v1 - v0; c2 = boot(dd)
                    row.update({k + '_dnative': float(dd.mean()), k + '_dn_lo': float(c2[0]), k + '_dn_hi': float(c2[1])}); extra = f" | vs native {dd.mean():+.3f} [{c2[0]:+.2f};{c2[1]:+.2f}]"
                print(f"   {D:5s} {k:19s} MAE {v.mean():6.3f} [{ci[0]:.2f};{ci[1]:.2f}] RMSE {row[k + '_RMSE']:5.2f} | vs const {dc.mean():+.3f} [{c1[0]:+.2f};{c1[1]:+.2f}]{extra}")
            g = ('a' if cov >= .9 else '-') + ('b' if row['fusion+pool_dc_lo'] <= 0 else '-') + ('c' if row['fusion+pool_dc_hi'] < 0 else '-'); row['gates'] = g
            print(f"   {D:5s} windows {len(sel)} ({len(ppl)} people) coverage vs native {cov:.3f} | rows flagged in {row['rows_flagged']:.3f} of windows | gates {g}"); out.append(row)
    keys = sorted({k for r in out for k in r}, key=lambda k: (k not in ('condition', 'variant', 'dataset', 'windows', 'people', 'coverage', 'gates'), k))
    with (W / 'out' / (tag + '_public_scores.csv')).open('w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(out)
    np.savez_compressed(W / 'out' / (tag + '_public_pred.npz'), **{f'{n}|{v}|{k}|{w_}': np.array(x) for (n, v), pr in P.items() for k in ('trk', 'ftrk', 'fus', 'const') for w_, x in (('idx', sorted(pr[k])), ('val', [pr[k][i] for i in sorted(pr[k])]))})
