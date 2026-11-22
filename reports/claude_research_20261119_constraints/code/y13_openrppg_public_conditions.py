"""y13 - reader_v4 experts (FacePhys.rlap, ME-chunk.pure, ME-flow.pure) on the PUBLIC sets UNDER THE CONSTRAINT CONDITIONS (Q7 = yes, 21 Nov).
Zero-shot, CPU only. Same band-spectrum operator and window rule as y12 (team's 10 s windows; window kept when the condition's own front
end has valid frames over it, duplicate camera time stamps merged).
  run <model> <cond> : cond = an out/x16/<cond>/ crop set (UBFC2 35 + MCD 38 recordings), or c5_split10 (native box crops of out/x10,
                       every k-th frame, k = ceil(fps/10), model told fps/k; all 154 recordings), or c5_split10_jpg60 (the same
                       stream with the 72x72 crop sent as JPEG q60: out/x16/c5_split10_jpg60 from x17, already decimated)  -> out/y13/<cond>/<model>/<key>.npz
                       (start, truth, spec[n,314]); recordings already written are skipped (restartable)
  score <models>     : public estimator (CNN, POS, CHROM + mean standardised log-spectrum of the experts as a 4th channel), readout =
                       forward-backward with quality weight (y07). Four readers, all fitted on training folds only:
                         N3 / N4 : fitted on NATIVE (3 or 4 channels), applied unchanged to every condition
                         P3 / P4 : POOLED as in y07 (native + c4_qvga, c6_us1, aidtrack_c1_g25gv01, c2_vit, track_c1_g25p, track_c7_blue on
                                   the common recordings; c5_face of y07 has no expert crops and is left out), applied unchanged everywhere
                       Paired on the windows where all four have an output. Gates as in PREREG: a coverage >= .9, b not worse than
                       constant, c better than constant (person-level paired bootstrap 10,000x, seed 0). -> out/y13_public_conditions.csv
Usage: venv_orppg/bin/python code/y13_openrppg_public_conditions.py run <model> <cond>   |   $PY code/y13_openrppg_public_conditions.py score <m1,m2,...>"""
import os, sys, json, time
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '2')
os.environ.setdefault('KERAS_BACKEND', 'jax'); os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
POOL = ['c4_qvga', 'c6_us1', 'aidtrack_c1_g25gv01', 'c2_vit', 'track_c1_g25p', 'track_c7_blue']
EVAL = ['c1_g50p', 'c1_g12p', 'c1_g25gv001', 'c7_white', 'c5_split10']   # c5_split10_jpg60 dropped 21 Nov: lossy expert input failed on BTC (y14); the split stream uses lossless crops (z08)
EVID = {'c5_split10_jpg60': 'c5_split10'}   # JPEG only on the expert input; block means (classical) and 36x36 CNN crop are sent unchanged
def merged(t, v):
    u, inv = np.unique(np.asarray(t, float), return_inverse=True); uv = np.ones(len(u), bool); np.logical_and.at(uv, inv, np.asarray(v, bool)); return u, uv
def tag_of(c): c = EVID.get(c, c); return 'x11' if c.startswith('c5_split') else ('x09' if (W / 'out/x09_spec' / c).exists() else 'x01')
if __name__ == '__main__':
    mode = sys.argv[1]; jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] in ('UBFC2', 'MCD')]
    if mode == 'run':
        import rppg
        from lib_btc import supported
        from y12_openrppg_public import band_spectrum
        name, cond = sys.argv[2], sys.argv[3]; split = cond.startswith('c5_split10'); src = (lambda k: W / 'out/x10' / (k + '.npz')) if cond == 'c5_split10' else (lambda k: W / 'out/x16' / cond / (k + '.npz'))
        out = W / 'out/y13' / cond / name; out.mkdir(parents=True, exist_ok=True)
        todo = [j for j in jobs if src(j['key']).exists() and not (out / (j['key'] + '.npz')).exists()]
        m = rppg.Model(name); print(name, cond, 'recordings to do', len(todo), flush=True); nd = 0
        for n, j in enumerate(todo):
            with np.load(src(j['key'])) as z: t, valid, crops = z['t'].astype(float), z['valid'].astype(bool), z['crop72']
            if split:
                fps0 = (len(t) - 1) / (t[-1] - t[0]); k = max(1, int(np.ceil(fps0 / 10 - 1e-3))); t, valid, crops = t[::k], valid[::k], crops[::k]
            fps = (len(t) - 1) / (t[-1] - t[0]); mt, mv = merged(t, valid)
            t0 = time.time(); m.process_faces_tensor(np.ascontiguousarray(crops), fps=float(fps)); bvp, ts = m.bvp(raw=True); sec = time.time() - t0
            bvp = np.asarray(bvp, float); ts = np.asarray(ts, float); ts = ts - ts[0] + t[0] if len(ts) else ts; st = []; tr = []; sp = []
            for w in j['windows']:
                if not w['eligible']: continue
                s = float(w['start']); p = None
                if len(bvp) >= (100 if split else 300) and supported(mt, mv, s, s + 10) and s >= ts[0] - 1e-6 and s + 10 <= ts[-1] + .2: p = band_spectrum(np.interp(s + np.arange(300) / 30., ts, bvp))
                st.append(s); tr.append(float(w['truth'])); sp.append(np.full(314, np.nan) if p is None else p)
            tmp = out / (j['key'] + '.tmp.npz'); np.savez_compressed(tmp, start=np.array(st), truth=np.array(tr), spec=np.array(sp, np.float64).reshape(len(st), 314), seconds=sec, frames=len(t)); tmp.replace(out / (j['key'] + '.npz')); nd += 1
            print(name, cond, n + 1, len(todo), j['key'], 'frames', len(t), 'windows', len(st), 'with output', int(np.isfinite(np.array(sp)).all(1).sum()) if st else 0, '%.1f s' % sec, flush=True)
        print(name, cond, 'done; recordings written', nd, flush=True)
    else:
        import itertools, csv
        from multiprocessing import Pool
        import x05_public_score as X
        import y02_public_improve as Y2
        import y07_public_prior as P7
        models = sys.argv[2].split(','); BPM = X.BPM; DS = X.DS; J = {j['recording']: j for j in jobs}; ZERO = np.zeros(len(BPM))
        def experts(d, cond):
            """4th channel for the windows of d (None where an expert is missing)."""
            key = [(r, round(float(s), 3)) for r, s in zip(d['rec'], d['start'])]; ok = np.ones(len(key), bool); ZE = []
            for m in models:
                tab = {}
                for rec in sorted(set(d['rec'].tolist())):
                    p = (W / 'out/y12' / m / (J[rec]['key'] + '.npz')) if cond == 'native' else (W / 'out/y13' / cond / m / (J[rec]['key'] + '.npz'))
                    if not p.exists(): continue
                    with np.load(p) as z:
                        for s, sp in zip(z['start'], z['spec']):
                            if np.isfinite(sp).all(): tab[(rec, round(float(s), 3))] = sp
                S = np.array([tab.get(k, np.full(len(BPM), np.nan)) for k in key]); h = np.isfinite(S).all(1); ok &= h; Zm = np.zeros((len(key), len(BPM))); Zm[h] = X.zs(S[h]); ZE.append(Zm)
            return np.mean(ZE, 0), ok
        def sub(d, m):
            return {k: (v[m] if k in ('ds', 'person', 'fold', 'rec', 'start', 'truth') else v) for k, v in d.items() if k not in ('Z', 'Zo', 'Z4', 'nrow')} | dict(Z4=d['Z4'][:, m])
        nat = X.load_old_native(); E4, ok = experts(nat, 'native'); nat['Z4'] = np.concatenate([nat['Z'], E4[None]]); nat_all = nat; nat = sub(nat, ok)
        D_ = {}; COV = {}
        for c in POOL + EVAL:
            dc = X.load(tag_of(c), [EVID.get(c, c)]).get(EVID.get(c, c))
            if dc is None: print(c, 'no classical/CNN evidence - skipped'); continue
            E4, okc = experts(dc, c)
            if not okc.any(): print(c, 'no expert output yet - skipped'); continue
            dc['Z4'] = np.concatenate([dc['Z'], E4[None]]); recs = sorted(set(dc['rec'].tolist())); base_n = {D: int(((nat_all['ds'] == D) & np.isin(nat_all['rec'], recs)).sum()) for D in DS}
            D_[c] = sub(dc, okc); COV[c] = {D: (int(((D_[c]['ds'] == D)).sum()), base_n[D]) for D in DS}
        common = sorted(set.intersection(*[set(D_[c]['rec'].tolist()) for c in POOL if c in D_])) if all(c in D_ for c in POOL) else []
        def cat(names):
            ds = [sub(nat, np.isin(nat['rec'], common))] + [sub(D_[c], np.isin(D_[c]['rec'], common)) for c in names]
            out = {k: np.concatenate([d[k] for d in ds]) for k in ('ds', 'person', 'fold', 'start', 'truth')}
            out['rec'] = np.concatenate([np.char.add(n + '|', d['rec'].astype(str)) for n, d in zip(['native'] + names, ds)]); out['Z4'] = np.concatenate([d['Z4'] for d in ds], 1); return out
        POOLD = cat([c for c in POOL if c in D_]) if common else None
        print('pooled data:', None if POOLD is None else (len(common), 'recordings', POOLD['Z4'].shape[1], 'windows'), flush=True)
        G4 = sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in itertools.product([0, .5, 1, 2], repeat=4) if sum(w) > 0}); GR = {3: [w for w in G4 if w[3] == 0], 4: G4}
        def fit(a):
            src, nch, f = a; d = nat if src == 'N' else POOLD; Z = d['Z4']; tr = {D: np.flatnonzero((d['ds'] == D) & (d['fold'] != f)) for D in DS}; best = (1e9, None)
            for w in GR[nch]:
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
            return (src, nch, f), out
        names = ['N3', 'N4'] + (['P3', 'P4'] if POOLD is not None else [])
        with Pool(20) as pool: FITS = dict(pool.map(fit, [(nm[0], int(nm[1]), f) for nm in names for f in range(5)], chunksize=1))
        def apply(d, nm):
            pred = np.full(len(d['truth']), np.nan); const = np.full(len(d['truth']), np.nan); Z = d['Z4']
            for f in range(5):
                m = FITS[(nm[0], int(nm[1]), f)]
                for D in DS:
                    te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f))
                    if not len(te) or D not in m: continue
                    g, s, ta, est, ref, cst = m[D]; const[te] = cst
                    for r in X.runs_of(d, te): pred[r] = P7.point(P7.marg(Y2.qscale(np.tensordot(m['w'], Z[:, r], 1), g, ref), s, ta, ZERO), est)
            return pred, const
        ix = {}
        def boot(v):
            if len(v) not in ix: ix[len(v)] = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
            return np.quantile(v[ix[len(v)]].mean(1), [.025, .975])
        NP = {nm: apply(nat, nm) for nm in names}; nkey = {(r, round(float(s), 3)): i for i, (r, s) in enumerate(zip(nat['rec'], nat['start']))}; rows = []
        for c in ['native'] + [c for c in POOL + EVAL if c in D_]:
            d = nat if c == 'native' else D_[c]; PR = NP if c == 'native' else {nm: apply(d, nm) for nm in names}
            ni = np.array([nkey.get((r, round(float(s), 3)), -1) for r, s in zip(d['rec'], d['start'])])
            for D in DS:
                sel = np.flatnonzero((d['ds'] == D) & (ni >= 0))
                if not len(sel): continue
                t = d['truth'][sel]; per = d['person'][sel]; V = {}
                for nm in names: V[nm] = X.pmae(np.abs(PR[nm][0][sel] - t), per)[1]; V[nm + '_nat'] = X.pmae(np.abs(NP[nm][0][ni[sel]] - t), per)[1]
                V['const'] = X.pmae(np.abs(PR[names[0]][1][sel] - t), per)[1]
                cov = 1.0 if c == 'native' else COV[c][D][0] / max(COV[c][D][1], 1); main = 'P4' if 'P4' in names else 'N4'; ref3 = 'P3' if 'P3' in names else 'N3'
                dc = boot(V[main] - V['const']); dn = boot(V[main] - V[main + '_nat']); d3 = boot(V[main] - V[ref3]); g_ = ('a' if cov >= .9 else '-') + ('b' if dc[0] <= 0 else '-') + ('c' if dc[1] < 0 else '-')
                q = PR[main][0][sel]; ci = boot(V[main])
                print(f"{c:22s} {D:5s} people {len(V[main]):2d} windows {len(sel):4d} cover {cov:.3f} | " + ' '.join(f'{nm} {V[nm].mean():6.3f}' for nm in names) + f" const {V['const'].mean():6.3f} | {main} [{ci[0]:.2f};{ci[1]:.2f}] RMSE {np.sqrt(np.mean((q - t) ** 2)):5.2f} r {np.corrcoef(q, t)[0, 1]:+.3f} | vs {ref3} {np.mean(V[main] - V[ref3]):+.3f} [{d3[0]:+.2f};{d3[1]:+.2f}] | vs native {np.mean(V[main] - V[main + '_nat']):+.3f} [{dn[0]:+.2f};{dn[1]:+.2f}] | vs const {np.mean(V[main] - V['const']):+.3f} [{dc[0]:+.2f};{dc[1]:+.2f}] | gates {g_}", flush=True)
                rows.append(dict(condition=c, dataset=D, people=len(V[main]), windows=len(sel), coverage=cov, **{nm: float(V[nm].mean()) for nm in names}, constant=float(V['const'].mean()), main=main, lo=ci[0], hi=ci[1], d_ref3=float(np.mean(V[main] - V[ref3])), d3_lo=d3[0], d3_hi=d3[1], d_native=float(np.mean(V[main] - V[main + '_nat'])), dn_lo=dn[0], dn_hi=dn[1], d_const=float(np.mean(V[main] - V['const'])), dc_lo=dc[0], dc_hi=dc[1], gates=g_, pooled_in_training=c in ['native'] + POOL))
        print('weights (CNN, POS, CHROM, experts) per fold:', {nm: [tuple(FITS[(nm[0], int(nm[1]), f)]['w']) for f in range(5)] for nm in names})
        with (W / 'out/y13_public_conditions.csv').open('w', newline='') as fh:
            w_ = csv.DictWriter(fh, list(rows[0])); w_.writeheader(); w_.writerows(rows)
