"""y12 - the pretrained experts of reader_v4 on the PUBLIC sets (UBFC2 dev, MCD), zero-shot: one pipeline benchmarked everywhere
(the official text asks for a benchmark against a recognised standard data set). No training, no GPU, no BTC holdout.
  run <model>      every public recording that has unmasked face-box crops (out/x10/<key>.npz) -> out/y12/<model>/<key>.npz:
                   one band spectrum (0.7-3 Hz, 314 FFT bins; same operator as the CNN / POS / CHROM evidence of x02) of the model's
                   BVP per eligible 10 s window of the manifest. Recordings already written are skipped (the run can be restarted).
  score <models>   public estimator of 18-20 Nov (CNN, POS, CHROM; fusion weights and forward-backward readout fitted on the training
                   folds) with the mean standardised log-spectrum of the listed experts as a FOURTH channel. Its weight is chosen on
                   the training folds from {0, .5, 1, 2} together with the other three, so 'no expert' stays available (nested).
                   Windows = those of the 18 Nov native evidence for which every listed expert has an output (paired comparison).
Usage: venv_orppg/bin/python code/y12_openrppg_public.py run <model> [keys]     |     $PY code/y12_openrppg_public.py score <m1,m2,...>"""
import os, sys, json, time
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '2')
os.environ.setdefault('KERAS_BACKEND', 'jax'); os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
def band_spectrum(w):
    from scipy import signal
    from lib_btc import trend, SOS1
    w = np.asarray(w, float)
    if not np.isfinite(w).all() or w.std() < 1e-10: return None
    y = signal.sosfiltfilt(SOS1, w - trend(len(w))(w)); f, p = signal.periodogram(y, fs=30, window='hann', nfft=4096, detrend=False); p = p[(f >= .7) & (f <= 3.)]
    return p if np.isfinite(p).all() and p.sum() > 1e-30 else None
if __name__ == '__main__':
    mode = sys.argv[1]; jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] in ('UBFC2', 'MCD')]
    if mode == 'run':
        import rppg
        from lib_btc import supported
        name = sys.argv[2]; out = W / 'out/y12' / name; out.mkdir(parents=True, exist_ok=True); todo = [j for j in jobs if (W / 'out/x10' / (j['key'] + '.npz')).exists() and not (out / (j['key'] + '.npz')).exists()]
        if len(sys.argv) > 3: todo = [j for j in todo if j['key'] in sys.argv[3].split(',')]          # optional: only these recordings (checks)
        m = rppg.Model(name); print(name, 'recordings to do', len(todo), 'of', len(jobs), flush=True); nd = 0
        for n, j in enumerate(todo):
            with np.load(W / 'out/x10' / (j['key'] + '.npz')) as z: t, valid, crops = z['t'], z['valid'].astype(bool), z['crop72']
            with np.load(j['meta']) as z: mt, mv = z['t'], z['valid'].astype(bool)                 # the team's time base (duplicate camera time stamps merged): the one their window rule uses
            fps = (len(t) - 1) / (t[-1] - t[0]); t0 = time.time(); m.process_faces_tensor(np.ascontiguousarray(crops), fps=float(fps)); bvp, ts = m.bvp(raw=True); sec = time.time() - t0
            bvp = np.asarray(bvp, float); ts = np.asarray(ts, float); ts = ts - ts[0] + t[0] if len(ts) else ts; st = []; tr = []; sp = []
            for w in j['windows']:
                if not w['eligible']: continue
                s = float(w['start']); p = None
                if len(bvp) >= 300 and supported(mt, mv, s, s + 10) and s >= ts[0] - 1e-6 and s + 10 <= ts[-1] + .2: p = band_spectrum(np.interp(s + np.arange(300) / 30., ts, bvp))
                st.append(s); tr.append(float(w['truth'])); sp.append(np.full(314, np.nan) if p is None else p)
            tmp = out / (j['key'] + '.tmp.npz'); np.savez_compressed(tmp, start=np.array(st), truth=np.array(tr), spec=np.array(sp, np.float64).reshape(len(st), 314), seconds=sec, frames=len(t)); tmp.replace(out / (j['key'] + '.npz')); nd += 1
            print(name, n + 1, len(todo), j['key'], 'frames', len(t), 'windows', len(st), 'with output', int(np.isfinite(np.array(sp)).all(1).sum()) if st else 0, '%.1f s' % sec, flush=True)
        print(name, 'done; recordings written', nd, flush=True)
    else:
        import itertools
        from collections import defaultdict
        from multiprocessing import Pool
        import x05_public_score as X
        import y02_public_improve as Y2
        import y07_public_prior as P7
        models = sys.argv[2].split(','); d0 = X.load_old_native(); n0 = len(d0['truth']); BPM = X.BPM; DS = X.DS; J = {j['recording']: j for j in jobs}
        key = [(r, round(float(s), 3)) for r, s in zip(d0['rec'], d0['start'])]; ZE = {}; ok = np.ones(n0, bool)
        for m in models:
            tab = {}
            for rec in sorted(d0['keys']):
                p = W / 'out/y12' / m / (J[rec]['key'] + '.npz')
                if not p.exists(): continue
                with np.load(p) as z:
                    for s, sp in zip(z['start'], z['spec']):
                        if np.isfinite(sp).all(): tab[(rec, round(float(s), 3))] = sp
            S = np.array([tab.get(k, np.full(len(BPM), np.nan)) for k in key]); have = np.isfinite(S).all(1); ok &= have; Zm = np.zeros((n0, len(BPM))); Zm[have] = X.zs(S[have]); ZE[m] = Zm
            print(f'{m}: windows with output {int(have.sum())} / {n0}', flush=True)
        d = {k: (v[ok] if k in ('ds', 'person', 'fold', 'rec', 'start', 'truth') else v) for k, v in d0.items()}; Z = np.concatenate([d0['Z'][:, ok], np.mean([ZE[m][ok] for m in models], 0)[None]]); n = int(ok.sum())
        print({D: (int((d['ds'] == D).sum()), int((d0['ds'] == D).sum())) for D in DS}, 'windows kept / all', flush=True)
        G4 = sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in itertools.product([0, .5, 1, 2], repeat=4) if sum(w) > 0}); GRIDS = dict(base=[w for w in G4 if w[3] == 0], full=G4, experts=[(0., 0., 0., 1.)]); ZERO = np.zeros(len(BPM))
        def fit(a):
            f, name = a; tr = {D: np.flatnonzero((d['ds'] == D) & (d['fold'] != f)) for D in DS}; best = (1e9, None)
            for w in GRIDS[name]:
                l = sum(X.pmae(np.abs(BPM[np.tensordot(np.array(w), Z[:, tr[D]], 1).argmax(1)] - d['truth'][tr[D]]), d['person'][tr[D]])[1].mean() for D in DS if len(tr[D]))
                if l < best[0] - 1e-9: best = (l, w)
            w = np.array(best[1]); out = dict(w=tuple(float(x) for x in w)); pred = {}; fus = {}
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
                out[D] = b2[1]; te = np.flatnonzero((d['ds'] == D) & (d['fold'] == f)); g, s, ta, est = b2[1]
                for r in X.runs_of(d, te):
                    E = np.tensordot(w, Z[:, r], 1); fus.update(zip(r.tolist(), BPM[E.argmax(1)])); pred.update(zip(r.tolist(), P7.point(P7.marg(Y2.qscale(E, g, ref), s, ta, ZERO), est)))
            return f, name, out, fus, pred
        with Pool(15) as pool: RES = pool.map(fit, [(f, nm) for f in range(5) for nm in GRIDS], chunksize=1)
        P = {nm: {} for nm in GRIDS}; Fu = {nm: {} for nm in GRIDS}; INFO = defaultdict(dict)
        for f, nm, out, fus, pred in RES: P[nm].update(pred); Fu[nm].update(fus); INFO[nm][f] = out
        ix = {}
        def boot(v):
            if len(v) not in ix: ix[len(v)] = np.random.default_rng(0).integers(0, len(v), (10000, len(v)))
            return np.quantile(v[ix[len(v)]].mean(1), [.025, .975])
        for D in DS:
            sel = np.flatnonzero(d['ds'] == D); t = d['truth'][sel]; per = d['person'][sel]; ppl, vc = X.pmae(np.abs(np.median(t) - t), per)
            print(f'\n=== {D}: {len(sel)} windows / {len(ppl)} people')
            for nm_, Zc in [('CNN (team)', Z[0]), ('POS', Z[1]), ('CHROM', Z[2])] + [(m, ZE[m][ok]) for m in models] + [('mean of the experts', Z[3])]:
                q = BPM[Zc[sel].argmax(1)]; _, v = X.pmae(np.abs(q - t), per); ci = boot(v); print(f"   single channel, per window: {nm_:22s} MAE {v.mean():6.3f} [{ci[0]:.2f};{ci[1]:.2f}] RMSE {np.sqrt(np.mean((q - t) ** 2)):6.3f}")
            V = {}
            for nm in ('base', 'full', 'experts'):
                for kind, src in (('fused, per window', Fu[nm]), ('fused + forward-backward', P[nm])):
                    q = np.array([src[i] for i in sel.tolist()]); _, v = X.pmae(np.abs(q - t), per); V[(nm, kind)] = v; ci = boot(v); ref = V[('base', kind)]; c2 = boot(v - ref)
                    print(f"   {nm:8s} {kind:26s} MAE {v.mean():6.3f} [{ci[0]:.2f};{ci[1]:.2f}] RMSE {np.sqrt(np.mean((q - t) ** 2)):6.3f} r {np.corrcoef(q, t)[0, 1]:+.3f} | vs base {np.mean(v - ref):+.3f} [{c2[0]:+.2f};{c2[1]:+.2f}]")
            print('   weights (CNN, POS, CHROM, experts) per fold: base', [INFO['base'][f]['w'] for f in range(5)], '| full', [INFO['full'][f]['w'] for f in range(5)])
        np.savez_compressed(W / 'out/y12_public_pred.npz', rec=d['rec'], start=d['start'], ds=d['ds'], person=d['person'], truth=d['truth'], **{f'{nm}_fbq': np.array([P[nm][i] for i in range(n)]) for nm in GRIDS}, **{f'{nm}_fused': np.array([Fu[nm][i] for i in range(n)]) for nm in GRIDS})
