"""y11 - pretrained open-rppg models under the constraint conditions (reader_v4 = reader_v3 + a FIXED set of pretrained experts).
Zero-shot, CPU only, no holdout. The models and the per-video readout are those of y05_openrppg.py (same code path, imported).
  run <model> <cond>      -> out/y11/<cond>/<model>.npz   (S [218, grid], seconds); videos already in the file are not recomputed
        cond = heavy      : the 17 4K/HDR videos added to out/x10 on 20 Nov (the other 201 native videos are in out/y05/<model>.npz)
               split10    : native box crops, every k-th frame (k = smallest integer giving <= 10 fps), frame rate given as fps / k  [C5 split stream]
               <name>     : box crops of out/x15/<name>/ (track_c1_g25p, aidtrack_c1_g25gv01, c2_vit, c4_vga, c4_qvga, c6_us1, track_c7_blue)
  eval <model,model,...>  -> reader_v4 with exactly these models, fitted on the NATIVE training folds, applied unchanged to every
                             condition; compared with reader_v3 under the same condition, with native and with the constant.
Usage: venv_orppg/bin/python code/y11_openrppg_conditions.py run <model> <cond>      |      $PY code/y11_openrppg_conditions.py eval <models>"""
import os, sys, json, time
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '2')
os.environ.setdefault('KERAS_BACKEND', 'jax'); os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
CONDS = ['track_c1_g25p', 'aidtrack_c1_g25gv01', 'c2_vit', 'c4_vga', 'c4_qvga', 'c6_us1', 'track_c7_blue']
def spectrum_of(model, path, split=False):
    """mean window spectrum on GRID (or None) for one crop file; identical to y05.video_spectrum apart from the input path / decimation."""
    from lib_btc import supported
    import y01_btc_improve as Y
    if not path.exists(): return None, 0.
    with np.load(path) as z: t, valid, crops = z['t'], z['valid'].astype(bool), z['crop72']
    fps = (len(t) - 1) / (t[-1] - t[0])
    if split: k = max(1, int(np.ceil(fps / 10 - 1e-3))); t, valid, crops, fps = t[::k], valid[::k], crops[::k], fps / k
    t0 = time.time(); model.process_faces_tensor(np.ascontiguousarray(crops), fps=float(fps)); bvp, ts = model.bvp(raw=True); sec = time.time() - t0
    if len(bvp) < (100 if split else 300): return None, sec
    bvp = np.asarray(bvp, float); ts = np.asarray(ts, float); ts = ts - ts[0]; P = []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10) or s + 10 > ts[-1] + .2: continue
        w = np.interp(s + np.arange(300) / 30., ts, bvp)
        if np.isfinite(w).all() and w.std() > 1e-12: P.append(Y.spec2(w)[0])
    return (np.mean(P, 0) if P else None), sec
if __name__ == '__main__':
    mode = sys.argv[1]
    import y05_openrppg as Y5
    from lib_btc import GRID, zlog
    Y, keys = Y5.btc_keys()
    if mode == 'run':
        import rppg
        name, cond = sys.argv[2], sys.argv[3]; out = W / 'out/y11' / cond / (name + '.npz'); out.parent.mkdir(parents=True, exist_ok=True); S = np.full((len(keys), len(GRID)), np.nan); secs = np.zeros(len(keys)); done = np.zeros(len(keys), bool)
        if out.exists():
            with np.load(out) as z: S, secs, done = z['S'].copy(), z['seconds'].copy(), z['done'].copy()
        if cond == 'heavy':
            with np.load(W / 'out/y05' / (name + '.npz')) as z: have = np.isfinite(z['S']).all(1)
            todo = [i for i, k in enumerate(keys) if not have[i] and (W / 'out/x10' / (k + '.npz')).exists() and not done[i]]; path = lambda k: W / 'out/x10' / (k + '.npz')
        elif cond == 'split10': todo = [i for i, k in enumerate(keys) if (W / 'out/x10' / (k + '.npz')).exists() and not done[i]]; path = lambda k: W / 'out/x10' / (k + '.npz')
        else: todo = [i for i, k in enumerate(keys) if (W / 'out/x15' / cond / (k + '.npz')).exists() and not done[i]]; path = lambda k: W / 'out/x15' / cond / (k + '.npz')
        m = rppg.Model(name); print(name, cond, 'videos to do', len(todo), flush=True)
        for n, i in enumerate(todo):
            s, sec = spectrum_of(m, path(keys[i]), split=cond == 'split10'); secs[i] = sec; done[i] = True
            if s is not None: S[i] = s
            if n % 20 == 19 or n == len(todo) - 1:
                tmp = out.with_suffix('.tmp.npz'); np.savez_compressed(tmp, S=S, keys=np.array(keys), seconds=secs, done=done); tmp.replace(out); print(name, cond, n + 1, len(todo), 'mean s/video %.1f' % secs[done].mean(), flush=True)
        print(name, cond, 'done; videos with output', int(np.isfinite(S).all(1).sum()), flush=True)
    else:
        A = sys.argv[:]; sys.argv = ['x']
        import y06_reader_quality as Q
        models = A[2].split(','); lab, per, low, task, fold, N = Q.lab, Q.per, Q.low, Q.task, Q.fold, Q.N; Zn, okn, qn = Q.Zn, Q.okn, Q.qn
        def zmodel(S):
            ok = np.isfinite(S).all(1); Z = np.zeros((N, len(GRID))); Z[ok] = zlog(S[ok]); return Z, ok
        def native_models():
            out = {}
            for m in models:
                with np.load(W / 'out/y05' / (m + '.npz')) as z: S = z['S'].copy()
                p = W / 'out/y11/heavy' / (m + '.npz')
                if p.exists():
                    with np.load(p) as z: h = np.isfinite(z['S']).all(1); S[h] = z['S'][h]
                out[m] = zmodel(S)
            return out
        def cond_models(cond):
            out = {}
            for m in models:
                p = W / 'out/y11' / cond / (m + '.npz')
                if not p.exists(): return None
                with np.load(p) as z: out[m] = zmodel(z['S'])
            return out
        def mix(Zc, ZM): return (2 * Zc + sum(ZM[m][0] for m in models)) / (2 + len(models))
        NM = native_models(); okm = np.all([NM[m][1] for m in models], 0); Z4n = mix(Zn, NM); one = [('seen', 'seen')]; print('reader_v4 models:', models, '| native videos with every model:', int((okm & okn).sum()), '/', N)
        def frozen(Zte, sub, Ztr): return Q.cv(one, sub, Zte, qn, low, Ztr, qn, low, lab, {})[0]
        sub = okn & okm; S = lambda a_, b_, m_: Q.summarize(a_[m_], lab[m_], per[m_], b_[m_]); v3 = frozen(Zn, sub, Zn); v4 = frozen(Z4n, sub, Z4n); a = S(v4, v3, sub); c = S(v4, Q.const, sub); b = S(v3, Q.const, sub)
        print(f"native  n={int(sub.sum())}: v3 {b['MAE']:.3f} RMSE {b['RMSE']:.2f} r {b['r']:+.2f} | v4 {a['MAE']:.3f} [{a['CI'][0]:.2f};{a['CI'][1]:.2f}] RMSE {a['RMSE']:.2f} r {a['r']:+.2f} | v4 vs v3 {a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}] | v4 vs constant {c['delta']:+.2f} [{c['dCI'][0]:+.2f};{c['dCI'][1]:+.2f}]   (fixed model set: optimistic if the set was chosen on these videos)")
        rows = []
        for cond, tag, cname in [('split10', 'sim', 'skip10')] + [(c_, 'x09' if (W / 'out/x09_btc' / (c_ + '.npz')).exists() else 'x01', c_) for c_ in CONDS]:
            CM = cond_models(cond)
            if CM is None: print(f'{cond}: model outputs missing - skipped'); continue
            if tag == 'sim':
                import y08_fps_sim as F
                from multiprocessing import Pool
                with Pool(16) as pool: RES = pool.map(F.one, F.KEYS, chunksize=4)
                d = dict(C=np.array([r[cname][0] for r in RES]), P=np.array([r[cname][1] for r in RES]), have=np.ones(N, bool))
            else: d = Q.load(tag, cname)
            Zc, okc, _ = Q.evidence(d); m_ = sub & d['have'] & okc & np.all([CM[x][1] for x in models], 0); cov = float((okc & np.all([CM[x][1] for x in models], 0))[sub & d['have']].mean())
            p3 = frozen(Zc, m_, Zn); p4 = frozen(mix(Zc, CM), m_, Z4n); n4 = frozen(Z4n, m_, Z4n); a = S(p4, n4, m_); c = S(p4, Q.const, m_); e = S(p4, p3, m_); g = ('a' if cov >= .9 else '-') + ('b' if c['dCI'][0] <= 0 else '-') + ('c' if c['dCI'][1] < 0 else '-')
            print(f"{cond:22s} n={int(m_.sum()):3d} cover {cov:.3f} | v4 {a['MAE']:6.3f} [{a['CI'][0]:.2f};{a['CI'][1]:.2f}] RMSE {a['RMSE']:5.2f} r {a['r']:+.2f} | vs native v4 {a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}] | vs constant {c['delta']:+.2f} [{c['dCI'][0]:+.2f};{c['dCI'][1]:+.2f}] | vs v3 same condition {e['delta']:+.2f} [{e['dCI'][0]:+.2f};{e['dCI'][1]:+.2f}] | gates {g}", flush=True)
            rows.append(dict(condition=cond, videos=int(m_.sum()), coverage=cov, v4_MAE=a['MAE'], lo=a['CI'][0], hi=a['CI'][1], RMSE=a['RMSE'], r=a['r'], d_native=a['delta'], dn_lo=a['dCI'][0], dn_hi=a['dCI'][1], d_const=c['delta'], dc_lo=c['dCI'][0], dc_hi=c['dCI'][1], d_v3=e['delta'], d3_lo=e['dCI'][0], d3_hi=e['dCI'][1], gates=g))
        if rows:
            import csv
            with (W / 'out/y11_v4_conditions.csv').open('w', newline='') as fh:
                w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
