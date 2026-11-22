"""y14 - reader_v4 on BTC dev in the C5 split mode WITH the expert input sent as JPEG (x17, quality Q), i.e. the stream that fits the
500 kbps limit (bit rates in logs/x17_split_jpeg_Q<Q>.json). Classical branch = exact block means at 10 fps (y08 'skip10'), unchanged.
Same code path as y11 (spectrum_of with split=True: the crops are already decimated, so k = 1 and fps ~ 10 is passed to the model).
  run <model> <Q>          -> out/y11/c5_split10_jpg<Q>/<model>.npz  (same format as y11; restartable)
  eval <models> <Q>        -> reader_v4 fitted on native training folds, applied unchanged; compared with native v4, constant,
                              v3 in the same split mode, and v4 in the split mode with raw (uncompressed) 72x72 crops (y11 split10)
Usage: venv_orppg/bin/python code/y14_btc_split_jpeg.py run <model> <Q>   |   $PY code/y14_btc_split_jpeg.py eval <m1,m2,...> <Q>"""
import os, sys
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '2')
os.environ.setdefault('KERAS_BACKEND', 'jax'); os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
if __name__ == '__main__':
    mode = sys.argv[1]
    import y05_openrppg as Y5
    import y11_openrppg_conditions as Y11
    from lib_btc import GRID, zlog
    Y, keys = Y5.btc_keys()
    if mode == 'run':
        import rppg
        name, Q = sys.argv[2], int(sys.argv[3]); cond = f'c5_split10_jpg{Q}'; out = W / 'out/y11' / cond / (name + '.npz'); out.parent.mkdir(parents=True, exist_ok=True)
        S = np.full((len(keys), len(GRID)), np.nan); secs = np.zeros(len(keys)); done = np.zeros(len(keys), bool)
        if out.exists():
            with np.load(out) as z: S, secs, done = z['S'].copy(), z['seconds'].copy(), z['done'].copy()
        path = lambda k: W / 'out/x15' / cond / (k + '.npz'); todo = [i for i, k in enumerate(keys) if path(k).exists() and not done[i]]
        m = rppg.Model(name); print(name, cond, 'videos to do', len(todo), flush=True)
        for n, i in enumerate(todo):
            s, sec = Y11.spectrum_of(m, path(keys[i]), split=True); secs[i] = sec; done[i] = True
            if s is not None: S[i] = s
            if n % 20 == 19 or n == len(todo) - 1:
                tmp = out.with_suffix('.tmp.npz'); np.savez_compressed(tmp, S=S, keys=np.array(keys), seconds=secs, done=done); tmp.replace(out); print(name, cond, n + 1, len(todo), 'mean s/video %.1f' % secs[done].mean(), flush=True)
        print(name, cond, 'done; videos with output', int(np.isfinite(S).all(1).sum()), flush=True)
    else:
        A = sys.argv[:]; sys.argv = ['x']
        import y06_reader_quality as Q
        import y08_fps_sim as F
        from multiprocessing import Pool
        models = A[2].split(','); q = int(A[3]); lab, per, N = Q.lab, Q.per, Q.N; Zn, okn, qn, low = Q.Zn, Q.okn, Q.qn, Q.low
        def zmodel(S):
            ok = np.isfinite(S).all(1); Z = np.zeros((N, len(GRID))); Z[ok] = zlog(S[ok]); return Z, ok
        def load(cond):
            out = {}
            for m in models:
                with np.load(W / 'out/y11' / cond / (m + '.npz')) as z: out[m] = zmodel(z['S'])
            return out
        NM = {}
        for m in models:
            with np.load(W / 'out/y05' / (m + '.npz')) as z: S = z['S'].copy()
            with np.load(W / 'out/y11/heavy' / (m + '.npz')) as z: h = np.isfinite(z['S']).all(1); S[h] = z['S'][h]
            NM[m] = zmodel(S)
        def mix(Zc, ZM): return (2 * Zc + sum(ZM[m][0] for m in models)) / (2 + len(models))
        okm = np.all([NM[m][1] for m in models], 0); Z4n = mix(Zn, NM); sub = okn & okm; one = [('seen', 'seen')]
        def frozen(Zte, s_, Ztr): return Q.cv(one, s_, Zte, qn, low, Ztr, qn, low, lab, {})[0]
        S_ = lambda a_, b_, m_: Q.summarize(a_[m_], lab[m_], per[m_], b_[m_])
        with Pool(16) as pool: RES = pool.map(F.one, F.KEYS, chunksize=4)
        d = dict(C=np.array([r['skip10'][0] for r in RES]), P=np.array([r['skip10'][1] for r in RES]), have=np.ones(N, bool)); Zc, okc, _ = Q.evidence(d)
        JP = load(f'c5_split10_jpg{q}'); RAW = load('split10'); okj = np.all([JP[m][1] for m in models], 0); okr = np.all([RAW[m][1] for m in models], 0)
        m_ = sub & okc & okj & okr; cov = float((okc & okj)[sub].mean())
        pj = frozen(mix(Zc, JP), m_, Z4n); pr = frozen(mix(Zc, RAW), m_, Z4n); p3 = frozen(Zc, m_, Zn); n4 = frozen(Z4n, m_, Z4n)
        a = S_(pj, n4, m_); c = S_(pj, Q.const, m_); e = S_(pj, p3, m_); r = S_(pj, pr, m_); g = ('a' if cov >= .9 else '-') + ('b' if c['dCI'][0] <= 0 else '-') + ('c' if c['dCI'][1] < 0 else '-')
        print(f"c5_split10_jpg{q} n={int(m_.sum())} cover {cov:.3f} | v4 {a['MAE']:.3f} [{a['CI'][0]:.2f};{a['CI'][1]:.2f}] RMSE {a['RMSE']:.2f} r {a['r']:+.2f} | vs native v4 {a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}] | vs constant {c['delta']:+.2f} [{c['dCI'][0]:+.2f};{c['dCI'][1]:+.2f}] | vs v3 split {e['delta']:+.2f} [{e['dCI'][0]:+.2f};{e['dCI'][1]:+.2f}] | vs v4 split raw crops {r['delta']:+.2f} [{r['dCI'][0]:+.2f};{r['dCI'][1]:+.2f}] | gates {g}", flush=True)
