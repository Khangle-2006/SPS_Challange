"""y05 - H1 of the 20 Nov plan: pretrained models of the open-rppg package (MIT code; weights trained on RLAP or PURE: ME-chunk, ME-flow,
PhysMamba, RhythmMamba, PhysFormer, TSCAN, EfficientPhys, PhysNet, FacePhys) used ZERO-SHOT on the unmasked BTC face-box crops of x10.
Each model returns a BVP waveform at its own frame rate; we read it with OUR frozen readout (10 s windows every 5 s -> spectrum on the
42-180 bpm grid -> window average), so that it enters reader_v3 exactly like the CHROM/POS evidence. No training, no holdout.
  pilot                : 2 videos x all models - seconds per video and peak vs label (always run this first)
  run <model> [keys]   : all BTC dev videos that have x10 crops -> out/y05/<model>.npz
  eval                 : reader_v3 with each model alone / added to CHROM+POS / forward selection nested in the training folds
Usage: venv_orppg/bin/python code/y05_openrppg.py pilot|run|eval ..."""
import os, sys, json, time
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '2')
os.environ.setdefault('KERAS_BACKEND', 'jax'); os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
MODELS = ['ME-chunk.rlap', 'ME-chunk.pure', 'ME-flow.rlap', 'ME-flow.pure', 'FacePhys.rlap', 'PhysMamba.rlap', 'PhysMamba.pure', 'RhythmMamba.rlap', 'RhythmMamba.pure',
          'PhysFormer.rlap', 'PhysFormer.pure', 'TSCAN.rlap', 'TSCAN.pure', 'PhysNet.rlap', 'PhysNet.pure', 'EfficientPhys.rlap', 'EfficientPhys.pure']
def video_spectrum(model, key):
    """mean window spectrum on GRID (or None) for one BTC video, plus seconds spent in the model."""
    from lib_btc import supported
    import y01_btc_improve as Y
    p = W / 'out/x10' / (key + '.npz')
    if not p.exists(): return None, 0.
    with np.load(p) as z: t, valid, crops = z['t'], z['valid'].astype(bool), z['crop72']
    fps = (len(t) - 1) / (t[-1] - t[0]); t0 = time.time(); model.process_faces_tensor(np.ascontiguousarray(crops), fps=float(fps)); bvp, ts = model.bvp(raw=True); sec = time.time() - t0
    if len(bvp) < 300: return None, sec
    bvp = np.asarray(bvp, float); ts = np.asarray(ts, float); ts = ts - ts[0]; P = []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10) or s + 10 > ts[-1] + .2: continue
        w = np.interp(s + np.arange(300) / 30., ts, bvp)
        if np.isfinite(w).all() and w.std() > 1e-12: P.append(Y.spec2(w)[0])
    return (np.mean(P, 0) if P else None), sec
def btc_keys():
    import y01_btc_improve as Y
    INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
    J = {j['recording']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
    return Y, [J[INV.get(r, r)]['key'] for r in Y.rec]
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode in ('pilot', 'run'):
        import rppg
        from lib_btc import GRID
        Y, keys = btc_keys(); lab = dict(zip(keys, Y.lab))
        if mode == 'pilot':
            for name in (sys.argv[2].split(',') if len(sys.argv) > 2 else MODELS):
                try:
                    t0 = time.time(); m = rppg.Model(name); load = time.time() - t0; out = []
                    for key in ('BTC_S_N2_Resting', 'BTC_S_N1_Task1'):
                        s, sec = video_spectrum(m, key); out.append(f"{key[4:]}: {sec:.1f}s peak {None if s is None else float(GRID[s.argmax()])} (label {lab[key]:.0f})")
                    print(f'{name:20s} load {load:.1f}s input {m.input} fps {m.fps} | ' + ' | '.join(out), flush=True)
                except Exception as e: print(name, 'ERROR', repr(e)[:300], flush=True)
        else:
            name = sys.argv[2]; m = rppg.Model(name); S = np.full((len(keys), len(GRID)), np.nan); secs = []; (W / 'out/y05').mkdir(exist_ok=True)
            for i, key in enumerate(keys):
                s, sec = video_spectrum(m, key); secs.append(sec)
                if s is not None: S[i] = s
                if i % 40 == 0: print(name, i, len(keys), 'mean s/video %.1f' % np.mean(secs), flush=True)
            np.savez_compressed(W / 'out/y05' / (name + '.npz'), S=S, keys=np.array(keys), seconds=np.array(secs)); print(name, 'done; videos with output', int(np.isfinite(S).all(1).sum()), flush=True)
    else:
        import pickle
        import y01_btc_improve as Y
        from lib_btc import zlog, GRID
        rm = Y.rm; rec, lab, per, fold, low, task = Y.rec, Y.lab, Y.per, Y.fold, Y.low, Y.task
        Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Z0 = np.zeros((len(rec), len(GRID)))
        for i, s in enumerate(Sp):
            if s['chrom']: Z0[i] = (zlog(np.mean([w[0] for w in s['chrom']], 0)) + zlog(np.mean([w[0] for w in s['pos']], 0))) / 2
        ZM = {};
        for f in sorted((W / 'out/y05').glob('*.npz')):
            with np.load(f) as z: S = z['S']
            ok = np.isfinite(S).all(1); Z = np.zeros_like(Z0); Z[ok] = zlog(S[ok]); ZM[f.stem] = (Z, ok)
        sub = np.all([ok for _, ok in ZM.values()], 0); hi = low == 0; top = lambda Z: GRID[Z.argmax(1)]
        print('models', list(ZM), '| videos with every model:', int(sub.sum()), '/ 218')
        def cv(Zfun, L=None):
            """Zfun(train_idx) -> evidence matrix (may depend on the training fold: nested selection)."""
            L = lab if L is None else L; p = np.full(len(L), np.nan); info = []
            for f in sorted(set(fold)):
                tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub); Z, tag = Zfun(tr, L); info.append(tag)
                m_ = rm.fit(Z[tr], L[tr], per[tr], low[tr], task[tr], 'F'); p[te] = rm.predict(m_, Z[te], low[te], task[te])
            return p, info
        def show(name, p, ref):
            Y.lab = lab[sub]; Y.per = per[sub]; a = Y.show(name, p[sub], ref[sub]); Y.lab, Y.per = lab, per; return a
        def mix(names): return (2 * Z0 + sum(ZM[n][0] for n in names)) / (2 + len(names)) if names else Z0      # CHROM and POS count as two experts
        def trainloss(Z, tr, L):
            m_ = rm.fit(Z[tr], L[tr], per[tr], low[tr], task[tr], 'F'); e = np.abs(rm.predict(m_, Z[tr], low[tr], task[tr]) - L[tr]); w = rm.pw(per[tr]); return float(np.sum(w * e) / w.sum())
        def forward(tr, L, kmax=4):
            chosen = []; best = trainloss(Z0, tr, L)
            for _ in range(kmax):
                cand = [(trainloss(mix(chosen + [n]), tr, L), n) for n in ZM if n not in chosen]; l, n = min(cand)
                if l >= best - .02: break
                chosen.append(n); best = l
            return mix(chosen), tuple(chosen)
        v3, _ = cv(lambda tr, L: (Z0, None)); show('v3 (CHROM+POS)', v3, v3)
        for n, (Z, ok) in ZM.items():
            a = cv(lambda tr, L: (Z, None))[0]; b = cv(lambda tr, L: (mix([n]), None))[0]
            print(f"  [{n}] peak hit (5 bpm): high rest {np.mean(np.abs(top(Z) - lab)[hi & ~task & sub] <= 5):.3f} low {np.mean(np.abs(top(Z) - lab)[~hi & sub] <= 5):.3f}"); show(f'{n} alone', a, v3); show(f'CHROM+POS + {n}', b, v3)
        show('CHROM+POS + ALL models (equal weights)', cv(lambda tr, L: (mix(list(ZM)), None))[0], v3)
        pf, tags = cv(forward); show('forward selection nested in training folds', pf, v3); print('    selected per fold:', tags)
        rng = np.random.default_rng(1); ppl = sorted(set(per[sub]))
        for k in range(3):
            perm = dict(zip(ppl, rng.permutation(ppl))); src = {q: lab[(per == q) & sub] for q in ppl}; L2 = lab.copy(); L2[sub] = [float(np.mean(src[perm[q]])) for q in per[sub]]
            a0 = cv(lambda tr, L: (Z0, None), L2)[0]; a1 = cv(forward, L2)[0]; print(f'   control, labels shuffled between people #{k}: video-level MAE v3 {np.abs(a0 - L2)[sub].mean():.2f} | forward selection {np.abs(a1 - L2)[sub].mean():.2f}')
