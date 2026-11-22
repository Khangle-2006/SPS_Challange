"""y04 - H1 pilot (plan of 20 Nov): ME-rPPG (Wang et al., arXiv 2504.01774; released ONNX model + state, thirdparty/ME-rPPG) used ZERO-SHOT
on the crops we already have (36x36, tight face box - here with our oval mask, which the model was not trained with -> pilot only).
The model is recurrent and takes the time step dt of every frame, so the real timestamps are used. Its output is a BVP sample per frame.
  btc    : 218 BTC dev videos (team's native crop.mkv) -> window spectra with the frozen readout -> reader_v3 with ME as extra evidence
  public : UBFC2 dev + MCD native caches (rgb.npy) -> per-window spectra aligned with the 18 Nov public_spectra.npz -> window-level fusion
No training. Usage: venv_ort/bin/python code/y04_merppg_pilot.py btc|public <workers>"""
import os, sys, json, subprocess
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'; ME = W / 'thirdparty/ME-rPPG'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
_S = {}
def me_run(crops, t, valid):
    """crops uint8/float [n,36,36,3] RGB, t seconds -> BVP [n]; invalid frames reuse the last valid crop."""
    import onnxruntime as ort
    if 'm' not in _S:
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        _S['m'] = ort.InferenceSession(str(ME / 'model.onnx'), so, providers=['CPUExecutionProvider']); _S['s'] = json.loads((ME / 'state.json').read_text())
    m = _S['m']; DT = {'tensor(float)': np.float32, 'tensor(int64)': np.int64, 'tensor(int32)': np.int32, 'tensor(double)': np.float64, 'tensor(bool)': np.bool_}; ty = {i.name: DT.get(i.type, np.float32) for i in m.get_inputs()}
    state = {k: np.asarray(v, ty.get(k, np.float32)) for k, v in _S['s'].items()}; names = list(state); out = np.zeros(len(t), np.float32); last = None; dts = np.diff(t, prepend=t[0] - np.median(np.diff(t)))
    for i in range(len(t)):
        if valid[i] or last is None: last = np.asarray(crops[i], np.float32) / 255.
        r = m.run(None, {'arg_0.1': last[None, None], 'onnx::Mul_37': np.array([max(float(dts[i]), 1e-3)], ty.get('onnx::Mul_37', np.float32)), **state}); out[i] = float(np.ravel(r[0])[0]); state = dict(zip(names, r[1:]))
    return out
def btc_one(a):
    from lib_btc import supported, GRID
    import y01_btc_improve as Y
    if str(a).endswith('.npz'):                      # unmasked box crops of x10 (72x72) -> 36x36 by 2x2 averaging
        if not Path(a).exists(): return None
        with np.load(a) as z: t, valid, c72 = z['t'], z['valid'].astype(bool), z['crop72']
        crops = c72.reshape(len(c72), 36, 2, 36, 2, 3).mean((2, 4))
    else:
        crop_dir = Path(a)
        with np.load(crop_dir / 'meta.npz') as z: t, valid = z['t'], z['valid'].astype(bool)
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(crop_dir / 'crop.mkv'), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']); crops = np.frombuffer(raw, np.uint8).reshape(-1, 36, 36, 3)
    bvp = me_run(crops, t, valid); P = []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10): continue
        w = np.interp(s + np.arange(300) / 30., t, bvp)
        if np.isfinite(w).all() and w.std() > 1e-12: P.append(Y.spec2(w)[0])
    return np.mean(P, 0) if P else None
def pub_one(a):
    j, starts = a; from x05_public_score import BPM
    from scipy import signal
    from lib_btc import trend
    with np.load(j['meta']) as z: tu, valid = z['t'], z['valid'].astype(bool)
    crops = np.asarray(np.load(Path(j['crop']) / 'rgb.npy', mmap_mode='r'), np.float32); bvp = me_run(crops, tu, valid); out = []
    F = np.fft.rfftfreq(4096, 1 / 30.); band = (F >= .7) & (F <= 3.); sos = signal.butter(1, [.7, 3.], btype='bandpass', fs=30, output='sos')
    for s in starts:
        w = np.interp(s + np.arange(300) / 30., tu, bvp); y = signal.sosfiltfilt(sos, w - trend(300)(w)); out.append(signal.periodogram(y, fs=30, window='hann', nfft=4096, detrend=False)[1][band])
    return np.array(out, np.float32)
if __name__ == '__main__':
    mode, nw = sys.argv[1], int(sys.argv[2])
    if mode in ('btc', 'btcbox'):
        import y01_btc_improve as Y
        from lib_btc import zlog, GRID
        rm = Y.rm; rec, lab, per, fold, low, task = Y.rec, Y.lab, Y.per, Y.fold, Y.low, Y.task
        INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
        J = {j['recording']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
        src = [J[INV.get(r, r)]['crop'] for r in rec] if mode == 'btc' else [str(W / 'out/x10' / (J[INV.get(r, r)]['key'] + '.npz')) for r in rec]
        with Pool(nw) as pool: S = pool.map(btc_one, src, chunksize=2)
        import pickle; pickle.dump(S, (W / ('out/y04_btc_me.pkl' if mode == 'btc' else 'out/y04_btcbox_me.pkl')).open('wb'))
        Zm = np.zeros((len(rec), len(GRID))); have = np.array([s is not None for s in S])
        for i, s in enumerate(S):
            if s is not None: Zm[i] = zlog(s)
        Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Z0 = np.zeros_like(Zm)
        for i, s in enumerate(Sp):
            if s['chrom']: Z0[i] = (zlog(np.mean([w[0] for w in s['chrom']], 0)) + zlog(np.mean([w[0] for w in s['pos']], 0))) / 2
        top = lambda Z: GRID[Z.argmax(1)]; hi = low == 0; rest = ~task; sub = have
        def cv(Z, wts=(None,), lab_=None):
            """reader_v3 on the videos in `sub`; with several candidate ME weights the weight is chosen per class on the training folds (nested)."""
            L = lab if lab_ is None else lab_; p = np.full(len(L), np.nan); chosen = []
            for f in sorted(set(fold)):
                tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
                if wts == (None,): Zf = Z
                else:
                    Zf = Z0.copy(); pick = {}
                    for c in (0, 1):
                        best = (1e9, 0.)
                        for w in wts:
                            Zc = (Z0 + w * Zm) / (1 + w); itr = tr[low[tr] == c]; m_ = rm.fit(Zc[itr], L[itr], per[itr], low[itr], task[itr], 'F'); e = np.abs(rm.predict(m_, Zc[itr], low[itr], task[itr]) - L[itr]); ww = rm.pw(per[itr]); loss = float(np.sum(ww * e) / ww.sum())
                            if loss < best[0] - 1e-9: best = (loss, w)
                        pick[c] = best[1]; Zf[low == c] = ((Z0 + best[1] * Zm) / (1 + best[1]))[low == c]
                    chosen.append(pick)
                m_ = rm.fit(Zf[tr], L[tr], per[tr], low[tr], task[tr], 'F'); p[te] = rm.predict(m_, Zf[te], low[te], task[te])
            return p, chosen
        def show(name, p, ref):
            keep = lab.copy(), per.copy(); Y.lab = lab[sub]; Y.per = per[sub]; a_ = Y.show(name, p[sub], ref[sub]); Y.lab, Y.per = lab, per; return a_
        print('videos with ME output', int(have.sum()), '/ 218 -> all comparisons on these videos')
        for nm, Z in (('CHROM+POS', Z0), ('ME-rPPG', Zm)):
            print(f"  peak within 5 bpm of the label - {nm:10s}: high class rest {np.mean(np.abs(top(Z) - lab)[hi & rest & sub] <= 5):.3f} | high class Task {np.mean(np.abs(top(Z) - lab)[hi & task & sub] <= 5):.3f} | low class {np.mean(np.abs(top(Z) - lab)[~hi & sub] <= 5):.3f}")
        v3, _ = cv(Z0); show('v3 (CHROM+POS)', v3, v3)
        show('ME-rPPG alone', cv(Zm)[0], v3); show('(2*[CHROM+POS] + ME)/3  (fixed)', cv((2 * Z0 + Zm) / 3)[0], v3); show('([CHROM+POS] + ME)/2  (fixed)', cv((Z0 + Zm) / 2)[0], v3)
        pn, ch = cv(None, (0., .25, .5, 1., 2.)); show('ME weight chosen per class on training folds', pn, v3); print('    chosen weights per fold {class: w}:', ch)
        rng = np.random.default_rng(1); ppl = sorted(set(per[sub]))
        for k in range(3):
            perm = dict(zip(ppl, rng.permutation(ppl))); src_ = {q: lab[(per == q) & sub] for q in ppl}; L2 = lab.copy(); L2[sub] = [float(np.mean(src_[perm[q]])) for q in per[sub]]
            a0 = cv(Z0, lab_=L2)[0]; a1 = cv((2 * Z0 + Zm) / 3, lab_=L2)[0]; e0 = np.abs(a0 - L2)[sub]; e1 = np.abs(a1 - L2)[sub]
            print(f'   control, labels shuffled between people #{k}: video-level MAE v3 {e0.mean():.2f} | with ME {e1.mean():.2f}')
    else:
        import x05_public_score as X
        z = np.load(OLD / 'out/public_spectra.npz'); m = z['cond'] == 'native'; ds, person, foldp, recp, start, truth = z['dataset'][m], z['person'][m], z['fold'][m], z['recording'][m], z['start'][m], z['truth'][m]
        jobs = {j['recording']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] != 'BTC'}; order = sorted(set(recp.tolist()))
        with Pool(nw) as pool: RES = pool.map(pub_one, [(jobs[r], start[recp == r].tolist()) for r in order], chunksize=1)
        Sm = np.zeros((int(m.sum()), len(X.BPM)), np.float32)
        for r, a in zip(order, RES): Sm[recp == r] = a
        np.savez_compressed(W / 'out/y04_public_me.npz', me=Sm, recording=recp, start=start)
        Z = np.stack([X.zs(z[k][m]) for k in ('pad', 'pos', 'chrom')] + [X.zs(Sm)]); BPM = X.BPM
        def cvfuse(idx_exp, D):
            import itertools
            GW = sorted({tuple(np.round(np.array(w) / max(w), 3)) for w in itertools.product([0, .5, 1, 2], repeat=len(idx_exp)) if sum(w) > 0}); pred = np.zeros(int((ds == D).sum())); sel = np.flatnonzero(ds == D)
            for f in range(5):
                tr = sel[foldp[sel] != f]; te = np.flatnonzero(foldp[sel] == f); best = (1e9, None)
                for w in GW:
                    e = np.abs(BPM[np.tensordot(np.array(w), Z[idx_exp][:, tr], 1).argmax(1)] - truth[tr]); mm = X.pmae(e, person[tr])[1].mean()
                    if mm < best[0] - 1e-9: best = (mm, w)
                pred[te] = BPM[np.tensordot(np.array(best[1]), Z[idx_exp][:, sel[te]], 1).argmax(1)]
            return pred
        for D in ('UBFC2', 'MCD'):
            sel = np.flatnonzero(ds == D); t_ = truth[sel]; pp = person[sel]; ppl = sorted(set(pp)); ix = np.random.default_rng(0).integers(0, len(ppl), (10000, len(ppl)))
            base = cvfuse([0, 1, 2], D); _, vb = X.pmae(np.abs(base - t_), pp); print(f'{D}: {len(sel)} windows / {len(ppl)} people (window-level, no pooling)')
            for nm, q in (('CNN (team)', BPM[Z[0][sel].argmax(1)]), ('POS', BPM[Z[1][sel].argmax(1)]), ('CHROM', BPM[Z[2][sel].argmax(1)]), ('ME-rPPG zero-shot', BPM[Z[3][sel].argmax(1)]), ('fusion CNN+POS+CHROM', base), ('fusion + ME-rPPG', cvfuse([0, 1, 2, 3], D)), ('fusion POS+CHROM+ME (no team CNN)', cvfuse([1, 2, 3], D))):
                _, v = X.pmae(np.abs(q - t_), pp); d = v - vb; ci = np.quantile(d[ix].mean(1), [.025, .975])
                print(f"   {nm:36s} MAE {v.mean():6.3f} RMSE {np.sqrt(np.mean((q - t_) ** 2)):6.3f} | vs fusion {d.mean():+.3f} [{ci[0]:+.3f};{ci[1]:+.3f}]")
