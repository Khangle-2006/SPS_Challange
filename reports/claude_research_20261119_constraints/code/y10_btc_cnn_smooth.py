"""y10 - does a temporal pre-filter make the team's CNN usable on BTC?  (20 Nov, after x11: on MCD the same CNN goes from MAE 6.58 to 2.69
when its input is first averaged over 3 frames and brought to 10 fps - MCD is MPEG-4 with strong frame-to-frame flicker; BTC camera
originals are HEVC 'IBBBP', and on 16-18 Nov the CNN was at chance on BTC with the unfiltered 30 fps crops.)
Input = the team's stored BTC crops (36x36 FFV1, btc_zeroshot/dev/<id>/native/crop.mkv); nothing is decoded from the source videos.
  raw   : crops as stored                                   ma10 : moving average over k frames, frame rate unchanged
  b10i  : mean of k consecutive frames, one kept (<= 10 fps), then the usual linear interpolation to 30 Hz inside the clip builder
  s10i  : every k-th frame kept (<= 10 fps), then interpolation     b7i : as b10i with <= 7.5 fps        (k = smallest integer giving the rate)
CNN = padding-fixed EfficientPhys, the 5 frozen checkpoints (FROZEN.json; none has seen a BTC person), spectra averaged over checkpoints
and over the 10 s windows of the video (same windows and readout as the CHROM/POS evidence).
  run <workers> [n_videos]  -> out/y10_btc_cnn.npz          eval  -> reader_v3 with the CNN as an extra expert, paired bootstrap, controls"""
import os, sys
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import json, subprocess
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
VAR = ['raw', 'ma10', 'b10i', 's10i', 'b7i']; NCK = 5
def variants(t, valid, crop):
    fps = (len(t) - 1) / (t[-1] - t[0]); out = {'raw': (t, valid, crop)}; w = valid.astype(np.float32)
    def binned(k):
        n = len(t) // k; ww = w[:n * k].reshape(n, k); c = (crop[:n * k].reshape(n, k, 36, 36, 3) * ww[..., None, None, None]).sum(1) / np.maximum(ww.sum(1), 1)[:, None, None, None]
        return t[:n * k].reshape(n, k).mean(1), valid[:n * k].reshape(n, k).all(1), c
    k = max(1, int(np.ceil(fps / 10 - 1e-3))); k7 = max(1, int(np.ceil(fps / 7.5 - 1e-3)))
    if k > 1:
        pad = k // 2; cs = np.cumsum(np.concatenate([np.zeros((1, 36, 36, 3), np.float64), crop * w[:, None, None, None]]), 0); cw = np.concatenate([[0.], np.cumsum(w)])
        lo = np.clip(np.arange(len(t)) - pad, 0, len(t)); hi = np.clip(np.arange(len(t)) - pad + k, 0, len(t)); ma = ((cs[hi] - cs[lo]) / np.maximum(cw[hi] - cw[lo], 1)[:, None, None, None]).astype(np.float32)
        out['ma10'] = (t, valid, ma); out['b10i'] = binned(k); out['s10i'] = (t[::k], valid[::k], crop[::k])
    else: out['ma10'] = out['b10i'] = out['s10i'] = out['raw']
    out['b7i'] = binned(k7) if k7 > 1 else out['raw']; return out
def run(j):
    import x02_infer as X2
    from lib_btc import supported, window_spectrum, GRID
    p = Path(j['crop']); res = {v: (np.full(len(GRID), np.nan), 0) for v in VAR}
    with np.load(p / 'meta.npz') as z: t, valid = z['t'], z['valid'].astype(bool)
    raw = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(p / 'crop.mkv'), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True).stdout
    if len(raw) != len(t) * 36 * 36 * 3: return j['recording'], res, f'crop frames {len(raw) // 3888} != {len(t)}'
    crop = np.frombuffer(raw, np.uint8).reshape(len(t), 36, 36, 3).astype(np.float32)
    for v, (tv, vv, cv) in variants(t, valid, crop).items():
        S = []
        for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
            if not supported(t, valid, s, s + 10): continue                                   # same window list as the CHROM/POS evidence
            q = s + np.arange(301) / 30
            if q[0] < tv[0] - 1e-9 or q[-1] > tv[-1] + 1e-9:                                    # binned time stamps do not reach the window edges: clamp (<= half a bin)
                q = np.clip(q, tv[0], tv[-1])
            ix = np.searchsorted(tv, q, side='right').clip(1, len(tv) - 1); a = ((q - tv[ix - 1]) / (tv[ix] - tv[ix - 1])).astype(np.float32); clip = cv[ix - 1] * (1 - a[:, None, None, None]) + cv[ix] * a[:, None, None, None]
            sp = []
            for f in range(NCK):
                wv = X2.cnn(f, clip.astype(np.float32))
                if wv is not None:
                    w_ = np.cumsum(wv)
                    if np.isfinite(w_).all() and w_.std() > 1e-12: sp.append(window_spectrum(w_))
            if sp: S.append(np.mean(sp, 0))
        if S: res[v] = (np.mean(S, 0), len(S))
    return j['recording'], res, ''
if __name__ == '__main__':
    mode = sys.argv[1]
    from lib_btc import videos, GRID, zlog
    INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
    J = {j['recording']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}; rec = [v['recording'] for v in videos()]; real = [INV.get(r, r) for r in rec]
    if mode == 'run':
        from multiprocessing import Pool
        idx = list(range(len(rec)))
        if len(sys.argv) > 3: idx = sorted(np.random.default_rng(0).permutation(len(rec))[:int(sys.argv[3])].tolist())
        S = {v: np.full((len(rec), len(GRID)), np.nan) for v in VAR}; nw = {v: np.zeros(len(rec), int) for v in VAR}; done = 0
        with Pool(int(sys.argv[2])) as pool:
            for r, res, err in pool.imap_unordered(run, [J[real[i]] for i in idx]):
                i = real.index(r); done += 1
                for v in VAR: S[v][i], nw[v][i] = res[v]
                if err or done % 20 == 0: print(done, len(idx), r, err, flush=True)
        np.savez_compressed(W / ('out/y10_btc_cnn.npz' if len(idx) == len(rec) else 'out/y10_btc_cnn_pilot.npz'), rec=np.array(rec), **{'S_' + v: S[v] for v in VAR}, **{'n_' + v: nw[v] for v in VAR})
        print('videos with CNN evidence:', {v: int(np.isfinite(S[v]).all(1).sum()) for v in VAR})
    else:
        A = sys.argv[:]; sys.argv = ['x']
        import y06_reader_quality as Y
        z = np.load(W / ('out/y10_btc_cnn_pilot.npz' if 'pilot' in A else 'out/y10_btc_cnn.npz')); assert (z['rec'] == Y.rec).all()
        lab, per, low, task, fold, N = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N; Zn, okn, qn = Y.Zn, Y.okn, Y.qn; hi = low == 0; rm = Y.rm; WG = [0., .5, 1., 2.]
        old = np.load(OLD / 'out/cnn_evidence.npz'); assert (old['rec'] == Y.rec).all()
        ZC = {v: np.zeros((N, len(GRID))) for v in VAR}; OK = {}
        for v in VAR:
            s = z['S_' + v]; OK[v] = np.isfinite(s).all(1); ZC[v][OK[v]] = zlog(s[OK[v]])
        sub = okn & np.all([OK[v] for v in VAR], 0); print('videos with every variant:', int(sub.sum()), '/', N)
        top = lambda Z: GRID[Z.argmax(1)]; h = lambda Z, m: float(np.mean(np.abs(top(Z) - lab)[m & sub] <= 5)) if (m & sub).any() else float('nan')
        so = np.isfinite(old['pad']).all(1); Zo = np.zeros_like(Zn); Zo[so] = zlog(old['pad'][so])
        print(f"peak within 5 bpm of the label | originals rest / originals Task / low-res class\n   CHROM+POS            {h(Zn, hi & ~task):.3f} / {h(Zn, hi & task):.3f} / {h(Zn, ~hi):.3f}\n   CNN saved 13 Nov     {h(Zo, hi & ~task & so):.3f} / {h(Zo, hi & task & so):.3f} / {h(Zo, ~hi & so):.3f}")
        for v in VAR: print(f"   CNN {v:5s}            {h(ZC[v], hi & ~task):.3f} / {h(ZC[v], hi & task):.3f} / {h(ZC[v], ~hi):.3f}   | agreement of its peak with the CHROM+POS peak (5 bpm): {float(np.mean(np.abs(top(ZC[v]) - top(Zn))[sub] <= 5)):.3f}")
        def cvw(Zfun, L=None):
            """Zfun(train rows, labels) -> evidence matrix; reader_v3 otherwise unchanged."""
            L = lab if L is None else L; p = np.full(N, np.nan); info = []
            for f in sorted(set(fold)):
                tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub); Z, tag = Zfun(tr, L); info.append(tag)
                m = Y.fitG(Z[tr], L[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(m, Z[te], low[te], low[te], task[te])
            return p, info
        def mix(v, w): return (2 * Zn + w * ZC[v]) / (2 + w)                       # CHROM and POS count as two experts, the CNN as w
        def trainloss(Z, tr, L):
            m = Y.fitG(Z[tr], L[tr], per[tr], low[tr], low[tr], task[tr]); e = np.abs(Y.predG(m, Z[tr], low[tr], low[tr], task[tr]) - L[tr]); w = rm.pw(per[tr]); return float(np.sum(w * e) / w.sum())
        def nested(vs):
            def f(tr, L):
                best = (trainloss(Zn, tr, L) - .02, Zn, 'none')
                for v in vs:
                    for w in WG[1:]:
                        l = trainloss(mix(v, w), tr, L)
                        if l < best[0]: best = (l, mix(v, w), f'{v}x{w}')
                return best[1], best[2]
            return f
        S = lambda a_, b_: Y.summarize(a_[sub], lab[sub], per[sub], b_[sub]); v3 = cvw(lambda tr, L: (Zn, ''))[0]; s0 = S(v3, Y.const); print(f"\nv3 (CHROM+POS) on these videos: MAE {s0['MAE']:.3f} RMSE {s0['RMSE']:.2f} r {s0['r']:+.2f} | vs constant {s0['delta']:+.2f} [{s0['dCI'][0]:+.2f};{s0['dCI'][1]:+.2f}]")
        def show(name, p, extra=''):
            a = S(p, v3); print(f"   {name:44s} MAE {a['MAE']:6.3f} RMSE {a['RMSE']:5.2f} r {a['r']:+.2f} | vs v3 {a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}] {extra}", flush=True)
        for v in VAR:
            show(f'CNN {v} alone', cvw(lambda tr, L: (ZC[v], ''))[0]); show(f'CHROM + POS + CNN {v} (three equal experts)', cvw(lambda tr, L: (mix(v, 1.), ''))[0])
            p, info = cvw(nested([v])); show(f'CNN {v}, weight chosen on training folds', p, str(info))
        p, info = cvw(nested(VAR)); show('variant and weight chosen on training folds', p, str(info))
        rng = np.random.default_rng(1); ppl = sorted(set(per))
        for k in range(3):
            perm = dict(zip(ppl, rng.permutation(ppl))); src = {q: lab[per == q] for q in ppl}; L2 = np.array([float(np.mean(src[perm[q]])) for q in per])
            a0 = cvw(lambda tr, L: (Zn, ''), L2)[0]; a1 = cvw(nested(VAR), L2)[0]; print(f'   control, labels shuffled between people #{k}: video-level MAE v3 {np.abs(a0 - L2)[sub].mean():.2f} | with nested CNN {np.abs(a1 - L2)[sub].mean():.2f}')
