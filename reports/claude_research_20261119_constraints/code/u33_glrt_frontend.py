"""u33 (22 Nov, Claude): adjustment after u31/u32. Diagnosis: MVDR with a FIXED pulse pattern lost to POS on real originals (hit 0.57 vs
0.82) although the oracle bound for 6 regions (u29b) is +3 to +4 dB: the real pulse pattern differs from the assumed one (signature spread
~29 deg, regional amplitudes vary), so the filter partly nulls the pulse itself (signal self-nulling, Cox 1973; Li, Stoica & Wang 2003).
Fix with a detector that does NOT assume the pattern: the generalised likelihood ratio test for a sinusoid of unknown spatial-colour
signature in Gaussian noise of covariance S (Kelly 1986; multichannel whitened periodogram):  T2(f) = x(f)^H S^-1 x(f),
x(f) = Hann DFT of the channels at f, S = in-window covariance (shrinkage 0.1). Physical prior kept: each region's RGB is projected on the
plane orthogonal to [1,1,1] (pulse is not an intensity change; POS geometry), so no intensity noise enters.
  GLRT12: 6 regions x 2 plane coordinates;   GLRT6: 3 rows (forehead / mid / lower) x 2.
Same windows, detrending, band-pass and unit-sum normalisation as the reader's classical channel. Video evidence = mean of window spectra.
Rule fixed before any real result: simulate first (u31 semi-synthetic, signature spread raised to sd 0.25), pick the better GLRT by mean
paired gain vs POS; real run: it replaces the classical channel of reader_v4, accept only if the paired CI lies below 0 (all videos)."""
import sys, pickle
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
MODE = sys.argv[1]; ARGS = sys.argv[2:]; sys.argv = ['x', 'none']
import u31_sch_frontend_v2 as U
GRID, N, FS, L = U.GRID, U.N, U.FS, U.L; OUT = W / 'out/u33'
one = np.ones(3) / np.sqrt(3); B = np.linalg.svd(np.eye(3) - np.outer(one, one))[0][:, :2]
E1 = np.exp(-2j * np.pi * (GRID / 60.)[:, None] * np.arange(L)[None] / FS); HW = np.hanning(L)
def glrt(Yc, shrink=.1):
    S = Yc.T @ Yc / len(Yc); S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S)); X = E1 @ (HW[:, None] * Yc)
    T = np.real(np.einsum('fi,ij,fj->f', X.conj(), np.linalg.inv(S), X)); return T / T.sum()
def evidence(X, starts):
    acc = {m: [] for m in ('POS', 'GLRT12', 'GLRT6')}
    for k0 in starts:
        x = X[k0:k0 + L]
        if not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        wp, wc = U.pos_team(x[:, :3]), U.chrom_team(x[:, :3])
        if not (np.isfinite(wp).all() and wp.std() > 1e-12 and np.isfinite(wc).all() and wc.std() > 1e-12): continue
        Yw = U.prep(x); R = [Yw[:, 3 + 3 * r:6 + 3 * r] @ B for r in range(6)]
        rows = [(R[0] + R[1]) / 2, (R[2] + R[3]) / 2, (R[4] + R[5]) / 2]
        acc['POS'].append(U.spec(wp)); acc['GLRT12'].append(glrt(np.concatenate(R, 1))); acc['GLRT6'].append(glrt(np.concatenate(rows, 1)))
    return {m: np.mean(v, 0) for m, v in acc.items()} if acc['POS'] else None
if __name__ == '__main__':
    p3 = np.load(U.OUT / 'calib.npz')['p3']; med, conf = U.consensus()
    from multiprocessing import Pool
    if MODE == 'sim':
        pick = [i for i in range(N) if U.Y.low[i] == 0 and not U.Y.task[i]] + [i for i in range(N) if U.Y.task[i] and U.Y.low[i] == 0][::3]
        rng0 = np.random.default_rng(0)
        def job(i):
            d = U.load(i)
            if d is None: return None
            t, v, X = d; q, Uu = U.uniform(t, v, X); st = [int(round(s * FS)) for s in U.windows(t, v)]; st = [k for k in st if k + L <= len(Uu)]
            rng = np.random.default_rng(3000 + i); out = []
            for a in (1e-3, 2e-3):
                p_ = np.abs(p3 + rng.normal(0, .25, 3)); p_ /= np.linalg.norm(p_)            # wider signature spread than u31 (real ~29 deg)
                Us, fs_ = U.notch_add(q, Uu, U.lab[i], rng, a, p_); ev = evidence(Us, st)
                if ev: out.append(dict(i=i, amp=a, conf=bool(conf[i]), **{m: bool(abs(GRID[np.argmax(c)] - fs_) <= 5) for m, c in ev.items()}))
            return out
        with Pool(int(ARGS[0])) as p: R = [r for o in p.map(job, pick) if o for r in o]
        best = {}
        for grp, f in (('easy noise', lambda r: r['conf']), ('hard noise', lambda r: not r['conf'])):
            Rs = [r for r in R if f(r)]; vids = sorted(set(r['i'] for r in Rs)); by = {v_: [r for r in Rs if r['i'] == v_] for v_ in vids}
            line = f'{grp} ({len(vids)} videos) POS {np.mean([r["POS"] for r in Rs]):.3f}'
            for m in ('GLRT12', 'GLRT6'):
                dd = np.array([np.mean([r[m] - r['POS'] for r in by[v_]]) for v_ in vids]); bs = [dd[rng0.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                best.setdefault(m, []).append(dd.mean()); line += f' | {m} {np.mean([r[m] for r in Rs]):.3f} ({dd.mean():+.3f} [{np.percentile(bs, 2.5):+.3f};{np.percentile(bs, 97.5):+.3f}])'
            print(line)
        prim = max(best, key=lambda m: np.mean(best[m])); OUT.mkdir(parents=True, exist_ok=True); (OUT / 'PRIMARY.txt').write_text(prim + '\n'); print('PRIMARY:', prim)
    elif MODE == 'real':
        def job(i):
            d = U.load(i)
            if d is None: return i, None
            t, v, X = d; q, Uu = U.uniform(t, v, X); st = [int(round(s * FS)) for s in U.windows(t, v)]; st = [k for k in st if k + L <= len(Uu)]
            return i, evidence(Uu, st)
        with Pool(int(ARGS[0])) as p: R = dict(p.map(job, range(N)))
        (OUT / 'npz').mkdir(parents=True, exist_ok=True)
        for m in ('POS', 'GLRT12', 'GLRT6'):
            S = np.full((N, len(GRID)), np.nan)
            for i, v_ in R.items():
                if v_: S[i] = v_[m]
            np.savez_compressed(OUT / 'npz' / f'{m}.npz', keys=np.array([str(r) for r in U.Y.rec]), S=S)
        print('saved')
    elif MODE == 'score':
        Y = U.Y; lab, per, low, task, fold = Y.lab, Y.per, Y.low, Y.task, Y.fold; prim = (OUT / 'PRIMARY.txt').read_text().strip()
        Uu = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); grp, sub = Uu['grp'], Uu['sub']
        Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Z0 = np.zeros((N, len(GRID)))
        for i, s in enumerate(Sp):
            if s['chrom']: Z0[i] = (Y.zlog(np.mean([w[0] for w in s['chrom']], 0)) + Y.zlog(np.mean([w[0] for w in s['pos']], 0))) / 2
        ZE = []
        for n in ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']:
            with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
            h = np.isfinite(S).all(1); Zm = np.zeros_like(Z0); Zm[h] = Y.zlog(S[h]); ZE.append(Zm)
        NEW = {}
        for m in ('GLRT12', 'GLRT6'):
            with np.load(OUT / 'npz' / f'{m}.npz') as z: S = z['S']
            ok = np.isfinite(S).all(1); Zm = np.zeros_like(Z0); Zm[ok] = Y.zlog(S[ok]); NEW[m] = (Zm, ok, S)
        use = sub & np.all([v[1] for v in NEW.values()], 0); gname = np.where(low == 1, 'low', np.where(task, 'Task', grp))
        for g in ('easy', 'hard', 'Task', 'low'):
            m = use & (gname == g); print(f'  {g:5s} n {int(m.sum()):3d} | classical {np.mean(np.abs(GRID[Z0[m].argmax(1)] - lab[m]) <= 5):.2f}' + ''.join(f' | {k} {np.mean(np.abs(GRID[v[2][m].argmax(1)] - lab[m]) <= 5):.2f}' for k, v in NEW.items()))
        def cvp(Z):
            p = np.full(N, np.nan)
            for f in sorted(set(fold)):
                tr = np.flatnonzero((fold != f) & use); te = np.flatnonzero((fold == f) & use)
                mm = Y.fitG(Z[tr], lab[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(mm, Z[te], low[te], low[te], task[te])
            return p
        v4 = cvp((2 * Z0 + sum(ZE)) / 5); P = {k: cvp((2 * NEW[k][0] + sum(ZE)) / 5) for k in NEW}; P['GLRT6 + classical'] = cvp((Z0 + NEW['GLRT6'][0] + sum(ZE)) / 5)
        S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
        for name, m in (('all', use), ('rest originals', use & ~task & (low == 0)), ('  - hard', use & (gname == 'hard')), ('Task originals', use & task & (low == 0)), ('low files', use & (low == 1))):
            print(f"{name:16s} n {int(m.sum()):3d} | v4 {S_(v4, Y.const, m)['MAE']:6.2f}" + ''.join(f" | {k} {S_(P[k], v4, m)['MAE']:6.2f} ({S_(P[k], v4, m)['delta']:+.2f} [{S_(P[k], v4, m)['dCI'][0]:+.2f};{S_(P[k], v4, m)['dCI'][1]:+.2f}])" for k in P))
        a = S_(P[prim], v4, use); print('PRIMARY', prim, 'decision:', 'ACCEPT' if a['dCI'][1] < 0 else 'REJECT')
