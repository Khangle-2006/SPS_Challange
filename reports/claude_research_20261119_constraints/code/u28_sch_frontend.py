"""u28 (22 Nov, Claude): a new reader front end built from first principles, SIMULATED on real BTC noise before any real scoring.
Physics (multiplicative reflectance model; de Haan & van Leest 2014, Wang et al. 2017): relative traces y_rc(t) = x_rc/MA(x_rc) - 1 of
region r and colour c are  y = a_r p_c s(t) + n_rc(t);  pulse s common to all regions with a fixed blood-volume colour signature p;
nuisance n spatially correlated (u27: in-band noise is ~95% one colour direction, ~8-12 deg from intensity [1,1,1], and over 6 regions
x RGB its top three components hold ~51/20/11% of the power, i.e. it is NOT purely common-mode across regions).
Architecture 'SCH' (spatio-chromatic harmonic tracker):
  1. 18 channels (6 face regions x RGB) from the x01 6x6 block traces;
  2. MVDR / Capon filter per 10 s window, w = S^-1 a / (a' S^-1 a), S = in-band covariance (shrinkage 0.1), steering a = calibrated
     region x colour pulse pattern (label-free: generalised-eigenvector patterns at the reader-consensus frequency of confident videos,
     Haufe et al. 2014 forward model) -> the minimum-variance distortionless pulse estimate under the model;
  3. harmonic NLS evidence (Christensen & Jakobsson 2009): L(f) = P~(f) + 0.5 P~(2f), P~ = periodogram / local median noise floor;
  4. track-before-detect over windows (HMM forward-backward, random-walk HR prior sd 3 bpm per 2 s hop, emission exp(g z_w(f)));
     video evidence = mean posterior over windows (time-averaged rate, as the oximeter averages).
Baselines on the SAME traces: POS, CHROM, PBV (adaptive signature, rPPG-Toolbox style), PBV-fixed (3 ch, calibrated signature).
Modes: calib | sim | real. sim = semi-synthetic: real traces of dev videos with their own pulse notched out (label +-4 bpm and 2x),
plus a synthetic pulse (fundamental + 0.4 second harmonic, slow frequency drift) with the calibrated pattern, at a known frequency
>= 10 bpm away from the label; amplitudes swept. Labels used in sim only to place the notch. Holdout untouched."""
import sys, pickle, json, os
from pathlib import Path
import numpy as np
from scipy import signal, linalg
from scipy.ndimage import uniform_filter1d
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
MODE = sys.argv[1] if len(sys.argv) > 1 else 'calib'; ARGS = sys.argv[2:]; sys.argv = ['x']
import y06_reader_quality as Y
N, GRID, lab = Y.N, Y.GRID, Y.lab
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
FS = 30.; WIN = 300; HOP = 60; OUT = W / 'out/u28'
REG = [(slice(0, 2), slice(0, 3)), (slice(0, 2), slice(3, 6)), (slice(2, 4), slice(0, 3)), (slice(2, 4), slice(3, 6)), (slice(4, 6), slice(0, 3)), (slice(4, 6), slice(3, 6))]
BPn = signal.butter(3, [.6, 3.2], btype='bandpass', fs=FS, output='sos'); BPw = signal.butter(3, [.6, 6.5], btype='bandpass', fs=FS, output='sos')
def consensus():
    Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan)
    for i, s in enumerate(Sp):
        if s['chrom']: pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
    for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
        with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
        h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
    ok = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); return med, ok & ((np.abs(pk - med) <= 3).sum(0) >= 4)
def traces(k):
    with np.load(W / 'out/x01/native' / (k + '.npz')) as z: t, v, blk, fill = z['t'], z['valid'].astype(bool), z['blk'].astype(float), z['fill'].astype(float)
    if v.sum() < 330: return None
    q = np.arange(t[0], t[-1], 1 / FS); out = []
    for rs, cs in [(slice(0, 6), slice(0, 6))] + REG:
        wg = fill[:, rs, cs]; num = (blk[:, rs, cs] * wg[..., None]).sum((1, 2)); den = wg.sum((1, 2))[:, None]
        m = v & (den[:, 0] > 0)
        if m.sum() < 330: return None
        x = num[m] / den[m]; out.append(np.stack([np.interp(q, t[m], x[:, c]) for c in range(3)], 1))
    x = np.concatenate(out, 1); return np.clip(x, 1e-3, None)        # (T, 21): face RGB, then 6 regions x RGB
def relative(x): return x / uniform_filter1d(x, 60, axis=0, mode='nearest') - 1
def csd_band(y, fsel):
    f, _ = signal.welch(y[:, 0], fs=FS, nperseg=WIN, noverlap=WIN // 2, nfft=2048); f = f * 60; m = fsel(f); C = np.zeros((y.shape[1], y.shape[1]))
    X = np.stack([signal.stft(y[:, c], fs=FS, nperseg=WIN, noverlap=WIN // 2, nfft=2048)[2] for c in range(y.shape[1])])  # (ch, F, seg)
    Xm = X[:, m, :].reshape(y.shape[1], -1); return np.real(Xm @ Xm.conj().T) / Xm.shape[1]
def calib():
    med, conf = consensus(); P3, P18 = [], []
    for i in np.flatnonzero(conf & (Y.low == 0)):
        x = traces(keys[i])
        if x is None: continue
        y = signal.sosfiltfilt(BPn, relative(x), axis=0); fp = med[i]
        for sl, store in ((slice(0, 3), P3), (slice(3, 21), P18)):
            Sp = csd_band(y[:, sl], lambda f: np.abs(f - fp) <= 1.5); Sn = csd_band(y[:, sl], lambda f: (f >= 45) & (f <= 170) & (np.abs(f - fp) > 6) & (np.abs(f - 2 * fp) > 6))
            Sn = Sn + 1e-3 * np.trace(Sn) / len(Sn) * np.eye(len(Sn)); ev, V = linalg.eigh(Sp, Sn); wv = V[:, -1]; a = Sn @ wv; a = a / np.linalg.norm(a); a = a * np.sign(a.sum())
            store.append(a)
    P3, P18 = np.array(P3), np.array(P18); p3 = np.median(P3, 0); p3 /= np.linalg.norm(p3); p18 = np.median(P18, 0); p18 /= np.linalg.norm(p18)
    ang = lambda A, p: np.degrees(np.arccos(np.clip(np.abs(A @ p), 0, 1)))
    print('calibration videos', len(P3), '| p3 (relative RGB)', np.round(p3, 3), '| angle of video patterns to p3: median', round(float(np.median(ang(P3, p3))), 1), 'IQR', np.round(np.percentile(ang(P3, p3), [25, 75]), 1))
    print('p18 by region (forehead L/R, mid L/R, lower L/R) x RGB:\n', np.round(p18.reshape(6, 3), 3), '\nangle of video patterns to p18: median', round(float(np.median(ang(P18, p18))), 1), 'IQR', np.round(np.percentile(ang(P18, p18), [25, 75]), 1))
    print('angle p3 to [1,1,1]:', round(float(np.degrees(np.arccos(abs(p3.sum()) / np.sqrt(3)))), 1))
    OUT.mkdir(parents=True, exist_ok=True); np.savez(OUT / 'calib.npz', p3=p3, p18=p18, P3=P3, P18=P18)
# ---------- front ends (per 10 s window, input = relative traces of the window) ----------
def pos(rgb):
    l = int(1.6 * FS); n = len(rgb); H = np.zeros(n); Pm = np.array([[0, 1, -1], [-2, 1, 1]])
    for s in range(n - l + 1):
        C = rgb[s:s + l] / rgb[s:s + l].mean(0); S = C @ Pm.T; h = S[:, 0] + (S[:, 0].std() / (S[:, 1].std() + 1e-12)) * S[:, 1]; H[s:s + l] += h - h.mean()
    return H
def chrom(rgb):
    C = rgb / rgb.mean(0); X = 3 * C[:, 0] - 2 * C[:, 1]; Yc = 1.5 * C[:, 0] + C[:, 1] - 1.5 * C[:, 2]
    Xf, Yf = signal.sosfiltfilt(BPn, X), signal.sosfiltfilt(BPn, Yc); return Xf - (Xf.std() / (Yf.std() + 1e-12)) * Yf
def mvdr(yb, a, shrink=.1):
    S = yb.T @ yb / len(yb); S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S)); w = np.linalg.solve(S, a); return w / (a @ w)
def pbv_adaptive(yb):
    p = yb.std(0); p = p / np.linalg.norm(p); return mvdr(yb, p, 1e-6)
F1 = GRID / 60.; F2 = 2 * GRID / 60.; NN = np.arange(WIN); HW = np.hanning(WIN)
E1 = np.exp(-2j * np.pi * F1[:, None] * NN[None] / FS); E2 = np.exp(-2j * np.pi * F2[:, None] * NN[None] / FS)
from scipy.ndimage import median_filter
def floor(P): return median_filter(P, size=101, mode='nearest')
def win_evidence(s):
    s = (s - s.mean()) * HW; P1 = np.abs(E1 @ s) ** 2; P2 = np.abs(E2 @ s) ** 2
    n1 = floor(P1) + 1e-30; return P1 / n1, (P1 + .5 * P2) / n1
TR = None
def tbd(Zw, g, sd=3.):
    global TR
    if TR is None or TR[0] != sd:
        d = GRID[:, None] - GRID[None]; T = np.exp(-d ** 2 / (2 * sd ** 2)); TR = (sd, T / T.sum(1, keepdims=True))
    T = TR[1]; E = np.exp(g * (Zw - Zw.max(1, keepdims=True))); n = len(E); al = np.zeros_like(E); be = np.ones_like(E)
    al[0] = E[0] / E[0].sum()
    for k in range(1, n): a = (al[k - 1] @ T) * E[k]; al[k] = a / a.sum()
    for k in range(n - 2, -1, -1): b = T @ (E[k + 1] * be[k + 1]); be[k] = b / b.sum()
    po = al * be; po /= po.sum(1, keepdims=True); return po.mean(0)
def zs(v): v = np.log(v + 1e-30); return (v - v.mean()) / (v.std() + 1e-12)
def video_evidence(x, p3, p18, gs=(1.,)):
    """x: (T, 21) raw traces. Returns dict method -> evidence curve on GRID. Round 2 (after the first simulation): MVDR alone did not
    beat POS and the harmonic term at weight 0.5 hurt, while track-before-detect helped; so TBD is applied to every base front end,
    and the harmonic weight is reduced to 0.15 (about the power ratio of the PPG second harmonic)."""
    yr = relative(x); yn = signal.sosfiltfilt(BPn, yr, axis=0); yw = signal.sosfiltfilt(BPw, yr, axis=0); rgb = 1 + yr[:, :3]
    base = ('POS', 'CHROM', 'PBVfix', 'SC'); acc = {m: [] for m in base}; acch = []
    for s0 in range(0, len(x) - WIN + 1, HOP):
        sl = slice(s0, s0 + WIN); seg_n, seg_w = yn[sl], yw[sl]
        outs = dict(POS=signal.sosfiltfilt(BPw, pos(rgb[sl])), CHROM=chrom(rgb[sl]), PBVfix=seg_w[:, :3] @ mvdr(seg_n[:, :3], p3), SC=seg_w[:, 3:] @ mvdr(seg_n[:, 3:], p18))
        for m, s in outs.items():
            s = (s - s.mean()) * HW; P1 = np.abs(E1 @ s) ** 2; n1 = floor(P1) + 1e-30; acc[m].append(P1 / n1)
            if m == 'SC': P2 = np.abs(E2 @ s) ** 2; acch.append((P1 + .15 * P2) / n1)
    if not acc['SC']: return None
    res = {m: np.mean(v, 0) for m, v in acc.items()}
    for g in gs:
        for m in base: res[f'{m}_T{g:g}'] = tbd(np.array([zs(e) for e in acc[m]]), g)
        res[f'SCh_T{g:g}'] = tbd(np.array([zs(e) for e in acch]), g)
    return res
def hit(curve, f): return abs(GRID[np.argmax(curve)] - f) <= 5
def notch_add(x, fl, rng, amp):
    """real traces with the real pulse removed (label +-4 bpm and 2x label, all channels), then a synthetic pulse with the calibrated
    region x colour pattern at a frequency >= 10 bpm from the label. Returns raw-like traces and the mean synthetic frequency (bpm)."""
    yr = relative(x); n = len(yr); Fq = np.fft.rfftfreq(n, 1 / FS) * 60; X = np.fft.rfft(yr, axis=0)
    X[(np.abs(Fq - fl) <= 4) | (np.abs(Fq - 2 * fl) <= 4)] = 0; yr = np.fft.irfft(X, n, axis=0)
    while True:
        f0 = rng.uniform(55, 110)
        if abs(f0 - fl) >= 10 and abs(f0 - fl / 2) >= 6 and abs(f0 - 2 * fl) >= 10: break
    drift = np.cumsum(rng.normal(0, 1, n)); drift = 1.5 * (drift - drift.mean()) / (np.abs(drift - drift.mean()).max() + 1e-9)
    ph = 2 * np.pi * np.cumsum((f0 + drift) / 60.) / FS; s = np.cos(ph) + .4 * np.cos(2 * ph + rng.uniform(0, 2 * np.pi))
    c = P3_ + rng.normal(0, .17, 3); c = np.abs(c) / np.linalg.norm(c)            # colour signature with realistic spread (~15 deg)
    ar = np.clip(1 + .3 * rng.standard_normal(6), .3, None)                       # regional relative amplitudes
    pat = np.r_[ar.mean() * c, np.kron(ar, c)]
    yr = yr + amp * s[:, None] * pat[None]
    return 1 + yr, float(np.mean(f0 + drift))
if __name__ == '__main__':
    if MODE == 'calib': calib(); sys.exit()
    C = np.load(OUT / 'calib.npz'); P3_ = C['p3']; P18_ = np.kron(np.ones(6), P3_) / np.sqrt(6); med, conf = consensus()  # steering: same colour signature, equal relative amplitude in all regions (physics prior)
    from multiprocessing import Pool
    if MODE == 'sim':
        AMPS = [1e-3, 2e-3]; G = (1., 2., 4.)
        pick = [i for i in range(N) if Y.low[i] == 0 and not Y.task[i]] + [i for i in range(N) if Y.task[i] and Y.low[i] == 0][::3]
        def job(i):
            x = traces(keys[i])
            if x is None: return None
            rng = np.random.default_rng(1000 + i); out = []
            for a in AMPS:
                xs, fs_ = notch_add(x, lab[i], rng, a); ev = video_evidence(xs, P3_, P18_, G)
                if ev is None: continue
                out.append(dict(i=i, amp=a, conf=bool(conf[i]), **{m: bool(hit(c, fs_)) for m, c in ev.items()}, **{'err_' + m: float(abs(GRID[np.argmax(c)] - fs_)) for m, c in ev.items()}))
            return out
        with Pool(int(ARGS[0]) if ARGS else 8) as p: R = [r for o in p.map(job, pick) if o for r in o]
        pickle.dump(R, (OUT / 'sim2.pkl').open('wb'))
        ms = ['POS', 'CHROM', 'PBVfix', 'SC'] + [f'{m}_T{g:g}' for g in G for m in ('POS', 'CHROM', 'PBVfix', 'SC', 'SCh')]
        print('semi-synthetic: share of videos whose evidence peak is within 5 bpm of the synthetic pulse (real BTC noise, real pulse notched)')
        for grp, f in (('easy noise', lambda r: r['conf']), ('hard noise', lambda r: not r['conf'])):
            for a in AMPS:
                Rs = [r for r in R if r['amp'] == a and f(r)]
                print(f'{grp:10s} amp {a:.0e} n {len(Rs):3d} | ' + ' '.join(f'{m} {np.mean([r[m] for r in Rs]):.2f}' for m in ms))
    elif MODE == 'real':
        def job(i):
            x = traces(keys[i])
            if x is None: return i, None
            return i, video_evidence(x, P3_, P18_, (float(ARGS[1]) if len(ARGS) > 1 else 1.,))
        with Pool(int(ARGS[0]) if ARGS else 8) as p: R = dict(p.map(job, range(N)))
        ms = sorted({m for v in R.values() if v for m in v}); (OUT / 'npz').mkdir(parents=True, exist_ok=True)
        for m in ms:
            S = np.full((N, len(GRID)), np.nan)
            for i, v in R.items():
                if v: S[i] = v[m]
            np.savez_compressed(OUT / 'npz' / f'{m}.npz', keys=np.array(keys), S=S)
        print('saved', ms, 'videos', sum(v is not None for v in R.values()))
