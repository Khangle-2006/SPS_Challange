"""u31 (22 Nov, Claude): u28 REDONE on the correct input. u28/u29/u30 built traces from x01 'blk', whose block means include the zero pixels
outside the oval, and then weighted them by coverage again: the traces carried mask-boundary jitter (our POS hit 0.50 on easy videos
vs 0.82 for the reader's own classical channel), so those results are void. Here every channel comes from the SAME source and windowing
as the reader's classical channel (y01): numeric_dev blocks (rgb_mean for the face; region means de-diluted as sum(block/128)/sum(fill/255)),
numeric_dev ROI 3 for the background, 10 s windows every 5 s accepted by lib_btc.supported, trend(lambda=100) + SOS1 (0.7-3 Hz),
Hann periodogram on GRID normalised to unit sum, team POS/CHROM implementations.
New front ends (first principles; multiplicative reflectance model, de Haan & van Leest 2014; Capon/MVDR beamforming; generalised
sidelobe canceller): per window w = S^-1 a / (a' S^-1 a), S = covariance of the window-normalised channels (shrinkage 0.1):
  PBVfix   3 face channels, a = p3 (blood-volume signature calibrated label-free on confident videos)
  FACEbg   3 face + 3 background channels, a = [p3, 0]  (MVDR uses the background as a noise reference)
  SC       18 region channels (6 regions x RGB), a = 1_6 (x) p3 / sqrt6
  SCbg     18 region + 3 background channels, a = [1_6 (x) p3 / sqrt6, 0]
and track-before-detect over windows (HMM forward-backward, random-walk sd 4 bpm per 5 s hop, emission exp(g z)), suffix _T.
Modes: calib | sim | real. sim = semi-synthetic (real traces, own pulse notched at label +-4 bpm and 2x, synthetic pulse with realistic
signature spread and regional amplitudes at a known frequency >= 10 bpm away). Holdout untouched."""
import sys, pickle, json
from pathlib import Path
import numpy as np
from scipy import signal, linalg
from scipy.ndimage import uniform_filter1d
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
MODE = sys.argv[1] if len(sys.argv) > 1 else 'calib'; ARGS = sys.argv[2:]; sys.argv = ['x']
import y06_reader_quality as Y
sys.path.insert(0, str(W.parent / 'claude_research_20261118/code'))
from lib_btc import supported, pos_team, chrom_team, trend, SOS1
N, GRID, lab = Y.N, Y.GRID, Y.lab; NUM = W.parent / 'w2_btc_20261116/numeric_dev'; OUT = W / 'out/u31'; FS = 30.; L = 300
REG = [(slice(0, 2), slice(0, 3)), (slice(0, 2), slice(3, 6)), (slice(2, 4), slice(0, 3)), (slice(2, 4), slice(3, 6)), (slice(4, 6), slice(0, 3)), (slice(4, 6), slice(3, 6))]
TRD = trend(L)
def consensus():
    Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan)
    for i, s in enumerate(Sp):
        if s['chrom']: pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
    for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
        with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
        h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
    ok = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); return med, ok & ((np.abs(pk - med) <= 3).sum(0) >= 4)
def load(i):
    b = np.load(NUM / 'blocks' / (Y.rec[i] + '.npz')); r = np.load(NUM / 'roi' / (Y.rec[i] + '.npz'))
    t, v = b['t'], b['valid'].astype(bool); blk = b['blocks'].astype(float) / 128.; fill = b['fill'].astype(float) / 255.
    if len(r['t']) != len(t): return None
    ch = [b['rgb_mean'].astype(float)]
    for rs, cs in REG:
        den = fill[:, rs, cs].sum((1, 2)); ch.append(np.where((den > .05)[:, None], blk[:, rs, cs].sum((1, 2)) / np.maximum(den, 1e-6)[:, None], np.nan))
    ch.append(r['rgb'][:, 3, :].astype(float)); return t, v, np.concatenate(ch, 1)      # (T, 24): face 3 | 6 regions x 3 | background 3
def windows(t, v):
    return [s for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.) if supported(t, v, s, s + 10)]
def spec(w):
    y = signal.sosfiltfilt(SOS1, w - TRD(w)); f, p = signal.periodogram(y, fs=FS, window='hann', nfft=4096, detrend=False); p = np.interp(GRID, 60 * f, p); return p / max(p.sum(), 1e-300)
def prep(x):
    c = x / x.mean(0) - 1; return np.stack([signal.sosfiltfilt(SOS1, c[:, k] - TRD(c[:, k])) for k in range(c.shape[1])], 1)
def mvdr(Yw, a, shrink=.1):
    S = Yw.T @ Yw / len(Yw); S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S)); w = np.linalg.solve(S, a); return Yw @ (w / (a @ w))
TR = {}
def tbd(Zw, g, sd=4.):
    if sd not in TR:
        d = GRID[:, None] - GRID[None]; T = np.exp(-d ** 2 / (2 * sd ** 2)); TR[sd] = T / T.sum(1, keepdims=True)
    T = TR[sd]; E = np.exp(g * (Zw - Zw.max(1, keepdims=True))); n = len(E); al = np.zeros_like(E); be = np.ones_like(E); al[0] = E[0] / E[0].sum()
    for k in range(1, n): a = (al[k - 1] @ T) * E[k]; al[k] = a / a.sum()
    for k in range(n - 2, -1, -1): b = T @ (E[k + 1] * be[k + 1]); be[k] = b / b.sum()
    po = al * be; po /= po.sum(1, keepdims=True); return po.mean(0)
def zs(p): v = np.log(p + 1e-30); return (v - v.mean()) / (v.std() + 1e-12)
def evidence_from_windows(X, starts_idx, p3, gs=(1., 2.)):
    a3 = p3; a18 = np.kron(np.ones(6), p3) / np.sqrt(6)
    acc = {m: [] for m in ('POS', 'CHROM', 'PBVfix', 'FACEbg', 'SC', 'SCbg')}
    for k0 in starts_idx:
        x = X[k0:k0 + L]
        if not np.isfinite(x).all() or x.mean(0).min() <= 0: continue
        wp, wc = pos_team(x[:, :3]), chrom_team(x[:, :3])
        if not (np.isfinite(wp).all() and wp.std() > 1e-12 and np.isfinite(wc).all() and wc.std() > 1e-12): continue
        Yw = prep(x)
        outs = dict(POS=wp, CHROM=wc, PBVfix=mvdr(Yw[:, :3], a3), FACEbg=mvdr(np.c_[Yw[:, :3], Yw[:, 21:]], np.r_[a3, np.zeros(3)]),
                    SC=mvdr(Yw[:, 3:21], a18), SCbg=mvdr(Yw[:, 3:], np.r_[a18, np.zeros(3)]))
        for m, s in outs.items(): acc[m].append(spec(s))
    if not acc['POS']: return None
    res = {m: np.mean(v, 0) for m, v in acc.items()}; res['CP'] = np.exp((zlog_(res['CHROM']) + zlog_(res['POS'])) / 2)
    for g in gs:
        for m in ('POS', 'PBVfix', 'FACEbg', 'SC', 'SCbg'): res[f'{m}_T{g:g}'] = tbd(np.array([zs(e) for e in acc[m]]), g)
    return res
def zlog_(p): v = np.log(p + 1e-30); return (v - v.mean()) / (v.std() + 1e-12)
def uniform(t, v, X):
    q = np.arange(0, t[-1] + 1e-9, 1 / FS); cols = []
    for c in range(X.shape[1]):
        m = v & np.isfinite(X[:, c]) & (X[:, c] > 0)
        if m.sum() >= 2: cols.append(np.interp(q, t[m], X[m, c]))
        else: cols.append(np.full(len(q), np.nan))
    U = np.stack(cols, 1)
    for c in range(3, 24):                       # a region or background channel with no data -> copy the face channel of that colour
        if not np.isfinite(U[:, c]).all(): U[:, c] = U[:, c % 3]
    return q, U
def calib():
    med, conf = consensus(); P = []
    for i in np.flatnonzero(conf & (Y.low == 0)):
        d = load(i)
        if d is None: continue
        t, v, X = d; q, U = uniform(t, v, X[:, :3]); y = U / uniform_filter1d(U, 60, axis=0, mode='nearest') - 1
        y = signal.sosfiltfilt(signal.butter(3, [.6, 3.2], btype='bandpass', fs=FS, output='sos'), y, axis=0); fp = med[i]
        Z = np.stack([signal.stft(y[:, c], fs=FS, nperseg=L, noverlap=L // 2, nfft=2048)[2] for c in range(3)]); f = signal.stft(y[:, 0], fs=FS, nperseg=L, noverlap=L // 2, nfft=2048)[0] * 60
        cs = lambda m: np.real(Z[:, m].reshape(3, -1) @ Z[:, m].reshape(3, -1).conj().T)
        Sp = cs(np.abs(f - fp) <= 1.5); Sn = cs((f >= 45) & (f <= 170) & (np.abs(f - fp) > 6) & (np.abs(f - 2 * fp) > 6)); Sn += 1e-3 * np.trace(Sn) / 3 * np.eye(3)
        ev, V = linalg.eigh(Sp, Sn); a = Sn @ V[:, -1]; a = a / np.linalg.norm(a); P.append(a * np.sign(a.sum()))
    P = np.array(P); p3 = np.median(P, 0); p3 /= np.linalg.norm(p3); ang = np.degrees(np.arccos(np.clip(np.abs(P @ p3), 0, 1)))
    one = np.ones(3) / np.sqrt(3); dh = np.array([.33, .77, .53]); dh /= np.linalg.norm(dh)
    print(f'calibration videos {len(P)} | p3 {np.round(p3, 3)} | spread median {np.median(ang):.1f} IQR {np.percentile(ang, 25):.1f}-{np.percentile(ang, 75):.1f} deg | to [1,1,1] {np.degrees(np.arccos(p3 @ one)):.1f} | to de Haan PBV {np.degrees(np.arccos(p3 @ dh)):.1f}')
    OUT.mkdir(parents=True, exist_ok=True); np.savez(OUT / 'calib.npz', p3=p3, P=P)
def notch_add(q, U, fl, rng, amp, p3):
    n = len(U); ma = uniform_filter1d(U, 60, axis=0, mode='nearest'); y = U / ma - 1; Fq = np.fft.rfftfreq(n, 1 / FS) * 60; Xf = np.fft.rfft(y, axis=0)
    Xf[(np.abs(Fq - fl) <= 4) | (np.abs(Fq - 2 * fl) <= 4)] = 0; y = np.fft.irfft(Xf, n, axis=0)
    while True:
        f0 = rng.uniform(55, 110)
        if abs(f0 - fl) >= 10 and abs(f0 - fl / 2) >= 6 and abs(f0 - 2 * fl) >= 10: break
    dr = np.cumsum(rng.normal(0, 1, n)); dr = 1.5 * (dr - dr.mean()) / (np.abs(dr - dr.mean()).max() + 1e-9)
    ph = 2 * np.pi * np.cumsum((f0 + dr) / 60.) / FS; s = np.cos(ph) + .4 * np.cos(2 * ph + rng.uniform(0, 2 * np.pi))
    c = np.abs(p3 + rng.normal(0, .17, 3)); c /= np.linalg.norm(c); ar = np.clip(1 + .3 * rng.standard_normal(6), .3, None)
    pat = np.r_[ar.mean() * c, np.kron(ar, c), np.zeros(3)]; y = y + amp * s[:, None] * pat[None]
    return ma * (1 + y), float(np.mean(f0 + dr))
if __name__ == '__main__':
    if MODE == 'calib': calib(); sys.exit()
    p3 = np.load(OUT / 'calib.npz')['p3']; med, conf = consensus()
    from multiprocessing import Pool
    if MODE == 'check':      # our POS/CHROM must reproduce the reader's classical channel
        Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); dif = []
        for i in range(0, N, 9):
            d = load(i)
            if d is None or not Sp[i]['chrom']: continue
            t, v, X = d; q, U = uniform(t, v, X); st = [int(round(s * FS)) for s in windows(t, v)]
            r = evidence_from_windows(X if False else U, st, p3, ())
            if r is None: continue
            ref = np.mean([w[0] for w in Sp[i]['pos']], 0); dif.append((GRID[r['POS'].argmax()] - GRID[ref.argmax()], np.corrcoef(r['POS'], ref)[0, 1]))
        d = np.array(dif); print('videos', len(d), '| POS peak equal to y01 POS peak (|diff| <= 1 bpm):', np.mean(np.abs(d[:, 0]) <= 1).round(3), '| median spectrum correlation', np.median(d[:, 1]).round(4))
    elif MODE == 'sim':
        AMPS = [1e-3, 2e-3]; pick = [i for i in range(N) if Y.low[i] == 0 and not Y.task[i]] + [i for i in range(N) if Y.task[i] and Y.low[i] == 0][::3]
        def job(i):
            d = load(i)
            if d is None: return None
            t, v, X = d; q, U = uniform(t, v, X); st = [int(round(s * FS)) for s in windows(t, v)]; st = [k for k in st if k + L <= len(U)]
            rng = np.random.default_rng(2000 + i); out = []
            for a in AMPS:
                Us, fs_ = notch_add(q, U, lab[i], rng, a, p3); ev = evidence_from_windows(Us, st, p3)
                if ev is None: continue
                out.append(dict(i=i, amp=a, conf=bool(conf[i]), **{m: bool(abs(GRID[np.argmax(c)] - fs_) <= 5) for m, c in ev.items()}, **{'err_' + m: float(min(abs(GRID[np.argmax(c)] - fs_), 40)) for m, c in ev.items()}))
            return out
        with Pool(int(ARGS[0]) if ARGS else 8) as p: R = [r for o in p.map(job, pick) if o for r in o]
        pickle.dump(R, (OUT / 'sim.pkl').open('wb'))
        ms = [m for m in R[0] if m not in ('i', 'amp', 'conf') and not m.startswith('err_')]; rng = np.random.default_rng(0)
        print('semi-synthetic, paired vs POS (pooled amplitudes; bootstrap over videos): hit share within 5 bpm of the synthetic pulse')
        best = {}
        for grp, f in (('easy noise', lambda r: r['conf']), ('hard noise', lambda r: not r['conf'])):
            Rs = [r for r in R if f(r)]; vids = sorted(set(r['i'] for r in Rs)); by = {v_: [r for r in Rs if r['i'] == v_] for v_ in vids}
            print(f'{grp} ({len(vids)} videos) POS hit {np.mean([r["POS"] for r in Rs]):.3f}')
            for m in ms:
                if m == 'POS': continue
                dd = np.array([np.mean([r[m] - r['POS'] for r in by[v_]]) for v_ in vids]); bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
                best.setdefault(m, []).append(dd.mean()); print(f'    {m:10s} {np.mean([r[m] for r in Rs]):.3f} | diff {dd.mean():+.3f} [{np.percentile(bs, 2.5):+.3f};{np.percentile(bs, 97.5):+.3f}]')
        cand = {m: np.mean(v_) for m, v_ in best.items() if m.endswith(('_T1', '_T2')) and not m.startswith('POS')}
        prim = max(cand, key=cand.get); (OUT / 'PRIMARY.txt').write_text(prim + '\n'); print('PRIMARY for the real run (best mean paired gain among the new MVDR+TBD variants):', prim)
    elif MODE == 'real':
        def job(i):
            d = load(i)
            if d is None: return i, None
            t, v, X = d; q, U = uniform(t, v, X); st = [int(round(s * FS)) for s in windows(t, v)]; st = [k for k in st if k + L <= len(U)]
            return i, evidence_from_windows(U, st, p3)
        with Pool(int(ARGS[0]) if ARGS else 8) as p: R = dict(p.map(job, range(N)))
        ms = sorted({m for v_ in R.values() if v_ for m in v_}); (OUT / 'npz').mkdir(parents=True, exist_ok=True)
        keys = np.array([str(r) for r in Y.rec])
        for m in ms:
            S = np.full((N, len(GRID)), np.nan)
            for i, v_ in R.items():
                if v_: S[i] = v_[m]
            np.savez_compressed(OUT / 'npz' / f'{m}.npz', keys=keys, S=S)
        print('saved', ms, 'videos', sum(v_ is not None for v_ in R.values()))
