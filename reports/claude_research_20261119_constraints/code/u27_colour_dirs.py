"""u27 (22 Nov, Claude): colour directions of the pulse and of the in-band noise in BTC dev (descriptive, sets the simulation of u28).
Relative traces y_c = x_c / MA_2s(x_c) - 1 (multiplicative model: a pure intensity change is the direction [1,1,1]/sqrt3), band 0.6-3.2 Hz,
Welch cross-spectral matrices (10 s, 50%). Pulse direction: leading eigenvector of Re S(f) within +-1.5 bpm of the pulse frequency
(label-free reader consensus for confident videos; the label for the others - descriptive only). Noise: Re S(f) over 45-170 bpm without
+-6 bpm around f_p and 2 f_p. Also 18 channels (6 face regions x RGB) to see how many dimensions the noise occupies."""
import sys, pickle, json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
N, GRID, lab = Y.N, Y.GRID, Y.lab
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan)
for i, s in enumerate(Sp):
    if s['chrom']: pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
ok5 = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); conf = ok5 & ((np.abs(pk - med) <= 3).sum(0) >= 4)
prof = {}
for l in (W / 'logs/u21.jsonl').open():
    r = json.loads(l)
    if r.get('mode') == 'native' and r.get('low'): prof[r['i']] = r['profile']
grp = np.array(['low ' + str(prof.get(i, '?')) if Y.low[i] else ('Task' if Y.task[i] else ('easy' if conf[i] else 'hard')) for i in range(N)])
REG = [(slice(0, 2), slice(0, 3)), (slice(0, 2), slice(3, 6)), (slice(2, 4), slice(0, 3)), (slice(2, 4), slice(3, 6)), (slice(4, 6), slice(0, 3)), (slice(4, 6), slice(3, 6))]
def traces(k):
    with np.load(W / 'out/x01/native' / (k + '.npz')) as z: t, v, blk, fill = z['t'], z['valid'].astype(bool), z['blk'].astype(float), z['fill'].astype(float)
    if v.sum() < 300: return None
    q = np.arange(t[0], t[-1], 1 / 30.); out = []
    for rs, cs in [(slice(0, 6), slice(0, 6))] + REG:
        wgt = fill[:, rs, cs]; num = (blk[:, rs, cs] * wgt[..., None]).sum((1, 2)); den = wgt.sum((1, 2))[:, None]
        m = v & (den[:, 0] > 0); x = np.where(m[:, None], num / np.maximum(den, 1e-9), np.nan)
        out.append(np.stack([np.interp(q, t[m], x[m, c]) for c in range(3)], 1))
    return np.concatenate(out, 1)  # (T, 3 + 18)
BP = signal.butter(3, [.6, 3.2], btype='bandpass', fs=30, output='sos')
def rel(x): return signal.sosfiltfilt(BP, x / uniform_filter1d(x, 60, axis=0, mode='nearest') - 1, axis=0)
def csd(y):
    f, _ = signal.welch(y[:, 0], fs=30, nperseg=300, noverlap=150, nfft=2048); C = np.zeros((len(f), y.shape[1], y.shape[1]))
    for a in range(y.shape[1]):
        for b in range(a, y.shape[1]):
            C[:, a, b] = C[:, b, a] = np.real(signal.csd(y[:, a], y[:, b], fs=30, nperseg=300, noverlap=150, nfft=2048)[1])
    return f * 60, C
def lead(M):
    w, V = np.linalg.eigh(M); u = V[:, -1]; u = u * np.sign(u[1] if len(u) == 3 else u.sum()); return u, w[::-1] / w.sum()
ONE = np.ones(3) / np.sqrt(3); PBV = np.array([.33, .77, .53]); PBV /= np.linalg.norm(PBV)
ang = lambda a, b: np.degrees(np.arccos(min(1., abs(float(a @ b)) / np.linalg.norm(a) / np.linalg.norm(b))))
R = []
for i, k in enumerate(keys):
    x = traces(k)
    if x is None: continue
    y = rel(x); f, C = csd(y[:, :3]); fp = med[i] if conf[i] else lab[i]
    mp = np.abs(f - fp) <= 1.5; mn = (f >= 45) & (f <= 170) & (np.abs(f - fp) > 6) & (np.abs(f - 2 * fp) > 6)
    up, ep = lead(C[mp].mean(0)); un, en = lead(C[mn].mean(0))
    snr = 10 * np.log10(np.trace(C[mp].mean(0)) / np.trace(C[mn].mean(0)))
    f2, C2 = csd(y[:, 3:]); _, e18 = lead(C2[(f2 >= 45) & (f2 <= 170) & (np.abs(f2 - fp) > 6) & (np.abs(f2 - 2 * fp) > 6)].mean(0))
    R.append(dict(i=i, g=grp[i], up=up, un=un, ep=ep[0], en=en, snr=snr, e18=e18[:3], a_pn=ang(up, un), a_p1=ang(up, ONE), a_n1=ang(un, ONE), a_pbv=ang(up, PBV)))
pickle.dump(R, (W / 'out/u27_colour_dirs.pkl').open('wb'))
E = [r for r in R if r['g'] == 'easy']; upm = np.median([r['up'] for r in E], 0); upm /= np.linalg.norm(upm)
print('pulse colour direction (relative RGB), median over easy videos:', np.round(upm, 3), '| angle to [1,1,1]:', round(ang(upm, ONE), 1), '| to de Haan PBV:', round(ang(upm, PBV), 1))
q = lambda a: f'{np.median(a):6.1f} [{np.percentile(a, 25):6.1f};{np.percentile(a, 75):6.1f}]'
for g in ['easy', 'hard', 'Task', 'low High', 'low Baseline']:
    G = [r for r in R if r['g'] == g]
    if not G: continue
    print(f'{g:12s} n {len(G):3d} | in-band SNR at f_p {q([r["snr"] for r in G])} dB | pulse dir. spread vs easy median {q([ang(r["up"], upm) for r in G])} deg | pulse coherent share {q([100 * r["ep"] for r in G])}%')
    print(f'{"":12s}        | angle pulse-noise {q([r["a_pn"] for r in G])} deg | noise vs [1,1,1] {q([r["a_n1"] for r in G])} | noise rank-1 share (RGB) {q([100 * r["en"][0] for r in G])}% | 18-ch noise top-3 share {q([100 * r["e18"][0] for r in G])} / {q([100 * r["e18"][1] for r in G])} / {q([100 * r["e18"][2] for r in G])}%')
