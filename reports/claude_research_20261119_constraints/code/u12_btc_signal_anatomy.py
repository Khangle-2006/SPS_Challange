"""u12 (22 Nov, Claude): signal anatomy of the BTC camera originals at rest - WHY are half of them hard?  Descriptive, no fit.
From the stored 6x6 block traces of the oval crop (out/x01/native): oval mean RGB on a 30 Hz grid, whole-video POS, Welch PSD.
  snr_cons  : SNR (dB) around the label-free consensus HR (median of the reader peaks) + its 2nd harmonic vs the rest of 0.7-4 Hz
  snr_label : the same around the LABEL (diagnostic only)
  amp       : std of the band-passed (0.7-3 Hz) green signal, % of the mean
  lowfrac   : power 0.05-0.6 Hz / power 0.7-3 Hz of the green signal (drift, breathing, slow motion)
  steps     : abrupt brightness changes per minute (|frame-to-frame change| > 8 robust SD)
  coh       : mean pairwise correlation of the band-passed green traces of the inner blocks (pulse is spatially coherent)
  ratio_rg  : std of R/G of the band-passed signals ~ how 'pulse-like' the colour direction is (POS assumes a fixed skin tone)"""
import sys, pickle, json
from pathlib import Path
import numpy as np
from scipy import signal
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
sys.path.insert(0, str(W.parent / 'claude_research_20261118/code'))
from lib_btc import pos_team
lab, per, low, task, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan)
for i, s in enumerate(Sp):
    if s['chrom']: pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
ok = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); agree = (np.abs(pk - med) <= 3).sum(0); conf = ok & (agree >= 4)
grp = np.where(~ok, 'hdr', np.where(conf, 'easy', 'hard'))
BP = signal.butter(2, [.7, 3.], btype='bandpass', fs=30, output='sos')
def snr(f, P, f0):
    band = (f >= .7) & (f <= 4.); sig = (np.abs(f - f0) <= .1) | (np.abs(f - 2 * f0) <= .2)
    return 10 * np.log10(P[band & sig].sum() / max(P[band & ~sig].sum(), 1e-30))
rows = []
for i in np.flatnonzero(~task & (low == 0)):
    with np.load(W / 'out/x01/native' / (keys[i] + '.npz')) as z: t, valid, blk, fill = z['t'], z['valid'], z['blk'].astype(float), z['fill'].astype(float)
    t = t[valid]; blk = blk[valid]; fill = fill[valid]
    q = np.arange(t[0], t[-1], 1 / 30.); wgt = fill / np.maximum(fill.sum((1, 2), keepdims=True), 1)
    rgb = (blk * wgt[..., None]).sum((1, 2)); X = np.stack([np.interp(q, t, rgb[:, c]) for c in range(3)], 1)
    h = pos_team(X); f, P = signal.welch(h, fs=30, nperseg=300, noverlap=150)
    g = X[:, 1] / X[:, 1].mean() - 1; fg, Pg = signal.welch(g, fs=30, nperseg=300, noverlap=150)
    gb = signal.sosfiltfilt(BP, g); rb = signal.sosfiltfilt(BP, X[:, 0] / X[:, 0].mean() - 1)
    Yl = X @ np.array([.299, .587, .114]); d = np.diff(Yl); rz = np.abs(d - np.median(d)) / (1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12)
    inner = np.argwhere(fill.mean(0) >= 30)
    B = np.stack([np.interp(q, t, blk[:, a, b, 1]) for a, b in inner], 1); B = signal.sosfiltfilt(BP, B / B.mean(0) - 1, axis=0); C = np.corrcoef(B.T)
    coh = (C.sum() - len(C)) / (len(C) * (len(C) - 1)) if len(C) > 1 else np.nan
    rows.append(dict(key=keys[i][4:], grp=grp[i], lab=lab[i], cons=med[i], snr_cons=snr(f, P, med[i] / 60), snr_label=snr(f, P, lab[i] / 60), amp=100 * gb.std(),
                     lowfrac=Pg[(fg >= .05) & (fg <= .6)].sum() / Pg[(fg >= .7) & (fg <= 3)].sum(), steps=60 * np.sum(rz > 8) / (t[-1] - t[0]), coh=coh,
                     ratio_rg=np.std(rb) / max(np.std(gb), 1e-12)))
cols = ['snr_cons', 'snr_label', 'amp', 'lowfrac', 'steps', 'coh', 'ratio_rg']
for gname in ('easy', 'hard', 'hdr'):
    R = [r for r in rows if r['grp'] == gname]
    print(f"{gname:5s} n={len(R):2d} | " + ' | '.join(f"{c} {np.median([r[c] for r in R]):7.3f} [{np.quantile([r[c] for r in R], .25):.2f};{np.quantile([r[c] for r in R], .75):.2f}]" for c in cols))
print('--- hard videos')
for r in sorted([r for r in rows if r['grp'] == 'hard'], key=lambda r: r['snr_label']):
    print(f"  {r['key']:20s} label {r['lab']:5.1f} consensus {r['cons']:5.1f} | " + ' '.join(f"{c} {r[c]:7.3f}" for c in cols))
np.save(W / 'out/u12_rows.npy', rows, allow_pickle=True)
