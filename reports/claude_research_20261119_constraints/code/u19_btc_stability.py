"""u19 (22 Nov, Claude): label-free TEMPORAL STABILITY as reader quality. At rest the true pulse persists over the ~30 s video, artifacts are
transient. Per video and reader (CHROM, POS: y01 window spectra; FacePhys: u17 waveform, same windows/operator), the SD of the window peaks;
diagnostic: is the label nearer to the most stable reader?  Descriptive (label only to measure)."""
import sys, pickle, json
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
sys.path.insert(0, str(W.parent / 'claude_research_20261118/code'))
from lib_btc import window_spectrum, supported
lab, low, task, N, GRID = Y.lab, Y.low, Y.task, Y.N, Y.GRID
d = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); grp = d['grp']
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb'))
res = {g: [] for g in ('easy', 'hard', 'low')}
for i, k in enumerate(keys):
    g = str(grp[i])
    if g not in res or not Sp[i]['chrom']: continue
    with np.load(W / 'out/x10' / (k + '.npz')) as z: t, valid = z['t'].astype(float), z['valid'].astype(bool)
    with np.load(W / 'out/u17/FacePhys.rlap' / (k + '.npz')) as z: bvp, ts = z['bvp'].astype(float), z['ts'].astype(float)
    ts = ts - ts[0]; F = []
    for s in np.arange(0, max(0., t[-1] - 10 + 1e-8), 5.):
        if not supported(t, valid, s, s + 10) or s + 10 > ts[-1] + .2: continue
        w = np.interp(s + np.arange(300) / 30., ts, bvp)
        if np.isfinite(w).all() and w.std() > 1e-12: F.append(window_spectrum(w))
    W_ = {'chrom': [w[0] for w in Sp[i]['chrom']], 'pos': [w[0] for w in Sp[i]['pos']], 'facephys': F}
    if min(len(v) for v in W_.values()) < 3: continue
    sd = {r: np.std([GRID[x.argmax()] for x in v]) for r, v in W_.items()}; peak = {r: GRID[np.mean(v, 0).argmax()] for r, v in W_.items()}
    best = min(sd, key=sd.get); res[g].append((abs(peak[best] - lab[i]), np.mean([abs(peak[r] - lab[i]) for r in peak]), abs(peak['facephys'] - lab[i]), sd['chrom'], sd['pos'], sd['facephys'], best))
for g, R in res.items():
    e = np.array([r[:6] for r in R]); bests = [r[6] for r in R]
    print(f"{g:5s} n={len(R):3d} | window-peak SD (bpm) CHROM {np.median(e[:, 3]):5.1f} POS {np.median(e[:, 4]):5.1f} FacePhys {np.median(e[:, 5]):5.1f} | most stable reader: {dict((b, bests.count(b)) for b in set(bests))}"
          f" | |err| most-stable {e[:, 0].mean():5.2f} (within 5: {np.mean(e[:, 0] <= 5):.2f}) vs FacePhys {e[:, 2].mean():5.2f} (within 5: {np.mean(e[:, 2] <= 5):.2f}) vs mean of the three {e[:, 1].mean():5.2f}")
