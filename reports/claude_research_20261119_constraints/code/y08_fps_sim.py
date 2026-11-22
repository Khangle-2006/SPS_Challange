"""y08 - SIMULATION before any pixel-level run: how much of the C5 loss on BTC is the frame rate alone?
Takes the per-frame crop means of the native re-extraction (out/x01/native, no re-encoding at all) and only changes the sampling in time:
  skip<k> : keep every k-th frame (what C5 does to a 30 fps video with k = 3: 10 fps, no anti-alias filter -> noise above 5 Hz folds back)
  bin<k>  : average k consecutive frames, then keep one (what a device that captures at 30 fps and transmits 10 fps could do)
k is chosen per video so that the output rate is <= 10 fps ('10') or <= 15 fps ('15'). Reader = reader_v3, pixel class unchanged.
  refit: fitted on the training folds of the same sampling    frozen: fitted on native
Usage: $PY code/y08_fps_sim.py <workers>"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(W / 'code')); sys.path.insert(0, str(OLD / 'code'))
A = sys.argv[:]; sys.argv = ['x']
import y06_reader_quality as Y
from lib_btc import avg_spectra, GRID
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}; KEYS = [J[INV.get(r, r)]['key'] for r in Y.rec]
MODES = ['native', 'skip15', 'bin15', 'skip10', 'bin10', 'skip7.5', 'bin7.5', 'ma10', 'ma15']      # ma<r>: moving average over the k frames that bin<r> would merge, frame rate unchanged (x13 pre-filter)
def resample(t, valid, x, mode):
    if mode == 'native': return t, valid, x
    kind, rate = mode[:4].rstrip('0123456789.'), float(mode.lstrip('skipbnma')); fps = (len(t) - 1) / (t[-1] - t[0]); k = max(1, int(np.ceil(fps / rate - 1e-3)))
    if k == 1: return t, valid, x
    if kind == 'skip': s = slice(0, None, k); return t[s], valid[s], x[s]
    if kind == 'ma':
        w = valid.astype(float); cs = np.cumsum(np.concatenate([np.zeros((1, 3)), x * w[:, None]]), 0); cw = np.concatenate([[0.], np.cumsum(w)]); lo = np.clip(np.arange(len(t)) - k // 2, 0, len(t)); hi = np.clip(np.arange(len(t)) - k // 2 + k, 0, len(t))
        return t, valid, (cs[hi] - cs[lo]) / np.maximum(cw[hi] - cw[lo], 1)[:, None]
    n = len(t) // k; tt = t[:n * k].reshape(n, k); vv = valid[:n * k].reshape(n, k); xx = x[:n * k].reshape(n, k, 3); w = vv.astype(float); ok = vv.any(1)
    return tt.mean(1), ok, (xx * w[..., None]).sum(1) / np.maximum(w.sum(1), 1)[:, None]
def one(key):
    p = W / 'out/x01/native' / (key + '.npz'); G = len(GRID); out = {m: (np.full(G, np.nan), np.full(G, np.nan), 0) for m in MODES}
    if not p.exists(): return out
    with np.load(p) as z: t, valid, mean = z['t'], z['valid'].astype(bool), z['blk'].astype(float).mean((1, 2))
    for m in MODES:
        a = avg_spectra(*resample(t, valid, mean, m))
        if a is not None: out[m] = a
    return out
if __name__ == '__main__':
    with Pool(int(A[1])) as pool: RES = pool.map(one, KEYS, chunksize=4)
    lab, per, low, task, N = Y.lab, Y.per, Y.low, Y.task, Y.N; D = {m: dict(C=np.array([r[m][0] for r in RES]), P=np.array([r[m][1] for r in RES])) for m in MODES}
    Zn, okn, qn = Y.evidence(D['native']); sub = okn.copy()
    for m in MODES: sub &= Y.evidence(D[m])[1]
    S = lambda a_, b_: Y.summarize(a_[sub], lab[sub], per[sub], b_[sub]); nat = Y.cv([('seen', 'seen')], sub, Zn, qn, low, Zn, qn, low, lab, {})[0]; hi = low == 0
    print(f'videos with evidence under every sampling: {int(sub.sum())} / {N} | constant {S(Y.const, Y.const)["MAE"]:.3f}')
    for m in MODES:
        Z, ok, q = Y.evidence(D[m]); top = GRID[Z.argmax(1)]; hit = np.abs(top - lab) <= 5
        p = Y.cv([('seen', 'seen')], sub, Z, q, low, Z, q, low, lab, {})[0]; pf = Y.cv([('seen', 'seen')], sub, Z, q, low, Zn, qn, low, lab, {})[0]; a = S(p, nat); b = S(pf, nat); c = S(p, Y.const)
        print(f"{m:8s} peak within 5 bpm: originals rest {hit[sub & hi & ~task].mean():.3f} task {hit[sub & hi & task].mean():.3f} low-res {hit[sub & ~hi].mean():.3f} | refit {a['MAE']:6.3f} RMSE {a['RMSE']:5.2f} r {a['r']:+.2f} vs native {a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}]"
              f" vs const {c['delta']:+.2f} [{c['dCI'][0]:+.2f};{c['dCI'][1]:+.2f}] | frozen {b['MAE']:6.3f} vs native {b['delta']:+.2f} [{b['dCI'][0]:+.2f};{b['dCI'][1]:+.2f}]", flush=True)
