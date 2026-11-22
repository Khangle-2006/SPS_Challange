"""u06 (22 Nov, Claude): does the head-motion channel of u04 help reader_v4 on the BTC low-resolution files?
Variant fixed before looking at any low-file result: for every video that has a motion spectrum (u04 'low' run, variant raw_best),
its standardised log-spectrum enters the reader_v4 mix with weight 1, like one more expert: Z = (2 Z_classical + sum_3 Z_expert + Z_motion) / 6;
videos without it keep the v4 mix. reader_v3 fit (prior, beta per capture class, cell offsets) on the training folds, as always.
Accept only if the low-resolution group is better than v4 with paired CI < 0 and the whole set is not worse (CI > 0).
Control: motion spectra permuted between the low files (must not help).
Usage: $PY code/u06_btc_motion_reader.py"""
import sys, pickle, json
from pathlib import Path
from collections import defaultdict
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, fold, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Z0 = np.zeros((N, len(GRID)))
for i, s in enumerate(Sp):
    if s['chrom']: Z0[i] = (Y.zlog(np.mean([w[0] for w in s['chrom']], 0)) + Y.zlog(np.mean([w[0] for w in s['pos']], 0))) / 2
V4 = ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']; ZE = []; ok = np.ones(N, bool)
for n in V4:
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); ok &= h; Zm = np.zeros_like(Z0); Zm[h] = Y.zlog(S[h]); ZE.append(Zm)
Zv4 = (2 * Z0 + sum(ZE)) / 5
Zb = np.zeros_like(Z0); hb = np.zeros(N, bool); pk = np.full(N, np.nan)
for i, k in enumerate(keys):
    p = W / 'out/u04/low' / (k + '.npz')
    if p.exists():
        with np.load(p) as z:
            if 'raw_best' in z.files: Zb[i] = Y.zlog(z['raw_best']); hb[i] = True; pk[i] = GRID[z['raw_best'].argmax()]
print('videos with a motion spectrum', int(hb.sum()), '| of them low-resolution', int((hb & (low == 1)).sum()), '| expert outputs', int(ok.sum()))
e = np.abs(pk[hb] - lab[hb]); print(f'motion peak alone on these videos: within 5 bpm {np.mean(e <= 5):.3f}, median |err| {np.median(e):.1f}')
zc = Z0[hb]; ec = np.abs(GRID[zc.argmax(1)] - lab[hb]); print(f'colour peak alone on the same videos: within 5 bpm {np.mean(ec <= 5):.3f}')
rng = np.random.default_rng(0); L = lab[hb]; print(f'chance: {np.mean([np.mean(np.abs(rng.permutation(L) - L) <= 5) for _ in range(200)]):.3f}')
sub = ok
def cv(Z):
    p = np.full(N, np.nan)
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & sub); te = np.flatnonzero((fold == f) & sub)
        m = Y.fitG(Z[tr], lab[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(m, Z[te], low[te], low[te], task[te])
    return p
def withb(Zb_):
    Z = Zv4.copy(); Z[hb] = (2 * Z0[hb] + sum(z[hb] for z in ZE) + Zb_[hb]) / 6; return Z
v4 = cv(Zv4); vb = cv(withb(Zb))
Zp = Zb.copy(); idx = np.flatnonzero(hb); Zp[idx] = Zb[rng.permutation(idx)]; vp = cv(withb(Zp))
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
for name, m in (('all', sub), ('low-resolution files', sub & (low == 1)), ('camera originals', sub & (low == 0))):
    a = S_(vb, v4, m); c = S_(vp, v4, m); k = S_(v4, Y.const, m)
    print(f"{name:22s} videos {int(m.sum()):3d} people {a['people']:3d} | const {S_(Y.const, Y.const, m)['MAE']:6.2f} | v4 {k['MAE']:6.2f} | v4+motion {a['MAE']:6.2f} ({a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}]) | permuted motion {c['MAE']:6.2f} ({c['delta']:+.2f})")
np.savez_compressed(W / 'out/u06_btc_motion_pred.npz', rec=Y.rec, lab=lab, per=per, sub=sub, v4=v4, v4_motion=vb, v4_motion_perm=vp)
