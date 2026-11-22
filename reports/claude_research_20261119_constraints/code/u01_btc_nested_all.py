"""u01 (22 Nov, Claude): nested forward selection of pretrained experts for the BTC reader, now with ALL open-rppg models that have
218-video outputs in out/y05 (11 of 20 Nov + PhysFormer.rlap/.pure, RhythmMamba.rlap/.pure from Codex prompt D).
Same machinery as scratch/d06.py (reader_v3 fit, Z = (2*Z_classical + sum Z_model)/(2+k), forward selection on the TRAIN folds only,
stop when the train loss gains < 0.02 bpm, at most 4 models).
Rule fixed before this run (same as reader_v4): a model enters reader_v5 only if the nested selection picks it in >= 4/5 folds.
Usage: $PY code/u01_btc_nested_all.py [extra_dir ...]   (extra dirs hold <model>.npz with keys/S like out/y05)"""
import sys, pickle, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
EXTRA = [Path(a) for a in sys.argv[1:]]; sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, fold, N, GRID, rm = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID, Y.rm
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); Z0 = np.zeros((N, len(GRID)))
for i, s in enumerate(Sp):
    if s['chrom']: Z0[i] = (Y.zlog(np.mean([w[0] for w in s['chrom']], 0)) + Y.zlog(np.mean([w[0] for w in s['pos']], 0))) / 2
ZM = {}
for d in [W / 'out/y05'] + EXTRA:
    for f in sorted(d.glob('*.npz')):
        with np.load(f) as z:
            S = z['S']; kk = list(z['keys'])
        if kk != keys:
            pos = {k: i for i, k in enumerate(kk)}; S2 = np.full((N, S.shape[1]), np.nan)
            for i, k in enumerate(keys):
                if k in pos: S2[i] = S[pos[k]]
            S = S2
        ok = np.isfinite(S).all(1); Z = np.zeros_like(Z0); Z[ok] = Y.zlog(S[ok]); ZM[f.stem] = (Z, ok)
for n, (_, ok) in ZM.items(): print(f'  {n:34s} videos with output {int(ok.sum())}')
sub = np.all([ok for _, ok in ZM.values()], 0); hi = low == 0; print('models', len(ZM), '| videos in the intersection', int(sub.sum()), '| people', len(set(per[sub])))
V4 = ['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']
def mix(names): return (2 * Z0 + sum(ZM[n][0] for n in names)) / (2 + len(names)) if names else Z0
def trainloss(Z, tr, L):
    m = Y.fitG(Z[tr], L[tr], per[tr], low[tr], low[tr], task[tr]); e = np.abs(Y.predG(m, Z[tr], low[tr], low[tr], task[tr]) - L[tr]); w = rm.pw(per[tr]); return float(np.sum(w * e) / w.sum())
def forward(tr, L, pool, kmax=4):
    chosen = []; best = trainloss(Z0, tr, L)
    for _ in range(kmax):
        cand = [(trainloss(mix(chosen + [n]), tr, L), n) for n in pool if n not in chosen]
        if not cand: break
        l, n = min(cand)
        if l >= best - .02: break
        chosen.append(n); best = l
    return mix(chosen), tuple(chosen)
def cv(Zfun, L=None, sel=None):
    L = lab if L is None else L; s = sub if sel is None else sel; p = np.full(N, np.nan); info = []
    for f in sorted(set(fold)):
        tr = np.flatnonzero((fold != f) & s); te = np.flatnonzero((fold == f) & s); Z, tag = Zfun(tr, L); info.append(tag)
        m = Y.fitG(Z[tr], L[tr], per[tr], low[tr], low[tr], task[tr]); p[te] = Y.predG(m, Z[te], low[te], low[te], task[te])
    return p, info
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
v3 = cv(lambda tr, L: (Z0, None))[0]; v4 = cv(lambda tr, L: (mix(V4), None))[0]
ALL = list(ZM); OLD11 = [n for n in ALL if not n.startswith(('PhysFormer', 'RhythmMamba', 'PhysMamba')) and not any(str(d) in n for d in EXTRA)]
fw_all, tags_all = cv(lambda tr, L: forward(tr, L, ALL)); fw_old, tags_old = cv(lambda tr, L: forward(tr, L, OLD11))
cnt = Counter(n for t in tags_all for n in t)
print('selected per fold (all models):', tags_all); print('selection counts:', dict(cnt))
print('selected per fold (old 11):', tags_old)
for name, m in (('all', sub), ('camera originals, rest', sub & hi & ~task), ('camera originals, Task', sub & hi & task), ('low-resolution files', sub & ~hi)):
    c = S_(v3, Y.const, m); a = S_(v4, v3, m); b = S_(fw_all, v4, m); o = S_(fw_old, v4, m)
    print(f"{name:24s} videos {int(m.sum()):3d} people {c['people']:3d} | const {S_(Y.const, Y.const, m)['MAE']:6.2f} | v3 {c['MAE']:6.2f} | v4 {a['MAE']:6.2f} | nested(all) {b['MAE']:6.2f} vs v4 {b['delta']:+.2f} [{b['dCI'][0]:+.2f};{b['dCI'][1]:+.2f}] RMSE {b['RMSE']:.2f} | nested(old11) {o['MAE']:6.2f} vs v4 {o['delta']:+.2f}")
v5 = sorted(n for n, k in cnt.items() if k >= 4)
print('reader_v5 by the >=4/5 rule:', v5, '(same as v4)' if sorted(v5) == sorted(V4) else '(DIFFERENT from v4)')
if sorted(v5) != sorted(V4) and v5:
    p5 = cv(lambda tr, L: (mix(v5), None))[0]; a = S_(p5, v4, sub); print(f"  fixed v5 set on the same videos (optimistic, chosen on dev): MAE {a['MAE']:.3f} vs v4 {a['delta']:+.2f} [{a['dCI'][0]:+.2f};{a['dCI'][1]:+.2f}] RMSE {a['RMSE']:.2f} r {a.get('r', float('nan')):.3f}")
rng = np.random.default_rng(0); ppl = np.array(sorted(set(per)))
for s in range(3):
    perm = dict(zip(ppl, rng.permutation(ppl))); Lsh = lab.copy()
    for p_ in ppl:
        src = np.flatnonzero(per == perm[p_]); dst = np.flatnonzero(per == p_); Lsh[dst] = lab[src][np.arange(len(dst)) % len(src)]
    ps, _ = cv(lambda tr, L: forward(tr, L, ALL), L=Lsh); e = defaultdict(list)
    for i in np.flatnonzero(sub): e[per[i]].append(abs(ps[i] - Lsh[i]))
    c0 = defaultdict(list)
    for i in np.flatnonzero(sub): c0[per[i]].append(abs(Y.const[i] - Lsh[i]))
    print(f'  shuffled labels seed {s}: nested(all) {np.mean([np.mean(v) for v in e.values()]):.2f} | constant {np.mean([np.mean(v) for v in c0.values()]):.2f}')
np.savez_compressed(W / 'out/u01_btc_nested_all.npz', rec=Y.rec, lab=lab, per=per, sub=sub, v3=v3, v4=v4, nested_all=fw_all, nested_old=fw_old)
