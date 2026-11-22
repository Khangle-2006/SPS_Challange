"""u08 (22 Nov, Claude): why are half of the BTC camera originals at rest hard?  Descriptive only (no fit, no selection).
Groups from u07: confident (>= 4 of 5 readers agree within 3 bpm), not confident, no expert output (4K/HDR, experts missing in out/y05)."""
import sys, pickle, json, csv
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
M = {r['recording']: r for r in csv.DictReader((W.parent / 'claude_research_20261118/out/dev_video_meta.csv').open())}
Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan); wstd = np.full(N, np.nan)
for i, s in enumerate(Sp):
    if s['chrom']:
        pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
        wstd[i] = np.std([GRID[w[0].argmax()] for w in s['pos']])
for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
    with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
    h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
ok = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); agree = (np.abs(pk - med) <= 3).sum(0); conf = ok & (agree >= 4)
v4 = np.load(W / 'out/u01_btc_nested_all.npz')['v4']
rest = ~task & (low == 0); grp = np.where(~ok, 'noexp', np.where(conf, 'conf', 'hard'))
rows = []
for i in np.flatnonzero(rest):
    k = keys[i]; m = M[Y.rec[i]]
    with np.load(W / 'out/x01/native' / (k + '.npz')) as z: t, valid, blk, box = z['t'], z['valid'], z['blk'], z['box'].astype(float)
    bw = np.median((box[valid, 2] - box[valid, 0])) if valid.any() else np.nan; bh = np.median((box[valid, 3] - box[valid, 1])) if valid.any() else np.nan
    c = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1); jit = np.median(np.linalg.norm(np.diff(c[valid], axis=0), axis=1)) / bw if valid.sum() > 2 else np.nan
    drift = np.ptp(c[valid], 0).max() / bw if valid.any() else np.nan; g = float(np.nanmean(blk[valid][..., 1])) if valid.any() else np.nan
    rows.append(dict(i=i, key=k, grp=grp[i], ab='_AB' in k, model=m['model'], hdr=m['color_transfer'] == 'arib-std-b67', rot=m['rotation'] not in ('', '0'), w=int(m['width']), h=int(m['height']), fps=float(m['avg_fps']),
                     valid=float(valid.mean()), bw=bw, bh=bh, jit=jit, drift=drift, g=g, lab=lab[i], v4=v4[i], pk=pk[:, i], wstd=wstd[i]))
for gname in ('conf', 'hard', 'noexp'):
    R = [r for r in rows if r['grp'] == gname]
    if not R: continue
    f = lambda k: np.nanmedian([r[k] for r in R])
    print(f"{gname:6s} n={len(R):2d} | patients {np.mean([r['ab'] for r in R]):.0%} | HDR {np.mean([r['hdr'] for r in R]):.0%} | rotated {np.mean([r['rot'] for r in R]):.0%} | iPhone14 {np.mean(['14' in r['model'] for r in R]):.0%}"
          f" | face box {f('bw'):.0f}x{f('bh'):.0f} px | valid {f('valid'):.3f} | jitter/frame {f('jit'):.4f} box | drift {f('drift'):.3f} box | green {f('g'):.0f} | POS window-peak std {f('wstd'):.1f} bpm | label {f('lab'):.0f}")
print('--- hard videos: label, v4, peaks (CHROM POS FacePhys ME-chunk ME-flow), face box, valid, jitter, green')
for r in sorted([r for r in rows if r['grp'] != 'conf'], key=lambda r: -abs(r['v4'] - r['lab']) if np.isfinite(r['v4']) else 0):
    print(f"  {r['key'][4:]:22s} {r['grp']:5s} {'P' if r['ab'] else 'N'} {r['model'][-9:]:>9s}{' HDR' if r['hdr'] else '    '}{' rot' if r['rot'] else '    '} | label {r['lab']:5.1f} v4 {r['v4']:6.1f} | peaks {' '.join(f'{x:5.1f}' for x in r['pk'])} | box {r['bw']:4.0f} valid {r['valid']:.2f} jit {r['jit']:.4f} drift {r['drift']:.2f} green {r['g']:5.1f} | POS-window std {r['wstd']:4.1f}")
