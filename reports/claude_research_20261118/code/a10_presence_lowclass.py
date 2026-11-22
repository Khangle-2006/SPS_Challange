"""Is there any pulse energy at the labelled rate in the low-resolution (WhatsApp-transcoded) class? Label-anchored statistic:
mean over videos of the evidence Z within +-3 bpm of the label, against the same statistic with labels permuted between the videos of the class.
Run for several extraction variants and for the CNN evidence. One-sided permutation p-value (5,000 permutations); also for the high class as a positive control."""
import sys, numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import *
V = videos(); M = meta(); rec = [v['recording'] for v in V]; lab = np.array([v['pr_bpm'] for v in V]); task = np.array([v['task'].startswith('Task') for v in V])
low = np.array([int(M[r]['width']) * int(M[r]['height']) < 1800000 for r in rec])
with np.load(OUT / 'variant_spectra.npz') as z: names = z['names'].tolist(); CH = z['chrom'].astype(float); PO = z['pos'].astype(float)
with np.load(OUT / 'cnn_evidence.npz') as z: CNN = {'cnn padding-fixed': z['pad'], 'cnn V3': z['v3']}
def stat(Z, l):
    return float(np.mean([Z[i][np.abs(GRID - l[i]) <= 3].max() for i in range(len(l))]))
rng = np.random.default_rng(0)
def test(Z, idx):
    l = lab[idx]; s = stat(Z[idx], l); null = np.array([stat(Z[idx], rng.permutation(l)) for _ in range(5000)]); return s, null.mean(), null.std(), float((np.sum(null >= s) + 1) / 5001)
E = {}
for n in ['oval_frozen', 'frame:c60', 'static:c60', 'smooth:c40', 'static:c40', 'smooth:forehead', 'smooth:cheeks', 'smooth:lower', 'smooth:fh+ck', 'static:fh+ck', 'smooth:box']:
    i = names.index(n); E[n] = (zlog(CH[i]) + zlog(PO[i])) / 2; E[n + ' [CHROM only]'] = zlog(CH[i]) if n in ('oval_frozen', 'smooth:fh+ck') else None; E[n + ' [POS only]'] = zlog(PO[i]) if n in ('oval_frozen', 'smooth:fh+ck') else None
for k, S in CNN.items(): E[k] = zlog(S)
E['mean of all classical variants'] = np.mean([v for k, v in E.items() if v is not None and '[' not in k and not k.startswith('cnn')], 0)
print(f"{'evidence':34s} | low class non-Task (n={int((low & ~task).sum())}): stat  null mean (sd)  z  p | high class rest (n={int((~low & ~task).sum())}): stat null z p")
for k, Z in E.items():
    if Z is None: continue
    a = test(Z, np.flatnonzero(low & ~task)); b = test(Z, np.flatnonzero(~low & ~task))
    print(f"{k:34s} | {a[0]:+.3f}  {a[1]:+.3f} ({a[2]:.3f})  z {(a[0]-a[1])/a[2]:+5.2f}  p {a[3]:.4f} | {b[0]:+.3f} {b[1]:+.3f} z {(b[0]-b[1])/b[2]:+5.2f} p {b[3]:.4f}", flush=True)
