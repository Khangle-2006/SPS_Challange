"""Compare extraction variants (out/variant_spectra.npz) on the 218 BTC dev videos: detection of the labelled rate by the top peak with a
permutation null per capture class, and the cross-validated readers v1 / v2 / F fed with each variant. Person-first MAE, paired bootstrap
against the constant (train-fold median, the team's saved values). Usage: python a07_variant_eval.py [n_workers] [variant ...]"""
import csv, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import *
import reader_models as rm
V = videos(); M = meta()
with np.load(OUT / 'variant_spectra.npz') as z: names = z['names'].tolist(); rec = z['rec'].tolist(); CH = z['chrom'].astype(float); PO = z['pos'].astype(float)   # in memory before the pool forks
assert rec == [v['recording'] for v in V]
lab = np.array([v['pr_bpm'] for v in V]); fold = np.array([v['fold'] for v in V]); per = np.array([v['person'] for v in V]); task = np.array([v['task'].startswith('Task') for v in V])
low = np.array([int(M[r]['width']) * int(M[r]['height']) < 1800000 for r in rec]).astype(int)
rows = list(csv.DictReader((NUM / 'tables/round1_bayes_20261108/VIDEO_PREDICTIONS.csv').open()))
const = {r['recording']: float(r['prediction']) for r in rows if r['method'] == 'constant' and r['condition'] == 'native'}; const = np.array([const[r] for r in rec])
strata = {'hi rest': (low == 0) & ~task, 'hi Task': (low == 0) & task, 'low (WhatsApp)': (low == 1) & ~task}
def evidence(n):
    i = names.index(n); C = CH[i]; P = PO[i]; ok = np.isfinite(C).all(1) & np.isfinite(P).all(1)
    Z = np.zeros((len(rec), len(GRID))); Z[ok] = (zlog(C[ok]) + zlog(P[ok])) / 2; return Z, ok
def one(n):
    rng = np.random.default_rng(0); Z, ok = evidence(n); top = GRID[Z.argmax(1)]; row = f"{n:22s} {ok.sum():3d} |"; keep = {}
    for k, s in strata.items():
        ix = np.flatnonzero(s & ok); h = np.mean(np.abs(top[ix] - lab[ix]) <= 5); nl = np.mean([np.mean(np.abs(top[ix] - rng.permutation(lab[ix])) <= 5) for _ in range(500)]); row += f" {h:.2f} ({nl:.2f}) {h - nl:+.2f} |"
    row += '|'
    for mode in ['v1', 'v2', 'F']:
        p, _ = rm.crossval(Z, lab, per, fold, low, task, mode); s = summarize(p, lab, per, const); row += f" {s['MAE']:6.3f} ({s['delta']:+.2f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]) |"; keep[mode] = p
    return n, row, keep
if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12; want = sys.argv[2:] or names
    print(f"{'variant':22s}   n | " + ' | '.join(f"{k}: hit (null) net" for k in strata) + ' || reader v1 | v2 | F : MAE (delta vs constant [CI])', flush=True)
    keys, preds = [], []
    with Pool(nw) as pool:
        for n, row, keep in pool.imap(one, want):
            print(row, flush=True)
            for m, p in keep.items(): keys.append(f'{n}|{m}'); preds.append(p)
    np.savez(OUT / 'variant_reader_predictions.npz', keys=np.array(keys), pred=np.array(preds), rec=np.array(rec))
