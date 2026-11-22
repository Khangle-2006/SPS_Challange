"""Which evidence should each capture class use? Classical spectra (frozen oval; native-resolution regions) and CNN spectra (saved waveforms)
are combined as means of standardised log-spectra; the class-aware reader F is cross-validated on each combination (5 person-folds).
Exploratory table: the chosen combination must be confirmed by controls and, once, on the holdout."""
import csv, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import *
import reader_models as rm
V = videos(); M = meta(); rec = [v['recording'] for v in V]
with np.load(OUT / 'variant_spectra.npz') as z: names = z['names'].tolist(); CH = z['chrom'].astype(float); PO = z['pos'].astype(float)
with np.load(OUT / 'cnn_evidence.npz') as z: PAD = z['pad']; V3 = z['v3']
lab = np.array([v['pr_bpm'] for v in V]); fold = np.array([v['fold'] for v in V]); per = np.array([v['person'] for v in V]); task = np.array([v['task'].startswith('Task') for v in V])
low = np.array([int(M[r]['width']) * int(M[r]['height']) < 1800000 for r in rec]).astype(int)
rows = list(csv.DictReader((NUM / 'tables/round1_bayes_20261108/VIDEO_PREDICTIONS.csv').open()))
const = {r['recording']: float(r['prediction']) for r in rows if r['method'] == 'constant' and r['condition'] == 'native'}; const = np.array([const[r] for r in rec])
def zc(n):
    i = names.index(n); Z = np.zeros((len(rec), len(GRID))); ok = np.isfinite(CH[i]).all(1) & np.isfinite(PO[i]).all(1); Z[ok] = (zlog(CH[i][ok]) + zlog(PO[i][ok])) / 2; return Z
def zn(S):
    Z = np.zeros((len(rec), len(GRID))); ok = np.isfinite(S).all(1); Z[ok] = zlog(S[ok]); return Z
E = {'oval': zc('oval_frozen'), 'c60': zc('frame:c60'), 'c60s': zc('static:c60'), 'c40s': zc('static:c40'), 'cheeks': zc('smooth:cheeks'), 'fh': zc('smooth:forehead'), 'lower': zc('smooth:lower'), 'pad': zn(PAD), 'v3': zn(V3)}
def mix(keys): return np.mean([E[k] for k in keys], 0)
COMBOS = {
 'A  hi oval | low oval (reader F as of this morning)': (['oval'], ['oval']),
 'A2 hi c60 | low c60': (['c60'], ['c60']),
 'B  hi oval | low v3': (['oval'], ['v3']),
 'B2 hi oval | low pad': (['oval'], ['pad']),
 'C  hi oval | low oval+v3': (['oval'], ['oval', 'v3']),
 'C2 hi oval | low oval+pad': (['oval'], ['oval', 'pad']),
 'C3 hi oval | low oval+v3+pad': (['oval'], ['oval', 'v3', 'pad']),
 'D  hi oval+v3 | low oval+v3': (['oval', 'v3'], ['oval', 'v3']),
 'E  hi oval+c60 | low oval+c60': (['oval', 'c60'], ['oval', 'c60']),
 'E2 hi oval+c60 | low oval+c60+v3': (['oval', 'c60'], ['oval', 'c60', 'v3']),
 'E3 hi oval+c60+c60s | low oval+c60+c60s': (['oval', 'c60', 'c60s'], ['oval', 'c60', 'c60s']),
 'G  hi oval+c60+c40s+cheeks+fh+lower | same': (['oval', 'c60', 'c40s', 'cheeks', 'fh', 'lower'], ['oval', 'c60', 'c40s', 'cheeks', 'fh', 'lower']),
 'G2 hi 6 regions | low 6 regions + v3': (['oval', 'c60', 'c40s', 'cheeks', 'fh', 'lower'], ['oval', 'c60', 'c40s', 'cheeks', 'fh', 'lower', 'v3', 'v3']),
}
strata = {'hi rest': (low == 0) & ~task, 'hi Task': (low == 0) & task, 'low': (low == 1) & ~task}
def one(item):
    name, (hi, lo) = item; Z = np.where(low[:, None] == 1, mix(lo), mix(hi)); rng = np.random.default_rng(0); top = GRID[Z.argmax(1)]; row = f"{name:52s} |"
    for k, s in strata.items():
        ix = np.flatnonzero(s); h = np.mean(np.abs(top[ix] - lab[ix]) <= 5); nl = np.mean([np.mean(np.abs(top[ix] - rng.permutation(lab[ix])) <= 5) for _ in range(500)]); row += f" {h:.2f} ({nl:.2f}) |"
    res = {}
    for mode in ['v2', 'F']:
        p, fits = rm.crossval(Z, lab, per, fold, low, task, mode); s = summarize(p, lab, per, const); row += f" {mode} {s['MAE']:6.3f} ({s['delta']:+.2f} [{s['dCI'][0]:+.2f};{s['dCI'][1]:+.2f}]) r {s['r']:.2f} |"; res[mode] = p
    row += ' beta hi/low per fold: ' + ' '.join(f"{f[0][0]}/{f[0][1]}" for f in fits)
    return name, row, res['F']
if __name__ == '__main__':
    print(f"{'combination':52s} | hit within 5 bpm (null): hi rest | hi Task | low | reader v2-type and F: MAE (delta vs constant [CI]) Pearson r", flush=True)
    keep = {}
    with Pool(13) as pool:
        for name, row, p in pool.imap(one, list(COMBOS.items())): print(row, flush=True); keep[name] = p
    np.savez(OUT / 'combo_predictions.npz', keys=np.array(list(keep)), pred=np.array(list(keep.values())), rec=np.array(rec))
