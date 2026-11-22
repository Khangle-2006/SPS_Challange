"""u09 (22 Nov, Claude): are the wrong spectral peaks of the hard camera originals at codec (key-frame) frequencies?  Descriptive.
Key-frame times from ffprobe (-skip_frame nokey); artifact lines = multiples of the key-frame rate inside 42-180 bpm."""
import sys, pickle, json, subprocess
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, low, task, N, GRID = Y.lab, Y.low, Y.task, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
jobs = [J[INV.get(r, r)] for r in Y.rec]; keys = [j['key'] for j in jobs]
def kf(src):
    o = subprocess.run(['ffprobe', '-v', 'error', '-skip_frame', 'nokey', '-select_streams', 'v:0', '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', src], capture_output=True, text=True).stdout.split()
    t = np.array([float(x) for x in o if x.replace('.', '', 1).isdigit()]); return t
if __name__ == '__main__':
    Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, N), np.nan)
    for i, s in enumerate(Sp):
        if s['chrom']: pk[0, i] = GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
    for j, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
        with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
        h = np.isfinite(S).all(1); pk[2 + j, h] = GRID[S[h].argmax(1)]
    ok = np.isfinite(pk).all(0); med = np.nanmedian(pk, 0); conf = ok & ((np.abs(pk - med) <= 3).sum(0) >= 4)
    sel = np.flatnonzero(low == 0)
    with Pool(12) as p: K = dict(zip([keys[i] for i in sel], p.map(kf, [jobs[i]['source'] for i in sel])))
    rows = []
    for i in sel:
        t = K[keys[i]]; d = np.diff(t); rate = 1. / np.median(d) if len(d) else np.nan
        lines = np.array([m * rate * 60 for m in range(1, 8) if 42 <= m * rate * 60 <= 180]) if np.isfinite(rate) else np.array([])
        near = lambda f: bool(len(lines)) and np.min(np.abs(lines - f)) <= 3
        wrong = [f for f in pk[:, i] if np.isfinite(f) and abs(f - lab[i]) > 5]
        rows.append(dict(i=i, rest=not task[i], conf=bool(conf[i]), rate=rate, nk=len(t), lines=lines, wrong=wrong, wrong_near=np.mean([near(f) for f in wrong]) if wrong else np.nan, label_near=near(lab[i])))
    rates = np.array([r['rate'] for r in rows]); print('key-frame rate (Hz) of camera originals: values', np.unique(np.round(rates, 2), return_counts=True))
    for name, f in (('rest, confident', lambda r: r['rest'] and r['conf']), ('rest, hard', lambda r: r['rest'] and not r['conf']), ('Task', lambda r: not r['rest'])):
        R = [r for r in rows if f(r)]; w = [x for r in R for x in [r['wrong_near']] if np.isfinite(x)]
        cover = np.mean([min(1, len(r['lines']) * 6 / 138) for r in R])
        print(f'{name:16s} videos {len(R):3d} | wrong reader peaks near a key-frame line (+-3 bpm): {np.mean(w) if w else float("nan"):.2f} (chance ~{cover:.2f}) | label near a line {np.mean([r["label_near"] for r in R]):.2f}')
    json.dump({keys[r['i']]: dict(rate=float(r['rate']), n_key=int(r['nk'])) for r in rows}, (W / 'out/u09_keyframes.json').open('w'))
