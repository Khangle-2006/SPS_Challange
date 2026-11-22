"""x03 - BTC dev (218 videos): per-video evidence under every condition of a tag directory (x01, x06 ...) and, for tag x01, the team's
existing literal C5 crops ('c5_lit'). Evidence = window-averaged CHROM and POS spectra (10 s windows every 5 s, frozen readout, lib_btc)
of the 36x36 crop mean; plus the occlusion-aware trace (rows flagged by colour on the whole-video mean are excluded).
Usage: $PY code/x03_btc_spectra.py <workers> [tag=x01]   -> out/<tag>_btc/<cond>.npz"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
from lib_btc import avg_spectra, GRID, videos
from xcommon import detect_rows, upper_trace
R = W.parent / 'round1_execution_20261110/inference_repair_20261113'; O = W.parent / 'round1_v4_20261105/btc_zeroshot'
def one(a):
    tag, cond, j = a; G = len(GRID); r = dict(C=np.full(G, np.nan), P=np.full(G, np.nan), Co=np.full(G, np.nan), Po=np.full(G, np.nan), n=0, no=0, rows=0, vf=np.nan, have=False)
    if cond == 'c5_lit':
        p = O / 'dev' / j['drive_id'] / 'c5/meta.npz'
        if not p.exists(): return r
        with np.load(p) as z: t, valid, mean = z['t'], z['valid'].astype(bool), z['rgb_mean']; blk = None
    else:
        p = W / 'out' / tag / cond / (j['key'] + '.npz')
        if not p.exists(): return r
        with np.load(p) as z: t, valid, blk, fill = z['t'], z['valid'].astype(bool), z['blk'].astype(float), z['fill'] / 36.
        mean = blk.mean((1, 2))
    r['have'] = True; r['vf'] = float(valid.mean()); a = avg_spectra(t, valid, mean)
    if a is not None: r['C'], r['P'], r['n'] = a
    if blk is not None and valid.any():
        rows = detect_rows(blk[valid].mean(0), fill[valid].mean(0)); r['rows'] = len(rows)
        b = avg_spectra(t, valid, upper_trace(blk, rows)) if rows else a
        if b is not None: r['Co'], r['Po'], r['no'] = b
    return r
if __name__ == '__main__':
    tag = sys.argv[2] if len(sys.argv) > 2 else 'x01'; jobs = {j['recording']: j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
    INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}      # export alias -> real recording name
    rec = [v['recording'] for v in videos()]; real = [INV.get(r, r) for r in rec]; assert len(rec) == 218 and all(r in jobs for r in real)
    conds = sorted(p.name for p in (W / 'out' / tag).iterdir() if p.is_dir()) + (['c5_lit'] if tag == 'x01' else []); (W / 'out' / (tag + '_btc')).mkdir(exist_ok=True)
    with Pool(int(sys.argv[1])) as pool:
        for c in conds:
            res = pool.map(one, [(tag, c, jobs[r]) for r in real], chunksize=4)
            np.savez_compressed(W / 'out' / (tag + '_btc') / (c + '.npz'), rec=np.array(rec), **{k: np.array([x[k] for x in res]) for k in res[0]})
            print(c, 'have', sum(x['have'] for x in res), 'with windows', sum(x['n'] > 0 for x in res), 'rows flagged in', sum(x['rows'] > 0 for x in res), flush=True)
