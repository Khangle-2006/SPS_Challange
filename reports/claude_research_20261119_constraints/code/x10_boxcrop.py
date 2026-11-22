"""x10 - unmasked face-box crops for pretrained third-party models (H1 of the 20 Nov plan). No new face detection: the tight box of the
oval stored by the team's native front end is reused; the frame is cut to that box and area-resized to 72x72 (36x36 is derived from it).
BTC dev (4K/HDR files skipped unless X01_SKIP_HEAVY=0) and, on request, the public sets.
Usage: nice -n 19 $AV code/x10_boxcrop.py <workers> [BTC|UBFC2|MCD,...]   -> out/x10/<key>.npz (crop72 uint8, t, valid)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import json, time, traceback
from pathlib import Path
import numpy as np, cv2
OUT = X.W / 'out/x10'
def run(j):
    key, ds = j['key'], j['dataset']; t0 = time.time()
    if (OUT / (key + '.npz')).exists(): return key, 'skip', 0
    if os.environ.get('X01_SKIP_HEAVY', '1') == '1' and X.is_heavy(j): return key, 'not_run_heavy', 0
    try:
        if ds == 'BTC':
            with np.load(Path(j['crop']) / 'meta.npz') as z: t, boxes, valid = z['t'], z['boxes'], z['valid'].astype(bool)
            src = iter(X.BTCVideo(j['source'], t))
        else:
            with np.load(j['meta']) as z: t, boxes = z['original_t'], z['boxes']
            valid = boxes[:, 2] > boxes[:, 0]; src = X.frames_of(j, len(t))
        out = np.zeros((len(t), 72, 72, 3), np.uint8); n = 0
        for rgb in src:
            x0, y0, x1, y1 = [int(v) for v in boxes[n]]
            if valid[n] and x1 - x0 >= 8 and y1 - y0 >= 8: out[n] = cv2.resize(rgb[y0:y1, x0:x1], (72, 72), interpolation=cv2.INTER_AREA)
            n += 1
        assert n == len(t), f'{n} frames, {len(t)} timestamps'
        OUT.mkdir(parents=True, exist_ok=True); tmp = OUT / (key + '.tmp.npz'); np.savez_compressed(tmp, crop72=out, t=t, valid=valid); tmp.replace(OUT / (key + '.npz'))
        return key, 'done', round(time.time() - t0, 1)
    except Exception as e:
        return key, 'error ' + repr(e) + traceback.format_exc()[-300:], 0
if __name__ == '__main__':
    from multiprocessing import Pool
    want = (sys.argv[2] if len(sys.argv) > 2 else 'BTC').split(','); jobs = [j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] in want or j['key'] in want]
    with Pool(int(sys.argv[1])) as pool:
        for k, (key, st, s) in enumerate(pool.imap_unordered(run, jobs)): print(k + 1, len(jobs), key, st, s, flush=True)
    (X.W / 'out/x10_DONE').write_text(time.strftime('%F %T'))
