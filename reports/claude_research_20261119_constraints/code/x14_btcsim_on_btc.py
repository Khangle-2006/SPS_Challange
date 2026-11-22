"""x14 - validation of 'BTC-sim' on BTC itself: put BTC CAMERA ORIGINALS (dev) through the measured chain of the BTC low-resolution class and
check whether their evidence falls to what the 67 real low-resolution files show (spectral peak within 5 bpm of the label ~ chance,
median peak height ~1.5). If it does, the app re-encode alone explains that class and the simulation can be trusted for training data.
  sim_l1 : long side -> 848 px, H.264 Baseline, 30 fps, key frame every 30 frames, 1.45 Mb/s (scaled by pixels / 848x480)
  sim_l2 : long side -> 848 px, H.264 High, 20 fps (nearest frame, nominal time stamps), one key frame, 1.50 Mb/s (same scaling)
(x264 veryfast, no B-frames, 1 reference frame - as measured by scratch/d03.py.)  The decoded result goes to the unchanged front end.
4K / HDR originals are not run (decoder memory; logs/DEVIATIONS.txt).
Usage: nice -n 19 $AV code/x14_btcsim_on_btc.py <workers> [keys|pilot]   -> out/x14/<cond>/<key>.npz ; then x03_btc_spectra.py <w> x14"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import x06_c5 as C5
import x12_btcsim as S
import json, time, traceback, resource, ast
from pathlib import Path
import numpy as np
W = X.W; OUT = W / 'out/x14'; TMP = W / 'out/x14_tmp'; co = X.co
def media(j):
    md = j['metadata'] if isinstance(j['metadata'], dict) else ast.literal_eval(j['metadata']); return md.get('media', {})
def run(job):
    idx, j = job; key = j['key']; t0 = time.time()
    if all((OUT / c / (key + '.npz')).exists() for c in S.CH): return dict(key=key, state='skip')
    try:
        with np.load(Path(j['crop']) / 'meta.npz') as z: ref_t = z['t']
        vid = X.BTCVideo(j['source'], ref_t); t = vid.t; n0 = len(t); fps = (n0 - 1) / (t[-1] - t[0]); sc = 848. / max(vid.w, vid.h); Wd, H = C5.even(vid.w * sc), C5.even(vid.h * sc); sel = {}; tt = {}; fo = {}
        for c, ch in S.CH.items():
            if fps <= ch['fps'] * 1.02: sel[c] = np.arange(n0); tt[c] = t; fo[c] = fps
            else:
                q = t[0] + np.arange(int((t[-1] - t[0]) * ch['fps']) + 1) / ch['fps']; i = np.searchsorted(t, q).clip(1, n0 - 1); i = np.where(q - t[i - 1] <= t[i] - q, i - 1, i); sel[c] = i; tt[c] = q; fo[c] = ch['fps']
        TMP.mkdir(parents=True, exist_ok=True); path = {c: TMP / f'{key}__{c}.mp4' for c in S.CH}; kb = {c: S.CH[c]['bps'] * (Wd * H / S.PX) * (fo[c] / S.CH[c]['fps']) / 1000 for c in S.CH}
        enc = {c: S.encoder(path[c], Wd, H, fo[c], kb[c], S.CH[c]['args']) for c in S.CH}; ptr = {c: 0 for c in S.CH}
        for i, rgb in enumerate(vid):
            raw = None
            for c in S.CH:
                while ptr[c] < len(sel[c]) and sel[c][ptr[c]] == i:
                    if raw is None: raw = np.ascontiguousarray(co.c4_resize(np.ascontiguousarray(rgb), (Wd, H))).tobytes()
                    enc[c].stdin.write(raw); ptr[c] += 1
        info = dict(key=key, source_size=[vid.w, vid.h], size=[Wd, H], source_fps=float(fps), frames=n0)
        for c in S.CH:
            assert ptr[c] == len(sel[c]), f'{c}: wrote {ptr[c]} of {len(sel[c])} frames'
            enc[c].stdin.close(); assert enc[c].wait() == 0, 'encoder failed'; n = len(sel[c]); (OUT / c).mkdir(parents=True, exist_ok=True); pk = S.packets(path[c])
            info.update({'kbps_' + c: round(path[c].stat().st_size * 8 / (n / fo[c]) / 1000, 1), 'bytes_I_' + c: pk[0], 'bytes_P_' + c: pk[1], 'key_frames_' + c: pk[2]})
            mesh = X.Mesh(); A = dict(valid=[], blk=[], fill=[], box=[]); m = 0
            for fr in C5.decode(path[c], Wd, H):
                fr = np.ascontiguousarray(fr); good, crop, box = X.fast_crop(fr, mesh.step(fr)); m += 1
                A['valid'].append(bool(good)); A['box'].append(box); A['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32)); A['fill'].append((crop.max(-1) > 0).reshape(6, 6, 6, 6).sum((1, 3)).astype(np.uint8))
            mesh.close(); path[c].unlink(); assert m == n, f'{c}: decoded {m} of {n} frames'
            tmp = OUT / c / (key + '.tmp.npz'); np.savez_compressed(tmp, t=tt[c], valid=np.array(A['valid']), blk=np.array(A['blk']), fill=np.array(A['fill']), box=np.array(A['box'], np.int16)); tmp.replace(OUT / c / (key + '.npz')); info['valid_' + c] = float(np.mean(A['valid']))
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] == 'BTC']
    jobs = [j for j in jobs if int(media(j).get('width', 0)) * int(media(j).get('height', 0)) >= 1800000 and not X.is_heavy(dict(j, metadata=dict(media=media(j))))]          # camera originals, not 4K / HDR
    arg = sys.argv[2] if len(sys.argv) > 2 else ''
    if arg == 'pilot': jobs = [jobs[i] for i in np.random.default_rng(3).permutation(len(jobs))[:12]]
    elif arg: jobs = [j for j in jobs if j['key'] in arg.split(',')]
    log = (W / 'logs/x14_btcsim_on_btc.jsonl').open('a'); done = 0; print('jobs', len(jobs), flush=True)
    with Pool(int(sys.argv[1]), maxtasksperchild=1) as pool:
        for info in pool.imap_unordered(run, list(enumerate(jobs))):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush(); print(done, len(jobs), {k: v for k, v in info.items() if k != 'traceback'}, flush=True)
