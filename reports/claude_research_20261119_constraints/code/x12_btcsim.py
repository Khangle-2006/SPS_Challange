"""x12 - 'BTC-sim' pilot: put the public sets (reliable reference HR) through the capture chain of the BTC low-resolution class, as measured
on the 67 dev files (scratch/d03.py: 848x480 H.264 at 1.4-1.5 Mb/s, no B-frames, 1 reference frame, no camera tags = re-encode by an app):
  sim_l1 : H.264 Baseline, 30 fps, key frame every 30 frames, 1.45 Mb/s x (pixels / 848x480)          [15 + 2 dev files]
  sim_l2 : H.264 High, 20 fps (nearest source frame, nominal time stamps), one key frame only, 1.5 Mb/s x (pixels / 848x480)   [38 dev files]
The bit budget per pixel and per frame equals the BTC files'. Each encode is decoded again and given to the unchanged front end.
Question: does this chain alone remove the pulse (BTC low class: spectral peak within 5 bpm of the label in ~20% of videos = chance)?
Usage: nice -n 19 $AV code/x12_btcsim.py <workers> [keys]   -> out/x12/<cond>/<key>.npz + out/x12c crops (then x02_infer.py ... x12)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import x06_c5 as C5
import json, time, traceback, subprocess, resource
from pathlib import Path
import numpy as np
W = X.W; OUT = W / 'out/x12'; CROPS = W / 'out/x12c'; TMP = W / 'out/x12_tmp'; PX = 848 * 480
CH = dict(sim_l1=dict(fps=30., bps=1.45e6, args=['-profile:v', 'baseline', '-level', '3.1', '-g', '30', '-keyint_min', '30']),
          sim_l2=dict(fps=20., bps=1.50e6, args=['-profile:v', 'high', '-level', '3.0', '-g', '100000', '-keyint_min', '100000']))
def encoder(path, w, h, fps, kbps, args):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', f'{fps:.6f}', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast'] + args + \
          ['-b:v', f'{kbps:.0f}k', '-maxrate', f'{kbps:.0f}k', '-bufsize', f'{2 * kbps:.0f}k', '-sc_threshold', '0', '-bf', '0', '-refs', '1', '-pix_fmt', 'yuv420p', '-threads', '1', '-y', str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
def packets(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=size,flags', '-of', 'csv=p=0', str(path)], capture_output=True, text=True).stdout.split()
    s = np.array([int(l.split(',')[0]) for l in out]); k = np.array(['K' in l.split(',')[1] for l in out]); return float(s[k].mean()), float(s[~k].mean()) if (~k).any() else float('nan'), int(k.sum())
def run(job):
    idx, j = job; key, ds = j['key'], j['dataset']; t0 = time.time()
    if all((OUT / c / (key + '.npz')).exists() for c in CH): return dict(key=key, state='skip')
    try:
        with np.load(j['meta']) as z: t = z['original_t']
        n0 = len(t); fps = (n0 - 1) / (t[-1] - t[0]); Wd, H = 640, 480; sel = {}; tt = {}; fo = {}
        for c, ch in CH.items():
            if fps <= ch['fps'] * 1.02: sel[c] = np.arange(n0); tt[c] = t; fo[c] = fps
            else:
                q = t[0] + np.arange(int((t[-1] - t[0]) * ch['fps']) + 1) / ch['fps']; i = np.searchsorted(t, q).clip(1, n0 - 1); i = np.where(q - t[i - 1] <= t[i] - q, i - 1, i); sel[c] = i; tt[c] = q; fo[c] = ch['fps']
        TMP.mkdir(parents=True, exist_ok=True); path = {c: TMP / f'{key}__{c}.mp4' for c in CH}; kb = {c: CH[c]['bps'] * (Wd * H / PX) * (fo[c] / CH[c]['fps']) / 1000 for c in CH}
        enc = {c: encoder(path[c], Wd, H, fo[c], kb[c], CH[c]['args']) for c in CH}; ptr = {c: 0 for c in CH}
        for i, rgb in enumerate(X.frames_of(j, n0)):
            raw = None
            for c in CH:
                while ptr[c] < len(sel[c]) and sel[c][ptr[c]] == i:
                    if raw is None: raw = np.ascontiguousarray(rgb).tobytes()
                    enc[c].stdin.write(raw); ptr[c] += 1
        info = dict(key=key, dataset=ds, source_fps=float(fps), frames=n0)
        for c in CH:
            assert ptr[c] == len(sel[c]), f'{c}: wrote {ptr[c]} of {len(sel[c])} frames'
            enc[c].stdin.close(); assert enc[c].wait() == 0, 'encoder failed'; n = len(sel[c]); (OUT / c).mkdir(parents=True, exist_ok=True); (CROPS / c).mkdir(parents=True, exist_ok=True)
            pk = packets(path[c]); info.update({'kbps_' + c: round(path[c].stat().st_size * 8 / (n / fo[c]) / 1000, 1), 'target_kbps_' + c: round(kb[c], 1), 'fps_' + c: float(fo[c]), 'bytes_I_' + c: pk[0], 'bytes_P_' + c: pk[1], 'key_frames_' + c: pk[2]})
            mesh = X.Mesh(); A = dict(valid=[], blk=[], fill=[], box=[], crop=[]); m = 0
            for fr in C5.decode(path[c], Wd, H):
                fr = np.ascontiguousarray(fr); good, crop, box = X.fast_crop(fr, mesh.step(fr)); m += 1
                A['valid'].append(bool(good)); A['box'].append(box); A['crop'].append(crop); A['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32)); A['fill'].append((crop.max(-1) > 0).reshape(6, 6, 6, 6).sum((1, 3)).astype(np.uint8))
            mesh.close(); path[c].unlink(); assert m == n, f'{c}: decoded {m} of {n} frames'
            tmp = OUT / c / (key + '.tmp.npz'); np.savez_compressed(tmp, t=tt[c], valid=np.array(A['valid']), blk=np.array(A['blk']), fill=np.array(A['fill']), box=np.array(A['box'], np.int16)); tmp.replace(OUT / c / (key + '.npz'))
            tmp = CROPS / c / (key + '.tmp.npz'); np.savez_compressed(tmp, crop=np.array(A['crop'])); tmp.replace(CROPS / c / (key + '.npz')); info['valid_' + c] = float(np.mean(A['valid']))
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [j for j in json.loads((X.R / 'MANIFEST.json').read_text()) if j['dataset'] != 'BTC']
    if len(sys.argv) > 2 and sys.argv[2] != 'x01': jobs = [j for j in jobs if j['key'] in sys.argv[2].split(',')]
    elif len(sys.argv) > 2: jobs = [j for j in jobs if (W / 'out/x01/native' / (j['key'] + '.npz')).exists()]          # the 35 UBFC2 + 38 MCD recordings of the x01 study
    log = (W / 'logs/x12_btcsim.jsonl').open('a'); done = 0
    with Pool(int(sys.argv[1]), maxtasksperchild=1) as pool:
        for info in pool.imap_unordered(run, list(enumerate(jobs))):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush(); print(done, len(jobs), {k: v for k, v in info.items() if k != 'traceback'}, flush=True)
    (W / 'out/x12_DONE').write_text(time.strftime('%F %T'))
