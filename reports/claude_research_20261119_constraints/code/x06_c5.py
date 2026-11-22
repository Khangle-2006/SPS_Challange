"""x06 - C5 on real video: frame-rate decimation to <=10 fps + H.264 at <=500 kbps, three ways (PREREG.md):
  c5_tuned : full frame, x264 medium, CRF 23 capped by VBV maxrate 450k / bufsize 900k (so that the whole-file average stays <= 500 kbps), no B-frames, no periodic key frames      [BTC]
  c5_vga   : frame first reduced to fit 640x480 (antialiased bicubic), then encoded as c5_tuned                               [BTC]
  c5_face  : 'device mode' - fixed face window (tight oval box of the first second, enlarged 30% per side) reduced to fit 192x240,
             encoded with CRF 18 under the same cap; the server-side front end then runs on that small video            [BTC, UBFC2, MCD]
(The public sets are 640x480 already, so c5_tuned/c5_vga coincide there with the team's existing 'c5prime_bf0'.)
Each encode is decoded again and given to the unchanged front end. Actual bitrate = file size / duration, must be <= 500 kbps.
Usage: nice -n 19 $AV code/x06_c5.py <workers> [keys]   -> out/x06/<cond>/<key>.npz (+ out/x06c crops for the public sets)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x01_extract as X
import json, time, traceback, subprocess, resource
from pathlib import Path
import numpy as np, cv2
co = X.co; W = X.W; OUT = W / 'out/x06'; CROPS = W / 'out/x06c'; TMP = W / 'out/x06_tmp'
def encoder(path, w, h, fps, crf):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', f'{fps:.6f}', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'medium',
           '-crf', str(crf), '-maxrate', '450k', '-bufsize', '900k', '-bf', '0', '-g', '100000', '-keyint_min', '100000', '-sc_threshold', '0', '-pix_fmt', 'yuv420p', '-threads', '1', '-y', str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
def decode(path, w, h):
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL); nb = w * h * 3
    try:
        while True:
            raw = p.stdout.read(nb)
            if len(raw) < nb: break
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close(); p.wait()
def even(v): return max(2, int(round(v / 2)) * 2)
def run(job):
    idx, j = job; key, ds = j['key'], j['dataset']; cs = ['c5_tuned', 'c5_vga', 'c5_face'] if ds == 'BTC' else ['c5_face']; t0 = time.time()
    if all((OUT / c / (key + '.npz')).exists() for c in cs): return dict(key=key, state='skip')
    if os.environ.get('X01_SKIP_HEAVY') == '1' and X.is_heavy(j): return dict(key=key, state='not_run_heavy')      # 4K / HDR files: see logs/DEVIATIONS.txt
    try:
        if ds == 'BTC':
            with np.load(Path(j['crop']) / 'meta.npz') as z: boxes, ref_t = z['boxes'], z['t']
            vid = X.BTCVideo(j['source'], ref_t); t = vid.t; src = iter(vid); Wd, H = vid.w, vid.h
        else:
            with np.load(j['meta']) as z: t, boxes = z['original_t'], z['boxes']
            src = X.frames_of(j, len(t)); Wd, H = 640, 480
        fps = (len(t) - 1) / (t[-1] - t[0]); step = max(1, int(np.ceil(fps / 10 - 1e-3))); keep = np.arange(0, len(t), step); tk = t[keep]; fo = fps / step
        ok = boxes[:, 2] > boxes[:, 0]; first = ok & (t <= t[0] + 1.); sel = first if first.any() else ok; size = {}
        if sel.any():
            b = np.median(boxes[sel].astype(float), 0); bw, bh = b[2] - b[0], b[3] - b[1]
            fx0, fy0 = int(max(0, b[0] - .3 * bw)), int(max(0, b[1] - .3 * bh)); fx1, fy1 = int(min(Wd, b[2] + .3 * bw)), int(min(H, b[3] + .3 * bh)); s = min(192 / (fx1 - fx0), 240 / (fy1 - fy0), 1.)
            size['c5_face'] = (even((fx1 - fx0) * s), even((fy1 - fy0) * s))
        size['c5_tuned'] = (Wd - Wd % 2, H - H % 2); size['c5_vga'] = co.c4_size_keep_aspect(Wd, H, (640, 480)); cs_run = [c for c in cs if c in size]
        TMP.mkdir(parents=True, exist_ok=True); path = {c: TMP / f'{key}__{c}.mp4' for c in cs_run}
        enc = {c: encoder(path[c], *size[c], fo, 18 if c == 'c5_face' else 23) for c in cs_run}; n = 0
        for i, rgb in enumerate(src):
            if i % step: continue
            n += 1
            for c in cs_run:
                if c == 'c5_tuned': fr = rgb[:size[c][1], :size[c][0]]
                elif c == 'c5_vga': fr = co.c4_resize(np.ascontiguousarray(rgb), size[c])
                else: fr = co.c4_resize(np.ascontiguousarray(rgb[fy0:fy1, fx0:fx1]), size[c])
                enc[c].stdin.write(np.ascontiguousarray(fr).tobytes())
        assert n == len(keep), f'{n} frames kept, expected {len(keep)}'
        info = dict(key=key, dataset=ds, frames=int(n), fps_out=float(fo), step=int(step), source_size=[int(Wd), int(H)])
        for c in cs:
            (OUT / c).mkdir(parents=True, exist_ok=True); A = dict(valid=[], blk=[], fill=[], box=[], crop=[])
            if c in enc:
                enc[c].stdin.close(); assert enc[c].wait() == 0, 'encoder failed'
                kbps = path[c].stat().st_size * 8 / (n / fo) / 1000; info['kbps_' + c] = round(float(kbps), 1); info['size_' + c] = list(size[c]); mesh = X.Mesh(); m = 0
                for fr in decode(path[c], *size[c]):
                    fr = np.ascontiguousarray(fr); good, crop, box = X.fast_crop(fr, mesh.step(fr)); m += 1
                    A['valid'].append(bool(good) and kbps <= 500.); A['box'].append(box); A['crop'].append(crop)
                    A['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32)); A['fill'].append((crop.max(-1) > 0).reshape(6, 6, 6, 6).sum((1, 3)).astype(np.uint8))
                mesh.close(); path[c].unlink(); assert m == n, f'{c}: decoded {m} of {n} frames'
            else:
                A = dict(valid=[False] * n, blk=np.zeros((n, 6, 6, 3), np.float32), fill=np.zeros((n, 6, 6), np.uint8), box=np.zeros((n, 4), np.int16), crop=np.zeros((n, 36, 36, 3), np.uint8)); info['kbps_' + c] = None
            tmp = OUT / c / (key + '.tmp.npz'); np.savez_compressed(tmp, t=tk, valid=np.array(A['valid']), blk=np.array(A['blk']), fill=np.array(A['fill']), box=np.array(A['box'], np.int16)); tmp.replace(OUT / c / (key + '.npz'))
            if ds != 'BTC':
                (CROPS / c).mkdir(parents=True, exist_ok=True); tmp = CROPS / c / (key + '.tmp.npz'); np.savez_compressed(tmp, crop=np.array(A['crop'])); tmp.replace(CROPS / c / (key + '.npz'))
            info['valid_' + c] = float(np.mean(A['valid']))
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done'); return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = json.loads((X.R / 'MANIFEST.json').read_text())
    if len(sys.argv) > 2: jobs = [j for j in jobs if j['key'] in sys.argv[2].split(',')]
    order = np.random.default_rng(1).permutation(len(jobs)); log = (W / 'logs/x06_c5.jsonl').open('a'); done = 0
    with Pool(int(sys.argv[1]), maxtasksperchild=1) as pool:
        for info in pool.imap_unordered(run, [(int(i), jobs[i]) for i in order]):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush(); print(done, len(jobs), {k: v for k, v in info.items() if k != 'traceback'}, flush=True)
    (W / 'out/x06_DONE').write_text(time.strftime('%F %T'))
