"""x01 - pixel-level constrained re-extraction (native, C1, C4, C6, C7) for UBFC2 dev (35), MCD (119) and BTC dev (218).
Every video is decoded ONCE; each condition gets its own FaceMesh instance (tracking state is per stream) and the degraded frame is
given to the team's unchanged front end (oval 36 points -> tight masked box -> 36x36 INTER_AREA). Nothing is written but small traces
(6x6 block means of the crop) and, for the public sets, the uint8 crops needed by the CNN (deleted after inference by x02).
Run with the storage_pilot env:  nice -n 19 $AV code/x01_extract.py <workers> [keys|-] [first_core] [max_heavy_BTC]
Read-only on everything outside this folder. BTC: dev videos only (MANIFEST of inference_repair_20261113 = 218 dev videos)."""
import os, sys
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['__EGL_VENDOR_LIBRARY_DIRS'] = '/nonexistent'   # keep MediaPipe off the shared GPUs (no EGL context); CPU inference is unchanged (checked)
import json, time, traceback, subprocess, zlib, resource
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
R = Path('/mnt/samsung_t5/khangle/facevitals/reports/round1_execution_20261110/inference_repair_20261113')
T8 = '/data_pool/khangle/facevitals/code/t8_oval_20261031'
sys.path.insert(0, str(W / 'code')); sys.path.insert(1, T8)
import constraint_ops as co
from oval_crop import crop_from_points          # the team's function, unchanged
OVAL = co.OVAL; OUT = W / 'out/x01'; CROPS = W / 'out/x01c'
C1 = [('c1_g50p', 50, 'p', 0), ('c1_g25p', 25, 'p', 0), ('c1_g12p', 12, 'p', 0), ('c1_g25gv001', 25, 'g', .001), ('c1_g25gv01', 25, 'g', .01)]
GAIN = {50: .5, 25: .25, 12: .125}
LUTS = {g: np.clip(np.floor(np.arange(256) * v + .5), 0, 255).astype(np.uint8) for g, v in GAIN.items()}
C7 = [('c7_blue', 'surgical_blue', 'standard'), ('c7_white', 'white', 'standard'), ('c7_black_high', 'black', 'high')]
KUS = co.unsharp_kernel(.2)
REDUCED = os.environ.get('X01_REDUCED') == '1'      # 2nd launch (22:40): the machine delivers ~1/3 of the single-job speed under full load
KEEP = ['native', 'c1_g25p', 'c1_g25gv01', 'c4_vga', 'c4_qvga', 'c6_us1', 'c7_blue']
def conds(ds): return ['native'] + [c[0] for c in C1] + (['c4_vga'] if ds == 'BTC' else []) + ['c4_qvga', 'c6_us1'] + [c[0] for c in C7]

KP = 14                                            # Poisson: 14-bit uniform per sample (quantile resolution 6e-5); Gaussian: 16-bit (1.5e-5)
def poisson_table():
    """inverse-CDF table for imnoise(uint8,'poisson'): PT[lam << KP | u] = quantile of Poisson(lam) at (u+.5)/2^KP, clipped to 255.
    Sampling by table look-up is exact up to the uniform resolution and ~10x faster than Generator.poisson on megapixel regions."""
    from scipy.stats import poisson
    k = np.arange(256); u = (np.arange(1 << KP) + .5) / (1 << KP); T = np.zeros((256, 1 << KP), np.uint8)
    for lam in range(1, 256): T[lam] = np.minimum(np.searchsorted(poisson.cdf(k, lam), u, side='left'), 255)
    return T.ravel()
def gauss_table(var):
    """imnoise(I,'gaussian',0,var) on uint8: I + round(255*sqrt(var)*z) clipped to [0,255]; z by inverse CDF of a 16-bit uniform."""
    from scipy.stats import norm
    return np.rint(255. * np.sqrt(var) * norm.ppf((np.arange(65536) + .5) / 65536)).astype(np.int16)
PT = None; GT = {}
def pois(roi, rng, rows=32):
    out = np.empty_like(roi)
    for r in range(0, roi.shape[0], rows):          # row blocks keep the temporaries and the active part of the table in cache
        x = roi[r:r + rows]; idx = x.astype(np.int32); idx <<= KP; idx += rng.integers(0, 1 << KP, x.shape, dtype=np.uint16); out[r:r + rows] = PT.take(idx)
    return out
def gauss(roi, var, rng):
    x = roi.astype(np.int16); x += GT[var].take(rng.integers(0, 65536, roi.shape, dtype=np.uint16)); np.clip(x, 0, 255, out=x)
    return x.astype(np.uint8)

EMPTY = np.zeros((36, 36, 3), np.uint8); ZERO4 = np.zeros(4, np.int32)
def fast_crop(image, points):
    """same output as oval_crop.crop_from_points (checked frame by frame in logs/prof3.py), but the hull mask is rasterised inside the
    tight box only instead of on a full-frame canvas - 3x faster on HD frames."""
    if points is None or not np.isfinite(points).all(): return False, EMPTY, ZERO4
    pp = np.rint(points).astype(np.int32); h, w = image.shape[:2]
    x0, y0 = np.maximum(pp.min(0), 0); x1, y1 = np.minimum(pp.max(0) + 1, [w, h])
    if x1 <= x0 or y1 <= y0: return False, EMPTY, ZERO4
    m = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(m, [cv2.convexHull(pp) - np.array([x0, y0], np.int32)], 1)
    if int(m.sum()) < 16: return False, EMPTY, ZERO4
    region = cv2.bitwise_and(image[y0:y1, x0:x1], image[y0:y1, x0:x1], mask=m)
    return True, cv2.resize(region, (36, 36), interpolation=cv2.INTER_AREA), np.array([x0, y0, x1, y1], np.int32)

class Mesh:
    """same construction as t8_oval_20261031/oval_frontend.OvalFrontend; returns the 36 oval points (or all 468)."""
    def __init__(self):
        import mediapipe as mp
        self.m = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_confidence=.5, min_tracking_confidence=.5)
    def step(self, image, full=False):
        h, w = image.shape[:2]; r = self.m.process(image)
        if not r.multi_face_landmarks: return None
        lm = r.multi_face_landmarks[0].landmark
        return np.asarray([(p.x * w, p.y * h) for p in lm] if full else [(lm[i].x * w, lm[i].y * h) for i in OVAL], np.float32)
    def close(self): self.m.close()

class BTCVideo:
    """decode exactly as btc_zeroshot/preprocess.Video (display PTS from ffprobe, autorotate once, same HDR->SDR filter chain)."""
    def __init__(self, path, t=None):
        """t: display timestamps already stored by the team's preprocessing for this file (then the frame-level ffprobe, which decodes the
        whole file, is skipped; the frame count is still checked against t after decoding)."""
        if t is None:
            pr = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)]))
            self.t = np.array([float(f['best_effort_timestamp_time']) for f in pr['frames']]); self.t -= self.t[0]
        else:
            pr = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', str(path)])); self.t = np.asarray(t, float)
        st = pr['streams'][0]
        rot = next((float(s['rotation']) for s in st.get('side_data_list', []) if 'rotation' in s), 0)
        self.w, self.h = int(st['width']), int(st['height'])
        if abs(rot) % 180 == 90: self.w, self.h = self.h, self.w
        self.hdr = st.get('color_transfer', 'unknown') in ['arib-std-b67', 'smpte2084']
        vf = 'zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0:peak=10,zscale=t=bt709:m=bt709:r=full,format=rgb24' if self.hdr else 'format=rgb24'
        self.cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-dn', '-filter_threads', '1', '-vf', vf, '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    def __iter__(self):
        p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL); nb = self.w * self.h * 3
        try:
            while True:
                raw = p.stdout.read(nb)
                if len(raw) < nb: break
                yield np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
        finally:
            p.stdout.close()
            if p.poll() is None: p.terminate(); p.wait()

def frames_of(j, n):
    if j['dataset'] == 'UBFC2':
        cap = cv2.VideoCapture(j['source']); k = 0
        while k < n:
            ok, bgr = cap.read()
            if not ok: break
            k += 1; yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cap.release()
    else:
        import av
        with av.open(j['source']) as c:
            c.streams.video[0].codec_context.thread_count = 1; k = 0
            for fr in c.decode(video=0):
                if k >= n: return
                k += 1; yield fr.to_ndarray(format='rgb24')

CORES = None; HEAVY = None        # kept for the scripts that import this module (x09); not used by x01 since 23:35
EXCLUDE = set(filter(None, os.environ.get('X01_EXCLUDE', '').split(',')))
def is_heavy(j):
    md = (j.get('metadata') or {}).get('media', {}) if isinstance(j.get('metadata'), dict) else {}
    return j['dataset'] == 'BTC' and (int(md.get('width', 0)) * int(md.get('height', 0)) > 4e6 or md.get('transfer') in ('arib-std-b67', 'smpte2084'))
def pin_worker(first, n):
    """pool initializer: persistent worker k is pinned to physical core first + k (both hyperthreads)."""
    import multiprocessing as mp
    k = (mp.current_process()._identity[0] - 1) % n; os.sched_setaffinity(0, {first + k, first + k + os.cpu_count() // 2})
def heavy_slot(n=8):
    """crash-safe limit on concurrent 4K/HDR decodes: hold an exclusive flock on one of n lock files (released when the process ends)."""
    import fcntl
    d = W / 'out/locks'; d.mkdir(parents=True, exist_ok=True)
    while True:
        for k in range(n):
            f = open(d / f'heavy_{k}.lock', 'w')
            try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); return f
            except OSError: f.close()
        time.sleep(3)
def run(job):
    idx, j = job; key, ds = j['key'], j['dataset']; cs = conds(ds); t0 = time.time()
    if all((OUT / c / (key + '.npz')).exists() for c in cs): return dict(key=key, state='skip')
    if key in EXCLUDE: return dict(key=key, state='excluded_in_progress_elsewhere')
    if REDUCED:
        if ds == 'MCD': return dict(key=key, state='not_run_budget')
        if ds == 'BTC':
            cs = [c for c in cs if c in KEEP]
            if all((OUT / c / (key + '.npz')).exists() for c in cs): return dict(key=key, state='skip')
    return _run(idx, j, key, ds, cs, t0)
def _run(idx, j, key, ds, cs, t0):
    held = False
    try:
        if ds == 'BTC':
            with np.load(Path(j['crop']) / 'meta.npz') as z: ref_t, ref_valid, ref_mean, boxes = z['t'], z['valid'], z['rgb_mean'], z['boxes']
            vid = BTCVideo(j['source'], ref_t); t = vid.t
            if vid.w * vid.h > 4e6 or vid.hdr: held = heavy_slot(int(os.environ.get('X01_HEAVY', '8')))      # 4K / HDR decoders hold ~1 GB each
            src = iter(vid)
            ok = boxes[:, 2] > boxes[:, 0]; nb = None
            if ok.any():                       # stored native box, nearest valid frame, enlarged by 50% on each side: region that receives noise
                near = np.flatnonzero(ok)[np.abs(np.arange(len(ok))[:, None] - np.flatnonzero(ok)[None]).argmin(1)]; b = boxes[near].astype(float)
                bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
                nb = np.stack([b[:, 0] - .5 * bw, b[:, 1] - .5 * bh, b[:, 2] + .5 * bw, b[:, 3] + .5 * bh], 1)
        else:
            with np.load(j['meta']) as z: t, ref_tu, ref_valid = z['original_t'], z['t'], z['valid']
            src = frames_of(j, len(t)); nb = None
        mesh = {c: Mesh() for c in cs}; seed = zlib.crc32(key.encode())
        rng = {c: np.random.default_rng([20261119, seed, i]) for i, c in enumerate(conds(ds))}
        A = {c: dict(valid=[], blk=[], fill=[], box=[], crop=[]) for c in cs}; nanp = np.full((36, 2), np.nan, np.float32); poly = {}; occ = []; n = 0
        def put(c, image, pts, force_invalid=False):
            good, crop, box = fast_crop(image, pts); a = A[c]
            a['valid'].append(bool(good) and not force_invalid); a['box'].append(box)
            a['blk'].append(crop.reshape(6, 6, 6, 6, 3).mean((1, 3), dtype=np.float32)); a['fill'].append((crop.max(-1) > 0).reshape(6, 6, 6, 6).sum((1, 3)).astype(np.uint8))
            if ds != 'BTC': a['crop'].append(crop)
        for rgb in src:
            rgb = np.ascontiguousarray(rgb); H, Wd = rgb.shape[:2]
            lm = mesh['native'].step(rgb, full=True); put('native', rgb, None if lm is None else lm[OVAL])
            if nb is None: x0, y0, x1, y1 = 0, 0, Wd, H
            else: x0, y0 = max(0, int(nb[n, 0])), max(0, int(nb[n, 1])); x1, y1 = min(Wd, int(nb[n, 2]) + 1), min(H, int(nb[n, 3]) + 1)
            att = {g: cv2.LUT(rgb, LUTS[g]) for g in GAIN if any(c1[1] == g and c1[0] in mesh for c1 in C1)}
            for c, g, kind, var in C1:
                if c not in mesh: continue
                out = att[g].copy(); roi = out[y0:y1, x0:x1]
                out[y0:y1, x0:x1] = pois(roi, rng[c]) if kind == 'p' else gauss(roi, var, rng[c])
                put(c, out, mesh[c].step(out))
            del att
            for c, tg in ([('c4_vga', (640, 480))] if ds == 'BTC' else []) + [('c4_qvga', (320, 240))]:
                small = np.ascontiguousarray(co.c4_resize(rgb, co.c4_size_keep_aspect(Wd, H, tg))); put(c, small, mesh[c].step(small))
            us = cv2.filter2D(rgb, -1, KUS, borderType=cv2.BORDER_REPLICATE); put('c6_us1', us, mesh['c6_us1'].step(us)); del us
            for c, colour, fit in C7:
                if c not in mesh: continue
                if lm is not None: poly[fit] = np.round(co.c7_polygon(lm, fit)).astype(np.int32)
                out = rgb.copy()
                if fit in poly: cv2.fillPoly(out, [poly[fit]], co.MASK_RGB[colour])
                put(c, out, mesh[c].step(out), force_invalid=lm is None)
                if c == 'c7_blue' and lm is not None and n % 15 == 0:
                    bx = np.round(lm[OVAL]).astype(np.int32); o0 = np.maximum(bx.min(0) - 2, 0); sh = tuple((bx.max(0) - o0 + 4)[::-1])
                    om = np.zeros(sh, np.uint8); cv2.fillPoly(om, [cv2.convexHull(bx - o0)], 1); pm = np.zeros(sh, np.uint8); cv2.fillPoly(pm, [poly[fit] - o0], 1)
                    occ.append(float((om & pm).sum() / max(om.sum(), 1)))
            n += 1
        for m in mesh.values(): m.close()
        assert n == len(t), f'frame count {n} != timestamps {len(t)}'
        info = dict(key=key, dataset=ds, frames=n, size=[int(Wd), int(H)], occluded_fraction_c7_blue=float(np.mean(occ)) if occ else None)
        for c in cs:
            a = A[c]; (OUT / c).mkdir(parents=True, exist_ok=True); tmp = OUT / c / (key + '.tmp.npz')
            np.savez_compressed(tmp, t=t, valid=np.array(a['valid']), blk=np.array(a['blk']), fill=np.array(a['fill']), box=np.array(a['box'], np.int16)); tmp.replace(OUT / c / (key + '.npz'))
            if ds != 'BTC':
                (CROPS / c).mkdir(parents=True, exist_ok=True); tmp = CROPS / c / (key + '.tmp.npz'); np.savez_compressed(tmp, crop=np.array(a['crop'])); tmp.replace(CROPS / c / (key + '.npz'))
            info['valid_' + c] = float(np.mean(a['valid']))
        mine = np.array(A['native']['blk']).mean((1, 2)); mv = np.array(A['native']['valid'])
        if ds == 'BTC': info['native_check'] = dict(max_abs_mean_diff=float(np.abs(mine - ref_mean).max()), valid_mismatch=int((mv != ref_valid).sum()))
        else:
            ref = np.asarray(np.load(Path(j['crop']) / 'rgb.npy', mmap_mode='r'), np.float32).mean((1, 2)); u, inv, cnt = np.unique(t, return_inverse=True, return_counts=True)
            mm = np.zeros((len(u), 3)); np.add.at(mm, inv, mine); mm /= cnt[:, None]; uv = np.ones(len(u), bool); np.logical_and.at(uv, inv, mv)
            info['native_check'] = dict(max_abs_mean_diff=float(np.abs(mm - ref).max()), valid_mismatch=int((uv != ref_valid).sum()), merged_frames=int(len(u)))
        info.update(seconds=round(time.time() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, state='done')
        return info
    except Exception as e:
        return dict(key=key, state='error', error=repr(e), traceback=traceback.format_exc()[-1500:])
    finally:
        if held: held.close()

if __name__ == '__main__':
    import multiprocessing as mp
    from multiprocessing import Pool
    workers = int(sys.argv[1]); limit = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    first = int(sys.argv[3]) if len(sys.argv) > 3 else None
    jobs = json.loads((R / 'MANIFEST.json').read_text()); assert sum(j['dataset'] == 'BTC' for j in jobs) == 218
    if limit: jobs = [j for j in jobs if j['key'] in limit.split(',')]
    order = np.random.default_rng(0).permutation(len(jobs)); order = sorted(order, key=lambda i: (jobs[i]['dataset'] != 'UBFC2', is_heavy(jobs[i]))); PT = poisson_table(); GT.update({v: gauss_table(v) for v in (.001, .01)}); log = (W / 'logs/x01_extract.jsonl').open('a')
    print('jobs', len(jobs), 'workers', workers, flush=True); done = 0
    with (Pool(workers, initializer=pin_worker, initargs=(first, workers)) if first is not None else Pool(workers)) as pool:
        for info in pool.imap_unordered(run, [(int(i), jobs[i]) for i in order]):
            done += 1; log.write(json.dumps(info) + '\n'); log.flush()
            print(done, len(jobs), info.get('key'), info.get('state'), info.get('seconds'), info.get('maxrss_mb'), info.get('native_check'), info.get('error', ''), flush=True)
    (W / 'out/x01_DONE').write_text(time.strftime('%F %T'))
