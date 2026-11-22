import json, subprocess
import numpy as np
import cv2
cv2.setNumThreads(1)
OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
EMPTY=np.zeros((36,36,3),np.uint8)
ZERO4=np.zeros(4,np.int32)


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

def aid(frame):
    H, Wd = frame.shape[:2]; s = min(1., 640 / max(H, Wd))
    small = frame if s == 1. else cv2.resize(frame, (int(round(Wd * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(small, (0, 0), 1.5); gain = min(max(180. / max(float(np.percentile(g[::4, ::4], 99)), 1.), 1.), 8.)
    return np.ascontiguousarray(g if gain == 1. else cv2.convertScaleAbs(g, alpha=gain))

def _grey(img, r, s):
    g = cv2.cvtColor(img[r[1]:r[3], r[0]:r[2]], cv2.COLOR_RGB2GRAY)
    return g if s >= 1 else cv2.resize(g, (max(8, int(round(g.shape[1] * s))), max(8, int(round(g.shape[0] * s)))), interpolation=cv2.INTER_AREA)

def tpl_make(img, pts):
    x0, y0 = pts.min(0); x1, y1 = pts.max(0); bw, bh = float(x1 - x0), float(y1 - y0)
    if bw < 12 or bh < 12: return None
    r = (int(max(0, x0)), int(max(0, y0)), int(min(img.shape[1], x1)), int(min(img.shape[0], y0 + .55 * bh)))
    if r[2] - r[0] < 8 or r[3] - r[1] < 8: return None
    s = min(1., 64. / bw); return dict(g=_grey(img, r, s), r=r, s=s, bw=bw, bh=bh)

def tpl_find(img, tp, thr=.6):
    """displacement (dx, dy) of the stored upper-face template in `img`, or None when the match is too weak."""
    r = tp['r']; mx, my = int(.15 * tp['bw']) + 1, int(.15 * tp['bh']) + 1; H, W_ = img.shape[:2]
    q = (max(0, r[0] - mx), max(0, r[1] - my), min(W_, r[2] + mx), min(H, r[3] + my)); g = _grey(img, q, tp['s'])
    if g.shape[0] < tp['g'].shape[0] or g.shape[1] < tp['g'].shape[1]: return None
    res = cv2.matchTemplate(g, tp['g'], cv2.TM_CCOEFF_NORMED); _, mxv, _, loc = cv2.minMaxLoc(res)
    if not np.isfinite(mxv) or mxv < thr: return None
    return np.array([q[0] + loc[0] / tp['s'] - r[0], q[1] + loc[1] / tp['s'] - r[1]], np.float32)

class Det:
    """MediaPipe FaceDetection, full-range model (faces up to ~5 m): boxes (x0, y0, x1, y1) in pixels for every face in the frame."""
    def __init__(self):
        import mediapipe as mp
        self.d = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=.5)
    def step(self, image):
        h, w = image.shape[:2]; r = self.d.process(image); out = []
        for d in (r.detections or []):
            b = d.location_data.relative_bounding_box; out.append(np.array([b.xmin * w, b.ymin * h, (b.xmin + b.width) * w, (b.ymin + b.height) * h]))
        return out
    def close(self): self.d.close()

def region_points(image, box, mesh, st):
    """FaceMesh on the region around one detection box (enlarged 50% sideways, 60% up, 40% down). The region is kept fixed until the box
    centre moves by >15% of its width or its size changes by >20%, so that the mesh tracker sees a stable image."""
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; r = st.get('reg')
    if r is None or abs(cx - st['c'][0]) > .15 * st['s'][0] or abs(cy - st['c'][1]) > .15 * st['s'][1] or abs(w / st['s'][0] - 1) > .2:
        H, W_ = image.shape[:2]; r = (int(max(0, x0 - .5 * w)), int(max(0, y0 - .6 * h)), int(min(W_, x1 + .5 * w)), int(min(H, y1 + .4 * h))); st.update(reg=r, c=(cx, cy), s=(w, h))
    if r[2] - r[0] < 16 or r[3] - r[1] < 16: return None
    pts = mesh.step(np.ascontiguousarray(image[r[1]:r[3], r[0]:r[2]]))
    return None if pts is None else pts + np.array([r[0], r[1]], np.float32)

def signature(gray, box):
    """12x12 mean-removed luminance thumbnail of the face box (cosine distance = 1 - correlation): a cheap appearance cue, not a face-ID model."""
    x0, y0, x1, y1 = [int(v) for v in box]; x0, y0 = max(x0, 0), max(y0, 0)
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    th = cv2.resize(gray[y0:y1, x0:x1], (12, 12), interpolation=cv2.INTER_AREA).astype(np.float64).ravel(); th -= th.mean(); n = np.linalg.norm(th)
    return th / n if n > 0 else None
