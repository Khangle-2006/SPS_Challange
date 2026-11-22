"""C3 - target-locked multi-face tracking (detector-agnostic).

Design (same structure as the MATLAB recipe in the official document: detections -> Kalman prediction ->
assignment -> persistent identity; cf. SORT, Bewley et al., ICIP 2016):
  * every face detection of every frame is assigned to a track by the Hungarian algorithm on a gated
    Mahalanobis cost (constant-velocity Kalman filter on box centre, random walk on box size);
  * ONE track is designated as the target (seed box / seed point, else largest face, ties -> closest to centre)
    and its identity is locked; the rPPG front end only ever receives that track's box;
  * the target is never silently replaced: when it is unmatched it coasts on the prediction for at most
    `coast_s` seconds, then becomes LOST; re-acquisition needs an appearance match (when descriptors are given)
    or an unambiguous single candidate inside a position gate;
  * frames are flagged invalid when the target is coasting/lost or when a bystander box overlaps the target box
    (IoU > `overlap_iou`), so the window validity rule of the pipeline rejects them instead of mixing people.

Pure numpy/scipy; boxes are (x1, y1, x2, y2) in pixels; timestamps in seconds (variable frame rate is fine)."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_4_999 = 18.47      # chi-square 99.9% quantile, 4 dof (centre x, centre y, width, height)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.


def to_z(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2., (y1 + y2) / 2., x2 - x1, y2 - y1], float)


def to_box(z):
    cx, cy, w, h = z[:4]
    return np.array([cx - w / 2., cy - h / 2., cx + w / 2., cy + h / 2.])


@dataclass
class Track:
    tid: int
    x: np.ndarray                      # state [cx, cy, w, h, vx, vy]
    P: np.ndarray
    t_update: float
    hits: int = 1
    desc: np.ndarray = None            # running appearance template (optional)

    def predict(self, dt, q_acc, q_size):
        F = np.eye(6); F[0, 4] = F[1, 5] = dt
        s = np.sqrt(max(self.x[2] * self.x[3], 1.))          # box scale in pixels
        qa = (q_acc * s) ** 2; qs = (q_size * s) ** 2 * dt
        Q = np.diag([qa * dt ** 4 / 4, qa * dt ** 4 / 4, qs, qs, qa * dt ** 2, qa * dt ** 2])
        Q[0, 4] = Q[4, 0] = Q[1, 5] = Q[5, 1] = qa * dt ** 3 / 2
        self.x = F @ self.x; self.P = F @ self.P @ F.T + Q
        self.x[2:4] = np.maximum(self.x[2:4], 2.)

    def gate(self, z, r_meas):
        H = np.zeros((4, 6)); H[:4, :4] = np.eye(4)
        s = np.sqrt(max(self.x[2] * self.x[3], 1.)); R = np.eye(4) * (r_meas * s) ** 2
        y = z - H @ self.x; S = H @ self.P @ H.T + R
        return float(y @ np.linalg.solve(S, y)), y, S, H

    def update(self, z, r_meas, desc=None, alpha=.1):
        _, y, S, H = self.gate(z, r_meas)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y; self.P = (np.eye(6) - K @ H) @ self.P; self.hits += 1
        if desc is not None:
            self.desc = desc.copy() if self.desc is None else (1 - alpha) * self.desc + alpha * desc


def desc_dist(a, b):
    if a is None or b is None: return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1. - float(a @ b) / (na * nb) if na > 0 and nb > 0 else 1.


class TargetLock:
    """step(t, boxes, descs=None) -> dict(box, valid, state, track_id, n_faces, contaminated)."""

    def __init__(self, frame_size=None, seed_box=None, seed_point=None, coast_s=.5, drop_s=1., overlap_iou=.15,
                 q_acc=3., q_size=.5, r_meas=.06, w_app=12., app_thr=.25, reacq_gate=1.5, min_hits=3, warmup_s=.7):
        self.frame_size = frame_size; self.seed_box = seed_box; self.seed_point = seed_point
        self.coast_s, self.drop_s, self.overlap_iou = coast_s, drop_s, overlap_iou
        self.q_acc, self.q_size, self.r_meas = q_acc, q_size, r_meas
        self.w_app, self.app_thr, self.reacq_gate, self.min_hits = w_app, app_thr, reacq_gate, min_hits
        self.warmup_s = warmup_s; self.t0 = None; self.n_frames = 0; self.area_sum = {}      # per-track evidence gathered before designation
        self.tracks = []; self.next_id = 0; self.target = None; self.t_prev = None; self.state = 'INIT'
        self.id_events = []            # (t, event) log: designation, loss, re-acquisition

    # -- target designation ------------------------------------------------------------------------------
    def _designate(self, t):
        """The target is chosen only after `warmup_s` seconds, among tracks seen in >=60% of the warm-up frames
        (a face missed by the detector in the very first frame must not hand the lock to a bystander)."""
        if self.t0 is None or (t - self.t0) < self.warmup_s: return
        cands = [k for k in self.tracks if k.hits >= max(self.min_hits, .6 * self.n_frames)]
        if not cands: return
        if self.seed_box is not None:
            best = max(cands, key=lambda k: iou(to_box(k.x), self.seed_box))
            if iou(to_box(best.x), self.seed_box) <= 0: return
        elif self.seed_point is not None:
            p = np.asarray(self.seed_point, float); best = min(cands, key=lambda k: np.linalg.norm(k.x[:2] - p))
        else:
            c = None if self.frame_size is None else np.asarray(self.frame_size, float) / 2
            area = np.array([self.area_sum.get(k.tid, 0.) / max(k.hits, 1) for k in cands]); big = [k for k, a in zip(cands, area) if a >= .8 * area.max()]
            best = big[0] if (c is None or len(big) == 1) else min(big, key=lambda k: np.linalg.norm(k.x[:2] - c))
        self.target = best.tid; self.state = 'LOCKED'; self.id_events.append((t, f'designate track {best.tid}'))

    def _get(self, tid):
        for k in self.tracks:
            if k.tid == tid: return k
        return None

    # -- one frame ------------------------------------------------------------------------------------------
    def step(self, t, boxes, descs=None):
        boxes = [np.asarray(b, float) for b in boxes]; Z = [to_z(b) for b in boxes]
        descs = [None] * len(boxes) if descs is None else list(descs)
        dt = 0. if self.t_prev is None else max(1e-3, t - self.t_prev); self.t_prev = t
        if self.t0 is None: self.t0 = t
        self.n_frames += 1
        for k in self.tracks:
            if dt > 0: k.predict(dt, self.q_acc, self.q_size)
        # gated assignment (Hungarian); the appearance term only refines the cost when descriptors exist
        nT, nD = len(self.tracks), len(Z); BIG = 1e6; C = np.full((nT, nD), BIG)
        for i, k in enumerate(self.tracks):
            for j, z in enumerate(Z):
                d2 = k.gate(z, self.r_meas)[0]
                if d2 > CHI2_4_999: continue
                a = desc_dist(k.desc, descs[j])
                if a is not None and k.tid == self.target and a > 2 * self.app_thr: continue   # never glue the target to a different-looking face
                C[i, j] = d2 + (self.w_app * a if a is not None else 0.)
        matched_t, matched_d = set(), set()
        if nT and nD:
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] >= BIG: continue
                self.tracks[i].update(Z[j], self.r_meas, descs[j]); self.tracks[i].t_update = t
                self.area_sum[self.tracks[i].tid] = self.area_sum.get(self.tracks[i].tid, 0.) + Z[j][2] * Z[j][3]
                matched_t.add(i); matched_d.add(j)
        tgt = self._get(self.target) if self.target is not None else None
        # re-acquisition of a lost target: appearance match, or a single unambiguous candidate near the prediction
        if tgt is not None and self.tracks.index(tgt) not in matched_t and (t - tgt.t_update) > self.coast_s:
            free = [j for j in range(nD) if j not in matched_d]; pick = None
            s = np.sqrt(max(tgt.x[2] * tgt.x[3], 1.))
            near = [j for j in free if np.linalg.norm(Z[j][:2] - tgt.x[:2]) <= self.reacq_gate * s * (1 + (t - tgt.t_update))]
            if tgt.desc is not None and any(descs[j] is not None for j in free):
                sc = [(desc_dist(tgt.desc, descs[j]), j) for j in free if descs[j] is not None]
                sc = [x for x in sc if x[0] <= self.app_thr]
                if sc: pick = min(sc)[1]
            elif len(near) == 1 and nD == 1:
                pick = near[0]
            if pick is not None:
                tgt.x[:4] = Z[pick]; tgt.x[4:] = 0.; tgt.P = np.diag([s ** 2 * .01] * 4 + [s ** 2] * 2)
                tgt.t_update = t; tgt.hits += 1; matched_d.add(pick); matched_t.add(self.tracks.index(tgt))
                if descs[pick] is not None and tgt.desc is not None: tgt.desc = .9 * tgt.desc + .1 * descs[pick]
                self.id_events.append((t, 're-acquired'))
        for j in range(nD):                       # births
            if j in matched_d: continue
            z = Z[j]; s = np.sqrt(max(z[2] * z[3], 1.))
            self.tracks.append(Track(self.next_id, np.r_[z, 0., 0.], np.diag([(self.r_meas * s) ** 2] * 4 + [(2 * s) ** 2] * 2), t,
                                     desc=None if descs[j] is None else descs[j].copy())); self.area_sum[self.next_id] = z[2] * z[3]; self.next_id += 1
        self.tracks = [k for k in self.tracks if (t - k.t_update) <= self.drop_s or k.tid == self.target]
        if self.target is None: self._designate(t)
        tgt = self._get(self.target) if self.target is not None else None
        if tgt is None:
            return dict(box=None, valid=False, state='INIT', track_id=None, n_faces=nD, contaminated=False)
        age = t - tgt.t_update; box = to_box(tgt.x)
        state = 'LOCKED' if age == 0 else ('COAST' if age <= self.coast_s else 'LOST')
        if state != self.state:
            if state == 'LOST': self.id_events.append((t, 'lost'))
            self.state = state
        others = [to_box(k.x) for k in self.tracks if k.tid != self.target and (t - k.t_update) <= self.coast_s]
        contaminated = any(iou(box, o) > self.overlap_iou for o in others)
        valid = state == 'LOCKED' and tgt.hits >= self.min_hits and not contaminated
        return dict(box=box, valid=valid, state=state, track_id=tgt.tid, n_faces=nD, contaminated=contaminated)
