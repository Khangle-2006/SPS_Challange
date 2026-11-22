"""f01 - illustration figures for the team's internal Notion page (22 Nov 2026). One MCD recording, same front-end code as the pipeline.
Usage: $AV code/f01_illustrations.py [key]  -> out/f01/*.png  (internal use only: lab members are bound by the dataset terms)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import signal
import x01_extract as X
import x09_followup as F
co = X.co; W = X.W; OUT = W / 'out/f01'; OUT.mkdir(parents=True, exist_ok=True)
KEY = sys.argv[1] if len(sys.argv) > 1 else 'MCD_1115_FullHDwebcam_before'
J = {j['key']: j for j in json.loads((X.R / 'MANIFEST.json').read_text())}; j = J[KEY]
with np.load(j['meta']) as z: z_t = z['original_t']
t = z_t
N = min(len(t), 30 * 40)                       # first 40 s
t = t[:N]; fps = (N - 1) / (t[-1] - t[0]); kk_ = max(1, int(np.ceil(fps / 10 - 1e-3))); KEEP = set(range(0, N, kk_ * 3))
mesh = X.Mesh(); LM = []; tr = []; frames = {}; i0 = None
for i, f in enumerate(X.frames_of(j, len(z_t))):
    if i >= N: break
    f = np.ascontiguousarray(f); l = mesh.step(f, full=True); LM.append(l)
    if l is None: tr.append([np.nan] * 3)
    else:
        g_, c_, _ = X.fast_crop(f, l[co.OVAL]); m_ = c_.reshape(-1, 3); m_ = m_[m_.max(1) > 0]; tr.append(m_.mean(0) if len(m_) else [np.nan] * 3)
        if i0 is None: i0 = i; frames[i] = f
    if i in KEEP and len(frames) < 12: frames[i] = f
mesh.close()
rgb = frames[i0]; lm = LM[i0]; oval = lm[co.OVAL]
def face_view(img, l, pad=.35, size=300):
    x0, y0 = l.min(0); x1, y1 = l.max(0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0) * (1 + pad) / 2
    H, Wd = img.shape[:2]; a, b = int(max(cx - s, 0)), int(max(cy - s, 0)); c, d = int(min(cx + s, Wd)), int(min(cy + s, H))
    return cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA), (a, b, (c - a) / size)
def box72(img, pts):
    good, crop, box = X.fast_crop(img, pts)
    return crop, cv2.resize(img[box[1]:box[3], box[0]:box[2]], (72, 72), interpolation=cv2.INTER_AREA)
def show(ax, im, title, nearest=False):
    ax.imshow(im, interpolation='nearest' if nearest else 'antialiased'); ax.set_title(title, fontsize=10); ax.axis('off')
# ---------- fig 1: front end ----------
fv, (a, b, k) = face_view(rgb, oval); fv = fv.copy()
pts = ((oval - [a, b]) / k).astype(np.int32); cv2.polylines(fv, [cv2.convexHull(pts)], True, (0, 255, 120), 2)
for p in pts: cv2.circle(fv, tuple(int(v) for v in p), 3, (255, 60, 60), -1)
crop36, crop72 = box72(rgb, oval)
blk = crop36.reshape(6, 6, 6, 6, 3).mean((1, 3)).astype(np.uint8)
fig, ax = plt.subplots(1, 4, figsize=(13, 4.3))
show(ax[0], fv, '1. FaceMesh: 36 điểm viền mặt (oval)')
show(ax[1], cv2.resize(crop36, (216, 216), interpolation=cv2.INTER_NEAREST), '2. Crop oval có che, 36×36\n(đầu vào CNN + POS/CHROM)', True)
show(ax[2], cv2.resize(blk, (216, 216), interpolation=cv2.INTER_NEAREST), '3. Trung bình màu lưới 6×6\n(gửi đi ở chế độ C5)', True)
show(ax[3], cv2.resize(crop72, (216, 216), interpolation=cv2.INTER_NEAREST), '4. Crop hộp không che, 72×72\n(đầu vào 3 mô hình)', True)
fig.tight_layout(); fig.savefig(OUT / 'fig1_frontend.png', dpi=110); plt.close(fig)
# ---------- fig 2: constraints ----------
rng = np.random.default_rng(0)
def view(img, l=None): return face_view(img, oval if l is None else l)[0]
c1p = co.c1(rgb, rng, gain=.125, poisson=True); c1g = co.c1(rgb, rng, gain=.25, gauss_var=.01)
aidv = F.aid(c1g)
vr = np.random.default_rng(99); cen = vr.choice(468, 8, replace=False); patches = [np.argsort(((lm - lm[c]) ** 2).sum(1))[:25] for c in cen]; vit = F.vit_apply(rgb, lm, patches)
H, Wd = rgb.shape[:2]; qv = co.c4_resize(rgb, co.c4_size_keep_aspect(Wd, H, (320, 240))); sq = 320 / Wd
qv_face = face_view(cv2.resize(qv, (Wd, H), interpolation=cv2.INTER_NEAREST), oval)[0]
us = cv2.filter2D(rgb, -1, X.KUS, borderType=cv2.BORDER_REPLICATE)
c7 = co.c7_mask(rgb, lm, "surgical_blue", "standard"); c7 = c7[0] if isinstance(c7, tuple) else c7; c7v, (a7, b7, k7) = face_view(c7, oval); c7v = c7v.copy()
cv2.polylines(c7v, [cv2.convexHull(((oval - [a7, b7]) / k7).astype(np.int32))], True, (0, 255, 120), 2)
other = next((jj for kk, jj in J.items() if jj['dataset'] == 'MCD' and kk.split('_')[1] != KEY.split('_')[1]), None)
by = rgb
if other:
    with np.load(other['meta']) as z2: n2 = len(z2['original_t'])
    by = next(iter(X.frames_of(other, n2)))
c3 = co.c3_composite(rgb, np.ascontiguousarray(by), layout="side", scale=.6); c3 = c3[0] if isinstance(c3, tuple) else c3
panels = [(view(rgb), 'Gốc'), (view(c1p), 'C1: sáng 12,5% + nhiễu Poisson'), (view(c1g), 'C1: sáng 25% + nhiễu Gauss mạnh'),
          (face_view(aidv, oval * (aidv.shape[1] / Wd))[0], 'C1: bản sao "aid" chỉ để dò mặt'), (view(vit), 'C2: mảng giống bạch biến'),
          (cv2.resize(c3, (420, int(420 * c3.shape[0] / c3.shape[1]))), 'C3: hai người trong khung (khóa người bên trái)'),
          (qv_face, 'C4: QVGA 320×240 (phóng to)'), (view(us), 'C6: unsharp (da lão hóa)'), (c7v, 'C7: khẩu trang, oval giữ bằng template')]
fig, ax = plt.subplots(3, 3, figsize=(11, 11))
for axx, (im, tt) in zip(ax.ravel(), panels): show(axx, im, tt, nearest='QVGA' in tt)
fig.tight_layout(); fig.savefig(OUT / 'fig2_constraints.png', dpi=100); plt.close(fig)
# ---------- fig 3/4: signal and readout on one 10 s window ----------
tr = np.array(tr, float); ok = np.isfinite(tr).all(1); tr[~ok] = np.nanmean(tr, 0)
q = t[0] + np.arange(int((t[-1] - t[0]) * 30)) / 30.; x = np.stack([np.interp(q, t, tr[:, c]) for c in range(3)], 1)
s0 = 10.; sel = (q >= q[0] + s0) & (q < q[0] + s0 + 10); xs = x[sel]; tt_ = q[sel] - q[sel][0]
def pos(x, fs=30, L=48):
    H = np.zeros(len(x))
    for n in range(len(x) - L + 1):
        c = x[n:n + L] / x[n:n + L].mean(0); S = c @ np.array([[0, 1, -1], [-2, 1, 1]]).T; h = S[:, 0] + S[:, 0].std() / S[:, 1].std() * S[:, 1]; H[n:n + L] += h - h.mean()
    return H
def detrend(z, lam=100):
    from scipy.sparse import spdiags, eye; from scipy.sparse.linalg import spsolve
    T = len(z); D = spdiags(np.array([np.ones(T), -2 * np.ones(T), np.ones(T)]), [0, 1, 2], T - 2, T); return z - spsolve(eye(T) + lam ** 2 * D.T @ D, z)
p_raw = pos(xs); p_dt = detrend(p_raw); bb, aa = signal.butter(1, [.7 / 15, 3 / 15], 'band'); p_bp = signal.filtfilt(bb, aa, p_dt)
f_, P = signal.periodogram(p_bp, 30, window='hann', nfft=4096); m = (f_ >= .7) & (f_ <= 3); bpm = f_[m] * 60; lz = np.log(P[m] + 1e-30); z = (lz - lz.mean()) / lz.std()
Z = np.load(W / 'out/z14_p4m_frozen.npz', allow_pickle=True); pid, sess = KEY.split('_')[1], KEY.split('_')[-1]; rec_m = [r for r in set(Z['rec'].tolist()) if pid in r and sess in r]
selz = (Z['cond'] == 'native') & np.isin(Z['rec'], rec_m); truth = Z['truth'][selz]; pred = Z['pred'][selz]; st = Z['start'][selz]; o = np.argsort(st)
hr_true = truth[np.argmin(np.abs(st - (st.min() + s0)))] if selz.any() else np.nan
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
for c, col in zip(range(3), 'rgb'): ax[0, 0].plot(tt_, xs[:, c] / xs[:, c].mean() - 1, col, lw=1)
ax[0, 0].set_title('a) Trung bình màu trong oval (R, G, B), 10 s'); ax[0, 0].set_xlabel('giây')
ax[0, 1].plot(tt_, p_raw / p_raw.std(), color='0.6', lw=1, label='POS thô'); ax[0, 1].plot(tt_, p_bp / p_bp.std(), 'C3', lw=1.5, label='sau khử xu hướng + lọc 0,7–3 Hz')
ax[0, 1].legend(fontsize=8); ax[0, 1].set_title('b) Sóng mạch POS: R2 khử xu hướng, R3 lọc dải'); ax[0, 1].set_xlabel('giây')
ax[1, 0].plot(bpm, z, 'C0'); ax[1, 0].axvline(hr_true, color='k', ls='--', label=f'nhịp tham chiếu {hr_true:.0f} bpm'); ax[1, 0].legend(fontsize=8)
ax[1, 0].set_title('c) R4–R5: log-phổ Hann, chuẩn hóa z (bằng chứng Z)'); ax[1, 0].set_xlabel('bpm')
if selz.any(): ax[1, 1].plot(st[o], truth[o], 'k-', lw=2, label='tham chiếu (PPG)'); ax[1, 1].plot(st[o], pred[o], 'C2o-', ms=3, label='P4m mỗi cửa sổ 10 s')
ax[1, 1].legend(fontsize=8); ax[1, 1].set_title('d) Kết quả P4m theo thời gian (bản ghi này)'); ax[1, 1].set_xlabel('giây'); ax[1, 1].set_ylabel('bpm')
fig.tight_layout(); fig.savefig(OUT / 'fig3_readout.png', dpi=100); plt.close(fig)
# ---------- fig 5: C5 split stream ----------
idx = sorted(i for i in frames if i in KEEP)[:8]
fig, ax = plt.subplots(2, 8, figsize=(13, 3.6))
for n, i in enumerate(idx):
    l = LM[i] if LM[i] is not None else lm; c36, c72 = box72(frames[i], l[co.OVAL]); b6 = c36.reshape(6, 6, 6, 6, 3).mean((1, 3)).astype(np.uint8)
    show(ax[0, n], cv2.resize(c72, (36, 36), interpolation=cv2.INTER_AREA), f't = {t[i] - t[0]:.1f} s', True); show(ax[1, n], b6, '', True)
ax[0, 0].set_ylabel('crop 36×36'); fig.suptitle('C5 chế độ tách: mỗi khung giữ lại (≤10 fps) gửi crop hộp 36×36 WebP không mất mát (~2,1 kB) + lưới 6×6 (432 B) + thời điểm (9 B)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'fig5_c5.png', dpi=100); plt.close(fig)
print('done', KEY, 'frames', N, 'fps %.2f' % fps, 'first face frame', i0, sorted(p.name for p in OUT.glob('*.png')))
