"""u22 (22 Nov, Claude): Task labels taken at another moment than the video (literature review 22/11, Q5a: heart-rate recovery after
exercise follows a first-order exponential, e.g. Pierpont et al. 2000). If the oximeter was read right after the task and the video
shows the recovery, the gap label - video HR should grow with how steeply HR is still falling inside the video.
Trajectory: FacePhys waveform (u17), 10 s windows, 2 s hop, periodogram peak within +-15 bpm of the video-level reader_v4 estimate
(cross-validated, so no label leaks), robust Theil-Sen slope in bpm/min.
Rule fixed before any result: correction pred = v4 + a + b*slope, (a, b) fitted by least squares on the TRAINING-fold Task camera
originals only, applied to the test fold. Accept only if, on the Task camera originals, the paired bootstrap CI (people, 10000, seed 0)
of the MAE change lies below 0 AND the change on all dev videos is negative. Descriptive: slope distribution, Spearman(slope, residual).
Usage: $PY code/u22_btc_task_recovery.py"""
import sys, pickle, json
from pathlib import Path
import numpy as np
from scipy import signal, stats
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
lab, per, low, task, fold, N, GRID = Y.lab, Y.per, Y.low, Y.task, Y.fold, Y.N, Y.GRID
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
keys = [J[INV.get(r, r)]['key'] for r in Y.rec]
U18 = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); assert list(U18['rec']) == list(Y.rec); v4 = U18['v4']; sub = U18['sub']
BP = signal.butter(3, [.7, 3.], btype='bandpass', fs=30, output='sos')
slope = np.full(N, np.nan); nwin = np.zeros(N, int); traj = {}
for i, k in enumerate(keys):
    p = W / 'out/u17/FacePhys.rlap' / (k + '.npz')
    if not p.exists() or not np.isfinite(v4[i]): continue
    with np.load(p) as z: bvp, ts = z['bvp'].astype(float), z['ts'].astype(float)
    if len(bvp) < 450: continue
    q = np.arange(ts[0], ts[-1], 1 / 30.); x = signal.sosfiltfilt(BP, np.interp(q, ts, bvp)); T = []; H = []
    for s in range(0, len(x) - 300 + 1, 60):
        seg = x[s:s + 300] * np.hanning(300); f = np.fft.rfftfreq(4096, 1 / 30.) * 60; P = np.abs(np.fft.rfft(seg, 4096)) ** 2
        m = (f >= max(42, v4[i] - 15)) & (f <= min(180, v4[i] + 15))
        if m.sum() < 3: continue
        T.append((s + 150) / 30. / 60.); H.append(f[m][P[m].argmax()])
    if len(T) >= 4: slope[i] = stats.theilslopes(H, T)[0]; nwin[i] = len(T); traj[k] = (np.array(T), np.array(H))
orig_task = sub & task & (low == 0) & np.isfinite(slope); orig_rest = sub & ~task & (low == 0) & np.isfinite(slope)
res = lab - v4
print(f'videos with a slope: Task originals {int(orig_task.sum())} ({len(set(per[orig_task]))} people), rest originals {int(orig_rest.sum())}')
for name, m in (('Task originals', orig_task), ('rest originals', orig_rest)):
    print(f'{name:15s} slope bpm/min median {np.median(slope[m]):+.2f} [{np.quantile(slope[m], .25):+.2f}; {np.quantile(slope[m], .75):+.2f}] | residual label-v4 median {np.median(res[m]):+.2f} | Spearman(slope, residual) {stats.spearmanr(slope[m], res[m])[0]:+.3f}')
pred = v4.copy(); coef = []
for f in sorted(set(fold)):
    tr = orig_task & (fold != f); te = orig_task & (fold == f)
    if tr.sum() < 5 or not te.any(): continue
    A = np.c_[np.ones(tr.sum()), slope[tr]]; a, b = np.linalg.lstsq(A, res[tr], rcond=None)[0]; coef.append((round(a, 2), round(b, 2)))
    pred[te] = v4[te] + a + b * slope[te]
print('fold coefficients (a, b):', coef)
S_ = lambda a_, b_, m: Y.summarize(a_[m], lab[m], per[m], b_[m])
for name, m in (('Task originals', orig_task), ('all dev (reader subset)', sub)):
    s = S_(pred, v4, m); print(f'{name:24s} videos {int(m.sum()):3d} people {s["people"]:3d} | v4 {S_(v4, Y.const, m)["MAE"]:6.2f} | v4 + recovery-slope correction {s["MAE"]:6.2f} ({s["delta"]:+.2f} [{s["dCI"][0]:+.2f};{s["dCI"][1]:+.2f}])')
np.savez_compressed(W / 'out/u22_task_recovery.npz', rec=Y.rec, slope=slope, nwin=nwin, pred=pred, v4=v4)
