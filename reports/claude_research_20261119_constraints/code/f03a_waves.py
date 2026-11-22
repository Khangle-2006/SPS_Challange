"""f03a - illustration (internal Notion): the WAVEFORM each source outputs on one 10 s window of one MCD recording (CNN, POS, CHROM,
plus the contact PPG). PY env (torch). Reuses x02_infer functions unchanged. Writes out/f01/f03_waves_cnn.npz."""
import sys, json, numpy as np
from pathlib import Path
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code'))
import x02_infer as X
KEY = sys.argv[1] if len(sys.argv) > 1 else 'MCD_1115_FullHDwebcam_before'
j = {j['key']: j for j in json.loads((X.R / 'MANIFEST.json').read_text())}[KEY]
crop = np.asarray(np.load(Path(j['crop']) / 'rgb.npy', mmap_mode='r'), np.float32)
with np.load(j['meta']) as z: t, valid, ppg_t, ppg = z['t'], z['valid'].astype(bool), z['ppg_t'], z['ppg']
N = len(t); fps = (N - 1) / (t[-1] - t[0]); k = max(1, int(np.ceil(fps / 10 - 1e-3))); w = valid.astype(np.float32); pad = k // 2
cs = np.cumsum(np.concatenate([np.zeros((1, 36, 36, 3)), crop * w[:, None, None, None]]), 0); cw = np.concatenate([[0.], np.cumsum(w)])
lo = np.clip(np.arange(N) - pad, 0, N); hi = np.clip(np.arange(N) - pad + k, 0, N); ma3 = ((cs[hi] - cs[lo]) / np.maximum(cw[hi] - cw[lo], 1)[:, None, None, None]).astype(np.float32)
el = [x for x in j['windows'] if x['eligible']]; st = np.array([x['start'] for x in el]); i = int(np.argmin(np.abs(st - (st.min() + 10)))); s = float(st[i]); hr = float(el[i]['truth'])
q = s + np.arange(301) / 30; tu, cu, vu = X.merge(t, ma3, valid)
ix = np.searchsorted(tu, q, side='right').clip(1, len(tu) - 1); a = ((q - tu[ix - 1]) / (tu[ix] - tu[ix - 1])).astype(np.float32)
raw = cu[ix - 1] * (1 - a[:, None, None, None]) + cu[ix] * a[:, None, None, None]
d = X.cnn(j['fold'], raw); cnn = np.cumsum(d)
_, cr = X.merge(t, crop); mean = cr.reshape(len(cr), -1, 3).mean(1)   # = blk.mean((1,2)) of the unfiltered crop
qq = s + np.arange(300) / 30; x = np.stack([np.interp(qq, tu, mean[:, c]) for c in range(3)], 1)
pos = X.FN['pos'](x, 30); chrom = X.FN['chrom'](x, 30)
filt = lambda v: X.signal.sosfiltfilt(X.SOS, np.asarray(v, float) - X.TREND(np.asarray(v, float)))
pp = np.interp(qq, ppg_t, ppg)
np.savez(W / 'out/f01/f03_waves_cnn.npz', start=s, truth=hr, fps=fps, k=k, rgb=x, cnn_diff=d, cnn=cnn, pos=pos, chrom=chrom, ppg=pp,
         cnn_f=filt(cnn), pos_f=filt(pos), chrom_f=filt(chrom), ppg_f=filt(pp))
print('ok', KEY, 'start', s, 'truth', hr, 'fps %.2f k %d' % (fps, k), 'fold', j['fold'])
