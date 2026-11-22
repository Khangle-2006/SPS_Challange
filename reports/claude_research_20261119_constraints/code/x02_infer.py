"""x02 - public sets (UBFC2 dev, MCD): per 10 s window spectra of the padding-fixed CNN (out-of-fold checkpoint), POS and CHROM for every
condition produced by x01 (or any directory with the same layout, e.g. C5 / C3 extractions). No training. CPU only (1 thread per process).
Same windows, eligibility rules, clip construction, vendor POS/CHROM functions and readout operator as inference_repair_20261113.
Extra, label-free: occlusion-aware variants - rows of the 6x6 grid whose colour departs from the forehead reference are excluded from the
POS/CHROM trace and zeroed in the CNN input (as pixels outside the oval already are).
Consumes <crops>/<cond>/<key>.npz and deletes it after the result is written.
Usage: $PY code/x02_infer.py <slot> <nslots> [tag=x01] [wait_flag]"""
import os, sys
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json, time, traceback
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints')
R = Path('/mnt/samsung_t5/khangle/facevitals/reports/round1_execution_20261110/inference_repair_20261113')
sys.path.insert(0, str(R)); sys.path.insert(0, str(W / 'code'))
from common_repair import read                      # also puts the team's code directories on sys.path (read-only imports)
from classical_repair_v2 import functions
from t8_t1_loader import supported
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
import torch
from scipy import signal, sparse
from scipy.sparse.linalg import factorized
torch.set_num_threads(1); torch.set_num_interop_threads(1)
N = 300; D = sparse.diags([np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)], [0, 1, 2], shape=(N - 2, N), format='csc')
TREND = factorized(sparse.eye(N, format='csc') + 100 ** 2 * (D.T @ D)); SOS = signal.butter(1, [.7, 3.], btype='bandpass', fs=30, output='sos')
F = np.fft.rfftfreq(4096, 1 / 30.); BAND = (F >= .7) & (F <= 3.); BPM = 60 * F[BAND]
def spectrum(w):
    w = np.asarray(w, float)
    if w.shape != (N,) or not np.isfinite(w).all() or w.std() < 1e-10: return None
    y = signal.sosfiltfilt(SOS, w - TREND(w)); p = signal.periodogram(y, fs=30, window='hann', nfft=4096, detrend=False)[1][BAND]
    return p if np.isfinite(p).all() and p.sum() > 1e-30 else None
from xcommon import detect_rows, upper_trace
def merge(t, *arrays):
    u, inv, cnt = np.unique(t, return_inverse=True, return_counts=True); out = []
    for a in arrays:
        if a.dtype == bool:
            m = np.ones(len(u), bool); np.logical_and.at(m, inv, a)
        else:
            m = np.zeros((len(u),) + a.shape[1:], np.float32); np.add.at(m, inv, a); m /= cnt.reshape((-1,) + (1,) * (a.ndim - 1))
        out.append(m)
    return [u] + out
_M = {}
def model(fold):
    if fold not in _M:
        a = read(R / 'FROZEN.json')['padding_checkpoints'][str(fold)]; m = use_replicate_padding(EfficientPhys(frame_depth=10, img_size=36))
        m.load_state_dict(torch.load(a['path'], map_location='cpu', weights_only=True)['model'], strict=True); _M[fold] = m.eval()
    return _M[fold]
def cnn(fold, raw):
    """raw [301,36,36,3] float32 -> difference waveform [300], with the team's clip standardisation."""
    if not np.isfinite(raw).all() or raw.std() < 1e-6: return None
    x = ((raw - raw.mean()) / raw.std()).transpose(0, 3, 1, 2).copy()
    with torch.inference_mode(): return model(fold)(torch.from_numpy(x[None]))[0].double().numpy()
FN = functions(.7)
def classical(tu, tr, start):
    q = start + np.arange(N) / 30; x = np.stack([np.interp(q, tu, tr[:, k]) for k in range(3)], 1); out = []
    for m in ('pos', 'chrom'):
        try: out.append(spectrum(FN[m](x, 30)))
        except Exception: out.append(None)
    return out
def process(j, cond, xdir, cdir, odir, native_dir):
    with np.load(j['meta']) as z: ref_valid, ppg_t, ppg_valid = z['valid'], z['ppg_t'], z['ppg_valid']
    with np.load(xdir / cond / (j['key'] + '.npz')) as z: t, valid, blk, fill = z['t'], z['valid'], z['blk'], z['fill']
    with np.load(cdir / cond / (j['key'] + '.npz')) as z: crop = z['crop']
    tu, crop, valid, blk, fill = merge(t, crop, valid, blk, fill.astype(np.float32) / 36.)
    if (native_dir / (j['key'] + '.npz')).exists():
        with np.load(native_dir / (j['key'] + '.npz')) as z: _, nv = merge(z['t'], z['valid'])
        if len(nv) == len(ref_valid) == len(valid): valid = valid & ~(nv & ~ref_valid)       # frames the team's cache marks unusable at source (corrupt decode)
    S = {k: [] for k in ('pad', 'pos', 'chrom', 'pad_oa', 'pos_oa', 'chrom_oa')}; st = []; tr = []; nrow = []; skipped = dict(ineligible=0, image=0, ppg=0, decode=0)
    mean = blk.mean((1, 2))
    for w in j['windows']:
        if not w['eligible']: skipped['ineligible'] += 1; continue
        s = w['start']; q = s + np.arange(301) / 30
        if not supported(tu, valid, q[0], q[-1]): skipped['image'] += 1; continue
        if not supported(ppg_t, ppg_valid, s, s + 10): skipped['ppg'] += 1; continue
        ix = np.searchsorted(tu, q, side='right').clip(1, len(tu) - 1); a = ((q - tu[ix - 1]) / (tu[ix] - tu[ix - 1])).astype(np.float32)
        raw = crop[ix - 1] * (1 - a[:, None, None, None]) + crop[ix] * a[:, None, None, None]
        wv = cnn(j['fold'], raw); sp = None if wv is None else spectrum(np.cumsum(wv)); so, sh = classical(tu, mean, s)
        if sp is None or so is None or sh is None: skipped['decode'] += 1; continue
        sel = (tu >= s) & (tu <= s + 10); rows = detect_rows(blk[sel].mean(0), fill[sel].mean(0))
        if rows:
            r2 = raw.copy()
            for i in rows: r2[:, 6 * i:6 * i + 6] = 0
            wv2 = cnn(j['fold'], r2); sp2 = None if wv2 is None else spectrum(np.cumsum(wv2)); so2, sh2 = classical(tu, upper_trace(blk, rows), s)
            sp2 = sp if sp2 is None else sp2; so2 = so if so2 is None else so2; sh2 = sh if sh2 is None else sh2
        else: sp2, so2, sh2 = sp, so, sh
        for k, v in zip(S, (sp, so, sh, sp2, so2, sh2)): S[k].append(v)
        st.append(s); tr.append(w['truth']); nrow.append(len(rows))
    (odir / cond).mkdir(parents=True, exist_ok=True); tmp = odir / cond / (j['key'] + '.tmp.npz')
    np.savez_compressed(tmp, start=np.array(st), truth=np.array(tr), nrow=np.array(nrow, np.int8), skipped=json.dumps(skipped), valid_fraction=float(valid.mean()),
                        **{k: np.array(v, np.float32).reshape(len(st), len(BPM)) for k, v in S.items()})
    tmp.replace(odir / cond / (j['key'] + '.npz')); (cdir / cond / (j['key'] + '.npz')).unlink()
if __name__ == '__main__':
    slot, nslots = int(sys.argv[1]), int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else 'x01'; flag = W / 'out' / (sys.argv[4] if len(sys.argv) > 4 else tag + '_DONE')
    xdir, cdir, odir = W / 'out' / tag, W / 'out' / (tag + 'c'), W / 'out' / (tag + '_spec'); native_dir = W / 'out/x01/native'
    jobs = [j for j in json.loads((R / 'MANIFEST.json').read_text()) if j['dataset'] != 'BTC']; byk = {j['key']: j for j in jobs}; idle = 0
    while True:
        todo = [(p.parent.name, p.name[:-4]) for p in sorted(cdir.glob('*/*.npz')) if not p.name.endswith('.tmp.npz')]
        todo = [(c, k) for c, k in todo if (sum(map(ord, k)) % nslots) == slot and (tag != 'x01' or (native_dir / (k + '.npz')).exists()) and (xdir / c / (k + '.npz')).exists()]
        if not todo:
            if flag.exists():
                idle += 1
                if idle >= 2: break
            time.sleep(20); continue
        idle = 0
        for c, k in todo:
            try: process(byk[k], c, xdir, cdir, odir, native_dir); print(c, k, 'ok', flush=True)
            except Exception as e:
                print(c, k, 'ERROR', repr(e), traceback.format_exc()[-800:], flush=True); (cdir / c / (k + '.npz')).rename(cdir / c / (k + '.failed'))
    print('slot', slot, 'finished', flush=True)
