"""Public sets (UBFC2 dev 35 people, MCD FullHD 119 recordings / 60 people): build per-window spectra from the SAVED waveforms of
reports/round1_execution_20261110/inference_repair_20261113 (padding-fixed CNN out-of-fold, POS, CHROM), with the frozen readout operator
(cumsum for the CNN, smoothness-prior detrend lambda=100, Butterworth order 1 0.7-3 Hz, Hann periodogram nfft 4096, band 0.7-3 Hz).
No training, no new inference. Checks that the argmax reproduces the saved 'power_0.7' / '0.7' readouts. Output: out/public_spectra.npz"""
import json
from pathlib import Path
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import factorized
R = Path('/mnt/samsung_t5/khangle/facevitals/reports/round1_execution_20261110/inference_repair_20261113')
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118')
N = 300
D = sparse.diags([np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)], [0, 1, 2], shape=(N - 2, N), format='csc')
TREND = factorized(sparse.eye(N, format='csc') + 100 ** 2 * (D.T @ D))
SOS = signal.butter(1, [.7, 3.], btype='bandpass', fs=30, output='sos')
F = np.fft.rfftfreq(4096, 1 / 30.); BAND = (F >= .7) & (F <= 3.); BPM = 60 * F[BAND]
def spectrum(w):
    w = np.asarray(w, float); y = signal.sosfiltfilt(SOS, w - TREND(w)); f, p = signal.periodogram(y, fs=30, window='hann', nfft=4096, detrend=False)
    return p[BAND]
rows = []; S = {'pad': [], 'pos': [], 'chrom': []}; mism = {'pad': 0, 'pos': 0, 'chrom': 0}; cnt = 0
for j in json.loads((R / 'MANIFEST.json').read_text()):
    if j['dataset'] == 'BTC': continue
    for cond in ['native', 'c5', 'c5prime_default', 'c5prime_bf0']:
        a = R / 'original' / j['key'] / cond; b = R / 'classical_original_v2' / j['key'] / cond
        if not (a / 'waveforms.npz').exists() or not (b / 'classical_waves.npz').exists() or (a / 'EXCLUDED.json').exists(): continue
        P = [r for r in json.loads((a / 'PREDICTIONS.json').read_text()) if r['model'].startswith('padding') and r['truth'] is not None and r['readouts']['power_0.7'] is not None]
        C = {(r['model'], round(r['start'], 3)): r for r in json.loads((b / 'CLASSICAL.json').read_text()) if r['truth'] is not None and r['readouts']['0.7'] is not None}
        zw = np.load(a / 'waveforms.npz'); wk = {(m, round(float(s), 3)): i for i, (m, s) in enumerate(zw['keys'].tolist())}
        zc = np.load(b / 'classical_waves.npz'); ck = {(m, bnd, round(float(s), 3)): i for i, (m, bnd, s) in enumerate(zc['keys'].tolist())}
        for r in P:
            st = round(r['start'], 3)
            if (r['model'], st) not in wk or ('POS', st) not in C or ('CHROM', st) not in C or ('POS', '0.7', st) not in ck: continue
            sp = spectrum(np.cumsum(zw['difference'][wk[(r['model'], st)]])); so = spectrum(zc['waves'][ck[('POS', '0.7', st)]]); sh = spectrum(zc['waves'][ck[('CHROM', '0.7', st)]])
            cnt += 1
            mism['pad'] += abs(BPM[sp.argmax()] - r['readouts']['power_0.7']) > 1e-6; mism['pos'] += abs(BPM[so.argmax()] - C[('POS', st)]['readouts']['0.7']) > 1e-6; mism['chrom'] += abs(BPM[sh.argmax()] - C[('CHROM', st)]['readouts']['0.7']) > 1e-6
            rows.append((j['dataset'], cond, j['person'], int(j['fold']), j['recording'], st, r['truth'])); S['pad'].append(sp); S['pos'].append(so); S['chrom'].append(sh)
print('windows', cnt, '| argmax differs from saved readout:', mism)
np.savez_compressed(W / 'out/public_spectra.npz', dataset=np.array([r[0] for r in rows]), cond=np.array([r[1] for r in rows]), person=np.array([r[2] for r in rows]), fold=np.array([r[3] for r in rows]),
                    recording=np.array([r[4] for r in rows]), start=np.array([r[5] for r in rows]), truth=np.array([r[6] for r in rows]), bpm=BPM, **{k: np.array(v, np.float32) for k, v in S.items()})
