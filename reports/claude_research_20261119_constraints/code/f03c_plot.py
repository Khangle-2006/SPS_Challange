"""f03c - plot: the waveform each source outputs on the same 10 s window, against the contact PPG. AV env (matplotlib)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal, sparse
from scipy.sparse.linalg import factorized
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); O = W / 'out/f01'
N = 300; D = sparse.diags([np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)], [0, 1, 2], shape=(N - 2, N), format='csc')
TREND = factorized(sparse.eye(N, format='csc') + 100 ** 2 * (D.T @ D)); SOS = signal.butter(1, [.7, 3.], btype='bandpass', fs=30, output='sos')
filt = lambda v: signal.sosfiltfilt(SOS, np.asarray(v, float) - TREND(np.asarray(v, float)))
zn = lambda v: (v - v.mean()) / (v.std() + 1e-12)
def peak(v):
    f, p = signal.periodogram(v, fs=30, window='hann', nfft=4096); b = (f >= .7) & (f <= 3); return 60 * f[b][np.argmax(p[b])]
z = np.load(O / 'f03_waves_cnn.npz'); hr = float(z['truth']); tt = np.arange(N) / 30
rows = [('PPG tiếp xúc (cảm biến kẹp, dùng làm nhãn)', z['ppg'], 'k'),
        ('POS (công thức vật lý)', z['pos'], 'C0'), ('CHROM (công thức vật lý)', z['chrom'], 'C2'),
        ('CNN của đội (EfficientPhys+TSM, đã lọc 0,1 s)', z['cnn'], 'C4')]
for nm, lab in (('FacePhys.rlap', 'FacePhys (huấn luyện sẵn trên RLAP)'), ('ME-chunk.pure', 'ME-chunk (huấn luyện sẵn trên PURE)'), ('ME-flow.pure', 'ME-flow (huấn luyện sẵn trên PURE)')):
    rows.append((lab, np.load(O / f'f03_waves_{nm}.npz')['wave'], 'C1'))
ref = zn(filt(z['ppg']))
fig, ax = plt.subplots(len(rows), 1, figsize=(12, 13), sharex=True)
for a, (lab, raw, c) in zip(ax, rows):
    f = filt(raw); r = np.corrcoef(f, ref)[0, 1]
    a.plot(tt, zn(raw - np.polyval(np.polyfit(tt, raw, 1), tt)), color='0.75', lw=.9, label='sóng thô mô hình xuất ra')
    a.plot(tt, zn(f), color=c, lw=1.6, label='sau khử xu hướng + lọc 0,7–3 Hz')
    if c != 'k': a.plot(tt, ref, color='k', lw=.8, ls=':', alpha=.6, label='PPG tiếp xúc (để so)')
    note = f'tần số lặp lại mạnh nhất ≈ {peak(f):.0f} bpm'
    a.set_title(f'{lab} — {note}', fontsize=10, loc='left'); a.set_yticks([]); a.set_ylim(-3.3, 3.3)
ax[1].legend(fontsize=8, loc='upper right', ncol=3); ax[-1].set_xlabel('giây (trong cửa sổ 10 s)')
fig.suptitle(f'Đầu ra của từng nguồn trên CÙNG một đoạn 10 s (MCD, nhịp thật {hr:.0f} bpm): mỗi nguồn cho một SÓNG MẠCH, mỗi khung một số.\n'
             'Trục dọc đã chuẩn hóa (mỗi nguồn có thang riêng, không đơn vị). Nhịp tim = số lần sóng lên xuống trong một phút.', fontsize=11)
fig.text(.01, .005, 'Ghi chú: sóng trên mặt có thể lệch pha hoặc ngược dấu so với PPG ở ngón tay (máu tới mặt và tới ngón tay ở thời điểm khác nhau; mỗi mô hình có quy ước dấu riêng).\n'
         'Điều đó không ảnh hưởng nhịp tim, vì nhịp chỉ phụ thuộc sóng lặp lại nhanh hay chậm. Đường xám của CNN trôi dần vì mạng xuất ra độ thay đổi giữa hai khung và ta cộng dồn lại.', fontsize=8.5, va='bottom')
fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(O / 'fig6_waves.png', dpi=110)
for lab, raw, c in rows: print(lab, 'peak %.1f' % peak(filt(raw)), 'r %.2f' % np.corrcoef(filt(raw), ref)[0, 1])
