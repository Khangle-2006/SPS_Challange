"""f02 - illustration (internal Notion): the four evidence channels of P4m and their fusion on one real MCD window."""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
W = Path("/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints"); KEY = sys.argv[1] if len(sys.argv) > 1 else "MCD_1115_FullHDwebcam_before"
F = np.fft.rfftfreq(4096, 1 / 30.); BPM = 60 * F[(F >= .7) & (F <= 3.)]
zs = lambda S: (np.log(S + 1e-30) - np.log(S + 1e-30).mean()) / np.log(S + 1e-30).std()
with np.load(W / "out/x13_spec/nat_ma3" / (KEY + ".npz")) as z: st, tr, cnn, pos, chrom = z["start"], z["truth"], z["pad"], z["pos"], z["chrom"]
EX = []
for m in ["FacePhys.rlap", "ME-chunk.pure", "ME-flow.pure"]:
    with np.load(W / "out/y12" / m / (KEY + ".npz")) as z: EX.append(dict(zip(np.round(z["start"], 3), z["spec"])))
i = int(np.argmin(np.abs(st - (st.min() + 10)))); s = round(float(st[i]), 3); hr = tr[i]
ch = {"CNN của đội (đã lọc 0,1 s)": zs(cnn[i]), "POS (công thức)": zs(pos[i]), "CHROM (công thức)": zs(chrom[i])}
names = ["FacePhys", "ME-chunk", "ME-flow"]; ex = [zs(e[s]) for e in EX]; ch["3 mô hình (trung bình)"] = np.mean(ex, 0)
w = {"CNN của đội (đã lọc 0,1 s)": 1, "POS (công thức)": 0, "CHROM (công thức)": .25, "3 mô hình (trung bình)": 1}
E = sum(w[k] * v for k, v in ch.items())
fig, ax = plt.subplots(2, 3, figsize=(13, 6.5)); ax = ax.ravel()
for a, (k, v) in zip(ax, ch.items()):
    a.plot(BPM, v, lw=1.3); a.axvline(hr, color="k", ls="--", lw=1); pk = BPM[np.argmax(v)]; a.set_title(f"{k}\nđỉnh {pk:.0f} bpm, trọng số {w[k]}", fontsize=10); a.set_xlabel("bpm")
a = ax[4]
for n, e in zip(names, ex): a.plot(BPM, e, lw=1, label=n)
a.axvline(hr, color="k", ls="--", lw=1); a.legend(fontsize=8); a.set_title("Từng mô hình trong nhóm 3 mô hình", fontsize=10); a.set_xlabel("bpm")
a = ax[5]; a.plot(BPM, E, "C3", lw=2); a.axvline(hr, color="k", ls="--", lw=1, label=f"nhịp thật {hr:.0f} bpm"); a.legend(fontsize=8)
a.set_title(f"GỘP: CNN + 0,25·CHROM + 3 mô hình\nđỉnh {BPM[np.argmax(E)]:.0f} bpm", fontsize=10); a.set_xlabel("bpm")
fig.suptitle(f"Cùng một đoạn 10 s của một bản ghi MCD: mỗi cách đọc cho một đường \"bằng chứng\"; gộp lại thì đỉnh rõ hơn (vạch đứt = nhịp đo bằng PPG)", fontsize=11)
fig.tight_layout(); fig.savefig(W / "out/f01/fig4_evidence.png", dpi=100); print("ok", s, hr)
