from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import detrend, periodogram
out=Path(__file__).resolve().parent
data=json.loads((out/"diagnostic.json").read_text())
waves=np.load(out/"five_clips.npz")
fig,axes=plt.subplots(5,3,figsize=(13,10),layout="constrained")
for i,row in enumerate(data["clips"]):
    target=waves[f"target{i}"].astype(float)
    raw=waves[f"raw{i}"].astype(float)
    t=np.arange(len(raw))/30.
    for arr,label,color in [(raw,"V4","#b84234"),(target,"PPG","#176386")]:
        wave=detrend(arr.cumsum())
        axes[i,0].plot(t,wave/wave.std(),color=color,label=label,lw=1)
        f,p=periodogram(arr,fs=30.,window="hann",detrend="linear")
        axes[i,1].plot(f,10*np.log10(np.maximum(p/p.max(),1e-10)),color=color,lw=1)
    axes[i,0].set_ylabel(row["rid"])
    axes[i,1].axvspan(.7,3,color="grey",alpha=.12)
    axes[i,1].set_ylim(-65,3)
    axes[i,2].plot(t[:40],(raw-raw.mean())[:40]*1e4,color="#b84234",label="clip")
    static=waves[f"static{i}"].astype(float)
    axes[i,2].plot(t[:40],(static-static.mean())[:40]*1e4,"--",color="#333333",label="static frame")
axes[0,0].set_title("Integrated waves, detrended and scaled")
axes[0,1].set_title("Raw difference PSD (dB relative to own peak)")
axes[0,2].set_title("Raw V4 minus DC, scaled by 10⁴")
for ax,x in zip(axes[-1],["Time (s)","Frequency (Hz)","Time (s)"]): ax.set_xlabel(x)
axes[0,0].legend(); axes[0,2].legend()
fig.savefig(out/"five_clips.png",dpi=150)
fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
for ds in ("UBFC2","MCD"):
    ax.plot([r["epoch"] for r in data["epochs"]],[r["dataset_train_loss"][ds] for r in data["epochs"]],label=ds+" train")
ax.scatter([30],[np.mean([r["val_loss"] for r in data["clips"]])],label="5 UBFC validation clips at epoch 30",marker="D",color="black")
ax.set(xlabel="Epoch",ylabel="Original V4 loss"); ax.legend()
fig.savefig(out/"loss_history.png",dpi=150)
