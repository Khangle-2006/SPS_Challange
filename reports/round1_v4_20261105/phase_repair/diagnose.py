"""Postmortem for the single stopped V4 run, using public dev clips only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, csv, json
from pathlib import Path
import numpy as np
import torch
from scipy import signal

OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
ROOT = REPORT.parents[1]
V3CODE = Path("/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102")
sys.path[:0] = [str(V3CODE), str(REPORT/"code")]
from t8_plan import read, sha, pools, draw
from t8_evaluate import decode_difference
from polarity_loader import load_canonical_clip
from EfficientPhys_fixed import EfficientPhys
from phase_loss import phase_loss

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
index = read(ROOT/"t8_oval_20261031/training_index.json")
fold = Path("fold0/seed2027/joint")
runs = {"v4": REPORT/"v4_phase"/fold, "v3": ROOT/"t8_polarity_v3_20261102"/fold}
models = {}
for name, run in runs.items():
    checkpoint = run/"final.pt"
    assert sha(checkpoint) == read(run/"status.json")["checkpoint_sha256"]
    net = EfficientPhys(frame_depth=10,img_size=36)
    net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    models[name] = net.eval()

def load(ds,rid,start,variant):
    return load_canonical_clip(Path(index["roots"][ds])/rid/variant/"crop", start, ds)[:2]

def spectral(wave):
    freq, power = signal.periodogram(wave,fs=30.,window="hann",detrend="linear")
    band = (freq>=0.7)&(freq<=3)
    return dict(peak=float(freq[power.argmax()]),above3=float(power[freq>3].sum()/power.sum()),
                inband=float(power[band].sum()/power.sum()),slow=float(power[freq<.7].sum()/power.sum()))

def repeated_pattern_fraction(wave):
    wave = np.asarray(wave, dtype=np.float64)
    blocks = wave.reshape(-1,10)
    return float(np.var(blocks.mean(0))/np.var(blocks))

v4 = models["v4"]
raw = {}
results=[]
with np.load(runs["v4"]/"evaluation/UBFC2_native/waveforms.npz") as archive:
    saved = {(str(rid),float(start)):w for (rid,start),w in zip(archive["keys"],archive["difference"])}
selected=sorted({rid for rid,_ in saved},key=lambda s:int(s.removeprefix("subject")))[:5]
with torch.no_grad():
    for i,rid in enumerate(selected):
        start = min(t for rec,t in saved if rec==rid)
        x,y = load("UBFC2",rid,start,"native")
        xt=torch.from_numpy(x[None])
        yhat={k:model(xt)[0].numpy().copy() for k,model in models.items()}
        diff=yhat["v4"]
        flat=v4(xt[:,:1].expand_as(xt).contiguous())[0].numpy().copy()
        cpu_gpu_max=float(np.max(np.abs(diff-saved[rid,start])))
        target=torch.from_numpy(y[None]); pred=torch.from_numpy(diff[None])
        row=dict(rid=rid,start=start,raw_mean=float(diff.mean()),raw_std=float(diff.std()),
                 target_hr=decode_difference(y),v4_hr=decode_difference(diff),v3_hr=decode_difference(yhat["v3"]),
                 cpu_gpu_max=cpu_gpu_max,saved_hr=decode_difference(saved[rid,start]),
                 raw_psd=spectral(diff),integrated_psd=spectral(np.cumsum(diff,dtype=float)),
                 target_integrated_psd=spectral(np.cumsum(y,dtype=float)),phase10_fraction=repeated_pattern_fraction(diff),
                 static_input_raw_std=float(flat.std()),static_phase10_fraction=repeated_pattern_fraction(flat),
                 static_hr=decode_difference(flat),val_loss=float(phase_loss(pred,target)),
                 same=float(phase_loss(target,target)),opposite=float(phase_loss(-target,target)),
                 shifted10=float(phase_loss(torch.roll(target,10,-1),target)))
        assert abs(row["v4_hr"]-row["saved_hr"])<.01
        for k,arr in dict(raw=diff,target=y,v3=yhat["v3"],static=flat).items():raw[f"{k}{i}"]=arr
        results.append(row)
    pool=pools(index,0)
    train=[load(*item)[0] for j in range(4) for item in draw(pool,2027,0,0,j,"joint")]
    val=[load("UBFC2",row["rid"],row["start"],"native")[0] for row in results]
    bn=v4.batch_norm
    stats=dict(running_mean=bn.running_mean.tolist(),running_var=bn.running_var.tolist())
    for label,clips in [("train_balanced8",train),("val_clean5",val)]:
        batch=torch.from_numpy(np.stack(clips))
        data=batch.diff(dim=1)
        stats[label]=dict(mean=data.mean(dim=(0,1,3,4)).tolist(),var=data.var(dim=(0,1,3,4),unbiased=False).tolist())
    # Enable batch statistics alone; dropout stays in evaluation mode.
    bn.train()
    bpred=v4(batch).numpy()
    stats["val_hr_with_batch_stats"]=[decode_difference(p) for p in bpred]
    bn.eval()
t=torch.arange(300)/30.
y=(torch.sin(2*torch.pi*1.3*t)+.3*torch.sin(2*torch.pi*2.6*t)).diff(prepend=torch.zeros(1))[None]
tests={k:float(phase_loss(x,y)) for k,x in [("same",y),("opposite",-y),("shift10",torch.roll(y,10,-1))]}
assert tests["same"]<1e-5 and min(tests["opposite"],tests["shift10"])>tests["same"]+.1
history=read(runs["v4"]/"history.json")
result=dict(clips=results,bn=stats,tests=tests,epochs=[{k:r[k] for k in ("epoch","train_loss","dataset_train_loss")} for r in history],
            checkpoint=sha(runs["v4"]/"final.pt"),val_history="Only final epoch checkpoint available")
np.savez_compressed(OUT/"five_clips.npz",**raw)
(OUT/"diagnostic.json").write_text(json.dumps(result,indent=2))
print(json.dumps(dict(clips=results,bn=stats,tests=tests,loss=[history[i]["train_loss"] for i in (0,14,29)])),flush=True)
