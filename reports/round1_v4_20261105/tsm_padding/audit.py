"""Exact same stored weights and decoder for each A/B; CPU and dev only."""
from test_static_tsm import static_result, EfficientPhys, use_replicate_padding, REMOTE
import torch,json,sys
from pathlib import Path
from t8_plan import sha,read
from t8_evaluate import decode_difference
from polarity_loader import load_canonical_clip
from collections import defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent
ROOT=REPORT.parents[1]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
def save(filename,value):
    tmp=HERE/(filename+".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(HERE/filename)

def build(run):
    ck=run/"final.pt"
    h=sha(ck)
    assert h==read(run/"status.json")["checkpoint_sha256"]
    m=EfficientPhys(frame_depth=10,img_size=36)
    m.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"])
    return m.eval(),h

results=[]
for branch,base,pairs in [("v3",ROOT/"t8_polarity_v3_20261102",[(f,s) for f in range(5) for s in (2027,2028,2029)]),("v4",REPORT/"v4_phase",[(0,2027)])]:
    for fold,seed in pairs:
        model,h=build(base/f"fold{fold}"/f"seed{seed}"/"joint")
        original=static_result(model)
        repaired=static_result(use_replicate_padding(model))
        results.append(dict(branch=branch,fold=fold,seed=seed,checkpoint=h,original=original,replicate=repaired))
        save("static_results.json",dict(source_sha256=sha(REMOTE/"EfficientPhys_fixed.py"),runs=results))
        print(branch,fold,seed,original,repaired,flush=True)

# Single matched fold, four predeclared conditions. Keep weights fixed; reuse frozen
# OOF rows and record every eligible row, any newly rejected prediction and coverage.
base=ROOT/"t8_polarity_v3_20261102/fold0/seed2027/joint"
model,h=build(base); use_replicate_padding(model)
index=read(ROOT/"t8_oval_20261031/training_index.json")
frozen=ROOT/"reports/round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json"
rows=[r for r in read(frozen) if r["branch"]=="v3_joint" and r["fold"]==0 and r["seed"]==2027]
conditions={"MCD native":("MCD","native"),"MCD-C5":("MCD","f10_b500"),"UBFC clean":("UBFC2","native"),"UBFC C5":("UBFC2","f10_b500")}
stats={}; output=[]
for cond,(ds,var) in conditions.items():
    all_rows=[r for r in rows if r["condition"]==cond]
    errors=defaultdict(list); high_ref=0; failures=0
    for row in all_rows:
        if not row["accepted"]:continue
        rid=row["recording_id"]; start=row["window_start_s"]
        x,*_=load_canonical_clip(Path(index["roots"][ds])/rid/var/"crop",start,ds)
        with torch.inference_mode(): wave=model(torch.from_numpy(x[None]))[0].numpy()
        try: new=decode_difference(wave); reason=None
        except ValueError as exc: new=None; reason=str(exc); failures+=1
        ref=row["reference_hr_bpm"]; old=row["prediction_hr_bpm"]
        output.append(dict(condition=cond,subject=row["subject_id"],record=rid,start=start,reference=ref,original=old,replicate=new,reason=reason))
        if new is not None:errors[row["subject_id"]].append([abs(new-ref),abs(old-ref)])
    macro=np.mean([np.mean(v,axis=0) for v in errors.values()],axis=0)
    stats[cond]=dict(candidates=len(all_rows),original_accepted=sum(r["accepted"] for r in all_rows),common=sum(map(len,errors.values())),new_failures=failures,replicate_MAE=macro[0],original_MAE=macro[1])
    save("matched_fold0.json",dict(checkpoint=h,conditions=stats,complete=len(stats)==4))
    save("matched_rows.json",output)
    print(cond,stats[cond],flush=True)
