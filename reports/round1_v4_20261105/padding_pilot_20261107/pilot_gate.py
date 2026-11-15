"""An explicit gate on the final, matched MCD-native pilot. Incomplete support fails closed."""
import csv, hashlib, json
from pathlib import Path
from statistics import mean
from collections import defaultdict
import math

def read(path):
    return json.loads(path.read_text())

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def check_epoch5(history):
    first_five=history[:5]
    if [r["epoch"] for r in first_five]!=list(range(1,6)):
        raise ValueError("Five consecutive completed epochs required")
    losses=[float(r["train_loss"]) for r in first_five]
    if not all(math.isfinite(v) and v>=0 for v in losses) or losses[0]<=0:
        raise ValueError("Invalid training loss")
    return dict(passed=losses[4]<=.90*losses[0], losses=losses,
                maximum_epoch5_loss=.90*losses[0], reduction=1-losses[4]/losses[0])

def score_pilot(report, out):
    joint=out/"fold0/seed2027/joint"
    status=read(joint/"status.json")
    assert status["state"]=="complete" and status["completed_epochs"]==30
    assert digest(joint/"final.pt")==status["checkpoint_sha256"]
    early=check_epoch5(read(joint/"history.json"))
    assert early["passed"] and early==read(out/"EPOCH5_GATE.json")
    condition=joint/"evaluation/MCD_native"
    reuse=read(condition/"REFERENCE_REUSE.json")
    assert digest(Path(reuse["path"]))==reuse["sha256"]
    assert digest(Path(reuse["raw_path"]))==reuse["raw_sha256"]
    metrics=read(condition/"metrics.json")
    pred=condition/"processed_output.csv"
    assert metrics["output_sha256"]==digest(pred) and metrics["reference_sha256"]==reuse["sha256"]
    def key(r):
        return r["subject_id"],r["recording_id"],float(r["window_start_s"])
    frozen=report.parent/"round1_final_20261105/revision_02/frozen_data/FROZEN_WINDOWS.json"
    base={key(r):r for r in read(frozen) if r["condition"]=="MCD native" and r["branch"]=="v3_joint" and r["fold"]==0 and r["seed"]==2027}
    with pred.open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==len(base)==408 and {key(r) for r in rows}==set(base)
    errs=defaultdict(list)
    controls=defaultdict(list)
    for r in rows:
        old=base[key(r)]
        accepted=old["reference_eligible"] and r["abstain"].lower()=="false"
        assert accepted==old["accepted"],"Window support differs"
        if accepted:
            p=r["subject_id"]; y=old["reference_hr_bpm"]; h=float(r["hr_bpm"])
            assert math.isfinite(h)
            errs[p].append(abs(h-y)); controls[p].append(abs(old["prediction_hr_bpm"]-y))
    assert len(errs)==12 and sum(map(len,errs.values()))==398
    mae=mean(mean(v) for v in errs.values()); control=mean(mean(v) for v in controls.values())
    assert abs(control-13.354737543650884)<1e-6
    assert abs(mae-metrics["subject_macro"]["mae"])<1e-6
    return dict(passed=mae<control,pilot_MAE=mae,V3_MAE=control,people=12,windows=398,
                checkpoint_sha256=status["checkpoint_sha256"],prediction_sha256=digest(pred),reference_sha256=reuse["sha256"])
