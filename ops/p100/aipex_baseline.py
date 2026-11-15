"""AiPEX POS/GREEN baseline; bounded CPU workers; fixed protocol."""
import argparse,csv,json,math,concurrent.futures,multiprocessing
from pathlib import Path
import numpy as np
from hr_baseline import pos,estimate_hr,read_rgb,metrics,save

def process_record(task):
    root,out,group,subject,truth=task
    rois=[]
    try:
        for region in ("forehead","leftcheek","rightcheek"):
            video=root/"Data"/group/(subject+"_"+region+".avi")
            marker=video.with_name(video.name+".complete.json")
            if not marker.exists(): raise ValueError("Download incomplete: "+str(video.relative_to(root)))
            checksum=json.loads(marker.read_text())["sha256"]
            cache=out/(group+"_"+subject+"_"+region+"_rgb.npz")
            if cache.exists():
                with np.load(cache,allow_pickle=False) as c:
                    assert str(c["sha256"])==checksum
                    rgb=c["rgb"]; fs=float(c["fs"]); info=json.loads(str(c["info"]))
            else:
                rgb,fs,info=read_rgb(video)
                np.savez_compressed(cache,rgb=rgb,fs=fs,info=json.dumps(info),sha256=checksum)
            bvp=pos(rgb,fs);trim=math.ceil(1.6*fs);n=round(10*fs)
            pr=[];gr=[]
            for start in range(trim,len(rgb)-trim-n+1,n):
                pr.append(estimate_hr(bvp[start:start+n],fs))
                gr.append(estimate_hr(rgb[start:start+n,1],fs))
            if not pr: raise ValueError("No complete 10-second window")
            rois.append({"group":group,"subject":subject,"roi":region,"pos_bpm":float(np.mean(pr)),"green_bpm":float(np.mean(gr)),"sensor_hr_bpm":truth,"windows":len(pr),"fps":fs,"frames":len(rgb),"whole_record_pos_bpm":estimate_hr(bvp[trim:-trim],fs)})
        row={"group":group,"subject":subject,"pos_bpm":float(np.median([r["pos_bpm"] for r in rois])),"green_bpm":float(np.median([r["green_bpm"] for r in rois])),"sensor_hr_bpm":truth}
        return row,rois,None
    except Exception as e:
        return None,[],{"group":group,"subject":subject,"error":str(e)}

def run(root,out,workers):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];roirows=[];failures=[];tasks=[]
    for group in ("India","Sierra_Leone"):
        with (root/"Ground_Truth"/(group+"_GT.csv")).open(encoding="utf-8-sig") as f:
            labels=list(csv.reader(f))[1:]
        for values in labels:
            if values and values[0].strip(): tasks.append((root,out,group,values[0].strip(),float(values[1])))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for row,rois,failure in pool.map(process_record,tasks):
            if failure: failures.append(failure);print("FAILED",json.dumps(failure),flush=True)
            else: rows.append(row);roirows.extend(rois);print(json.dumps(row),flush=True)
    expected=len(tasks)
    summary={"dataset":"AiPEX rppg_biases","expected_recordings":expected,"processed":len(rows),"complete":len(rows)==expected,"coverage":len(rows)/max(expected,1),"protocol":"45-180 bpm; 1.6s POS; discard 1.6s each end; mean HR of nonoverlapping 10s windows for each ROI; median of 3 ROI HRs; fixed, no tuning; scalar sensor HR reference; distinct from UBFC PPG evaluation","pos_vs_sensor":metrics(rows,truth="sensor_hr_bpm"),"green_vs_sensor":metrics(rows,"green_bpm","sensor_hr_bpm"),"by_group":{g:metrics([r for r in rows if r["group"]==g],truth="sensor_hr_bpm") for g in ("India","Sierra_Leone")},"failures":failures}
    save(rows,out/"recordings.csv");save(roirows,out/"rois.csv")
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    summary=run(a.root,a.out,a.workers)
    raise SystemExit(0 if summary["complete"] else 2)

