"""Label-blind subset timing provenance audit. Never fit a time scale to HR."""
import csv,hashlib,json
from pathlib import Path
import cv2
import numpy as np
R=Path('/data_pool/khangle/facevitals');DATA=R/'datasets/AiPEX';OLD=R/'results/aipex_audit_v1';OUT=R/'results/aipex_timing_gate_20261019'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    manifest=json.loads((DATA/'source_manifest.json').read_text());files=manifest['files']
    possible_timing=[f['path'] for f in files if any(w in f['path'].lower() for w in ['timestamp','sync','ppg','bvp','ecg'])]
    prior=json.loads((OLD/'rois.json').read_text());selected=[]
    for g in ['India','Sierra_Leone']:
        rows=sorted([x for x in prior if x['group']==g and x['roi']=='forehead'],key=lambda x:(x['container_duration_s'],x['subject']))
        selected += [rows[0],rows[len(rows)//2],rows[-1]]
    protocol={'selection':'shortest, upper-median and longest container duration in each group; forehead used for selection, no HR label selection',
       'source_commit':manifest['commit'],'selected':[{'group':r['group'],'subject':r['subject']} for r in selected],
       'scope':'inspect all three ROI headers and selected decoded temporal traces; not model evaluation',
       'forbidden':'No FPS/time-scale/offset search against HR labels. No waveform metrics against scalar mean-HR labels.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));records=[]
    for r in selected:
        videos=[]
        for region in ['forehead','leftcheek','rightcheek']:
            p=DATA/'Data'/r['group']/f"{r['subject']}_{region}.avi";marker=json.loads(p.with_name(p.name+'.complete.json').read_text());assert sha(p)==marker['sha256']
            cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));pts=[];rgb=[]
            while True:
                ok,f=cap.read()
                if not ok:break
                pts.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000);rgb.append(cv2.mean(f)[:3][::-1])
            cap.release();assert len(pts)==declared
            a=np.array(rgb);old=np.load(R/'results/aipex_pos_green_v1'/f"{r['group']}_{r['subject']}_{region}_rgb.npz")
            np.testing.assert_allclose(a,old['rgb'],rtol=0,atol=1e-10)
            videos.append({'region':region,'frames':declared,'fps':fps,'container_duration_s':declared/fps,'pts_nondecreasing':bool(np.all(np.diff(pts)>0)),
                'container_grid_max_abs_error_s':float(np.max(np.abs(np.array(pts)-np.arange(declared)/fps))),
                'recorded_acquisition_timestamps_available':False,'file_sha256':marker['sha256']})
        assert len({v['frames'] for v in videos})==1
        records.append({'group':r['group'],'subject':r['subject'],'videos':videos})
        print('AUDITED',r['group'],r['subject'],flush=True)
    result={'complete':True,'selected_recordings':6,'decoded_videos':18,'records':records,'source_commit':manifest['commit'],
       'possible_timing_files_in_released_manifest':possible_timing,
       'reference_type':'one scalar mean HR per recording; no synchronized reference waveform',
       'container_integrity_passed':True,'physiological_time_mapping_confirmed':False,'independent_model_benchmark_gate_passed':False,
       'gate_reason':'Uniform 15fps container timestamps do not establish original acquisition elapsed time. Released manifest lacks original frame timestamps or an export mapping; collection paper describes ~25fps. No label-based timing calibration is permitted.',
       'consequence':'Do not rank frozen models by AiPEX HR error until timing provenance is resolved. No model training/inference was performed on this dataset in this audit.',
       'sources':['https://github.com/AiPEX-Lab/rppg_biases','https://doi.org/10.1038/s41746-021-00462-z'],
       'code_sha256':sha(Path(__file__))}
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print('TIMING GATE: UNRESOLVED',flush=True)
if __name__=='__main__':main()
