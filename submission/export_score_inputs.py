"""Export canonical scoring CSVs without deriving reference windows from predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def export(ppg_csv,window_manifest,prediction_csv,out_dir,subject_id,recording_id,split,condition):
    if split.upper()=='D3':raise ValueError('D3 export locked during development')
    out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    raw=np.genfromtxt(ppg_csv,delimiter=',',names=True,encoding='utf-8')
    t=np.atleast_1d(raw['timestamps_s']);v=np.atleast_1d(raw['ppg'])
    if not np.isfinite(t).all() or np.any(np.diff(t)<0):raise ValueError('Invalid source clock')
    manifest=json.loads(Path(window_manifest).read_text(encoding='utf-8'))
    windows=sorted(manifest['valid_windows']+manifest['rejected_windows'],key=lambda w:w['start_s'])
    reference=[]
    for w in windows:
        start=float(w['start_s']);end=start+10
        left=max(0,int(np.searchsorted(t,start,side='right'))-1)
        right=min(len(t),int(np.searchsorted(t,end,side='left'))+1)
        # Include all samples at duplicate endpoint timestamps, including invalid ones.
        while left>0 and t[left-1]==t[left]:left-=1
        while right<len(t) and t[right]==t[right-1]:right+=1
        reference.append(dict(subject_id=subject_id,recording_id=recording_id,window_start_s=start,
          video_support_valid=not any(str(r).startswith('video_') for r in w['reasons']),
          ppg_timestamps_json=json.dumps(t[left:right].tolist(),allow_nan=False),
          ppg_values_json=json.dumps([float(x) if np.isfinite(x) else None for x in v[left:right]],allow_nan=False),
          source_reasons='|'.join(w['reasons']),split=split,condition=condition))
    with open(prediction_csv,newline='',encoding='utf-8-sig') as f:
        predictions=[dict(subject_id=subject_id,recording_id=recording_id,window_start_s=r['window_start_s'],
                          hr_bpm=r['hr_bpm'],abstain=r['abstain']) for r in csv.DictReader(f)]
    out.mkdir(parents=True)
    for name,rows in [('reference_signal.csv',reference),('processed_output.csv',predictions)]:
        if not rows:raise ValueError('Cannot export empty input')
        with open(out/name,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [ppg_csv,window_manifest,prediction_csv]}
    (out/'export_provenance.json').write_text(json.dumps({'source_hashes':hashes,
       'expected_windows_source':'T1 original source manifest, independent of prediction availability',
       'reference_hr_from_prediction_csv_used':False,'offset_changed':False,'split':split,'condition':condition},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['ppg-csv','window-manifest','prediction-csv','out-dir','subject-id','recording-id','split','condition']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args();export(a.ppg_csv,a.window_manifest,a.prediction_csv,a.out_dir,a.subject_id,a.recording_id,a.split,a.condition)
