"""Create a PRIVATE Round1_Baseline only after all three S1 raw-video parity checks pass."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from submission.mat_writer import write_mat,verify_mat,file_sha256
from submission.metrics import compute_metrics


def frames(path):
    cap=cv2.VideoCapture(str(path))
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            yield cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    finally:cap.release()


def csvwrite(path,header,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--git-commit',required=True);a=p.parse_args();o=a.work.resolve()
    gate=json.loads((o/'A1_COMPLETE.json').read_text())
    if gate.get('gate') is not True or len(gate['rows'])!=3:raise ValueError('A1 did not pass; no submission bundle permitted')
    j=json.loads((o/'DEV_SELECTION.json').read_text())[0]
    if j['dataset']!='UBFC2':raise ValueError('Baseline must be the preregistered UBFC2 DEV choice')
    src=Path(j['source'])
    if file_sha256(src)!=j['source_sha256']:raise ValueError('Raw source SHA changed')
    out=o/'Round1_Baseline';out.mkdir(exist_ok=False)
    run=o/'a1'/j['key'];pred=json.loads((run/'result.json').read_text());rows=pred['windows'];cap=cv2.VideoCapture(str(src))
    w,h,n=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    mem=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0])<15*1024**2:raise RuntimeError('Available RAM <15 GiB')
    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264rgb','-crf','0','-preset','medium','-threads','2',str(out/'target_video.mp4')]
    digest=hashlib.sha256();count=0
    with (o/'baseline_transcode.log').open('w') as log:
        enc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            for fr in frames(src):
                digest.update(fr.tobytes());enc.stdin.write(fr.tobytes());count+=1
            enc.stdin.close()
            if enc.wait():raise RuntimeError('Lossless transcode failed')
        finally:
            if enc.poll() is None:enc.terminate();enc.wait()
    if count!=n:raise ValueError('Source decoded frame count changed')
    got=hashlib.sha256();n2=0
    for fr in frames(out/'target_video.mp4'):got.update(fr.tobytes());n2+=1
    if got.hexdigest()!=digest.hexdigest() or n2!=n:raise ValueError('MP4 is not decoded-bit-exact; STOP A')
    # Original UBFC2 contact PPG, acquisition clock and per-window reference.
    gt=np.loadtxt(src.parent/'ground_truth.txt');ppg=gt[0];ppgt=gt[2]-gt[2,0]
    csvwrite(out/'reference_signal.csv',['timestamps_s','ppg'],zip(ppgt,ppg))
    csvwrite(out/'reference_hr_windows.csv',['window_start_s','hr_ref_bpm'],[(x['start'],x['truth']) for x in j['windows']])
    csvwrite(out/'processed_output.csv',['window_start_s','window_end_s','hr_bpm','abstain'],[(x['window_start_s'],x['window_end_s'],'' if x['abstain'] else x['hr_bpm'],int(x['abstain'])) for x in rows])
    with np.load(o/(j['key']+'_clock.npz')) as z:clock={k:z[k] for k in z.files}
    clock['source_sha256']=np.asarray(file_sha256(out/'target_video.mp4'));np.savez_compressed(out/'acquisition_clock.npz',**clock)
    csvwrite(out/'video_timestamps.csv',['frame','timestamps_s'],enumerate(clock['t']))
    with np.load(run/'trace.npz') as z:trace={k:z[k] for k in ['boxes','oval_points','valid','t']}
    waves=np.load(run/'pulse_waves.npz');rr=[];tt=[];wi=[]
    for i,x in enumerate(rows):
        key='window_'+str(i)
        if key not in waves:continue
        rr.extend(waves[key][1]);tt.extend(x['window_start_s']+np.arange(300)/30);wi.extend([i]*300)
    reference={round(x['start'],8):x['truth'] for x in j['windows']};y=[reference[round(x['window_start_s'],8)] for x in rows]
    hp=np.array([np.nan if x['abstain'] else x['hr_bpm'] for x in rows]);ab=np.array([x['abstain'] for x in rows]);met=compute_metrics(hp,y,[j['person']]*len(rows),ab)
    config=dict(dataset='UBFC2',fold=j['fold'],pipeline='P4m R1',git_commit=a.git_commit,reader_sha256=pred['reader_sha256'],checkpoint_sha256=pred['checkpoint_sha256'],source_sha256=j['source_sha256'],target_sha256=clock['source_sha256'].item())
    fields=dict(roi_data=dict(oval_points=trace['oval_points'],boxes=trace['boxes'],valid=trace['valid'],timestamps_s=trace['t'],roi_type='oval36_actual_frontend',regions_note='This pipeline uses the whole oval, not separate forehead/cheek masks; no invented regions.'),
        reference_signal=dict(ppg=ppg,timestamps_s=ppgt,valid=np.isfinite(ppg),source='UBFC2 ground_truth.txt row 1 contact PPG; timestamps row 3',fs_hz=(len(ppgt)-1)/(ppgt[-1]-ppgt[0]),hr_ref_bpm_per_window=np.array(y)),
        processed_signal=dict(rppg=np.array(rr),timestamps_s=np.array(tt),window_index_per_sample=np.array(wi),fs_hz=30,hr_bpm_per_window=hp,window_start_s=np.array([x['window_start_s'] for x in rows]),abstain=ab,waveform_source='CNN cumsum per window; P4m fuses spectral HR evidence and does not produce a single fused waveform'),
        metrics=met,params=dict(**config,resolution=np.array([w,h]),fps_nominal=fps,fps_effective=(len(clock['t'])-1)/(clock['t'][-1]-clock['t'][0]),video_timestamps_s=clock['t'],lux='NA',distance_m='NA',fitzpatrick='NA',age='NA',constraints_applied='None',video_storage='decoded RGB, lossless libx264rgb MP4; acquisition clock supplied separately'))
    info=write_mat(out/'Round1_Baseline_None.mat',fields,frames=frames(out/'target_video.mp4'),shape=(h,w,3,n))
    check=verify_mat(out/'Round1_Baseline_None.mat',(h,w,3,n),digest.hexdigest())
    (out/'baseline_config.json').write_text(json.dumps(config,indent=2)+'\n')
    shutil.copyfile(ROOT/'submission/process_public_round1.py',out/'process_round1.py');shutil.copyfile(ROOT/'submission/process_public_round1.m',out/'process_round1.m')
    shutil.copyfile(run/'pulse_waves.npz',out/'pulse_waves.npz');shutil.copyfile(run/'expert_waves.npz',out/'expert_waves.npz')
    readme=f'''# Round 1 baseline format demonstration

Repository: https://github.com/Khangle-2006/SPS_Challange/tree/{a.git_commit}
Pipeline: P4m R1, matching held-out fold {j['fold']}; 10 s windows, 10 s stride. No fitting.

The team's main results (BTC reader_v4 and seven constraints) are in the technical report. This single baseline recording demonstrates the submission format; it does not establish general performance.

| File | Content |
|---|---|
| target_video.mp4 | Raw decoded RGB frames, lossless repackaging; original acquisition clock is separate |
| Round1_Baseline_None.mat | Six variables: video, actual oval ROI, reference, output, metrics, metadata |
| reference_signal.csv | Original contact PPG waveform and acquisition timestamps; this file is NOT device HR |
| reference_hr_windows.csv | PPG-derived HR reference at the declared 10-second windows |
| processed_output.csv | P4m window HR and abstention, including rejected windows |
| acquisition_clock.npz / video_timestamps.csv | Source-bound clock; excludes labels and cached ROIs |
| pulse_waves.npz | Per accepted window: CNN difference, CNN cumsum, POS, CHROM, three experts (300 samples each) |
| expert_waves.npz | Full expert waveform and time arrays, after reset/self-warm |
| baseline_config.json | Code/checkpoint/reader/source hashes and matching fold |
| process_round1.py / process_round1.m | Python CLI wrapper and MATLAB/Octave system wrapper |

Clone the repository beside this folder as `repo`. Run `python3.12 tools/bootstrap_video.py` and `python3.12 tools/setup_env.py torch-cpu` in that checkout. Then:

```sh
python3.12 process_round1.py --repo ../repo --output ../baseline_rerun
```

In MATLAB/Octave: `process_round1('../repo','../baseline_rerun','python3.12')`.
Output must be new. Setup is required; no network/model installation occurs inside the wrapper.
The MP4 clock is nominal CFR; use the supplied acquisition clock for the PPG-aligned benchmark. Unknown lux, distance, age and skin type are NA. The ROI is the actual whole oval; separate anatomical region masks are not claimed. Single-person repeated-measure Bland-Altman population limits are NA. Real MATLAB validation is not claimed; consult the external validation receipt for mat73/Octave results.
This private package contains dataset media/signals. Do not upload it to the public repository.
'''
    (out/'README_SUBMISSION.md').write_text(readme)
    receipt=dict(mat=info,mat73=check,source_decoded_sha256=digest.hexdigest(),target_decoded_sha256=got.hexdigest(),shape=[h,w,3,n],source_frame_count=n,no_holdout=True,git_commit=a.git_commit)
    (o/'BASELINE_MAT_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    # ZIP is sealed only after wrapper/Octave checks, by the separate finalizer.


if __name__=='__main__':main()
