"""One padding pilot on fixed205 and encoded-keyframe audit; CPU only, no training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,fcntl,traceback,subprocess
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;B=O.parent;R=B.parent
sys.path.insert(0,str(O));sys.path.insert(0,str(B));sys.path.insert(0,str(R/'padding_pilot_20261107'))
from audit import guard
from infer_dev import read,save,sha,image_clip,decode_model,crop_rgb,windows_base
from replicate_tsm import use_replicate_padding
from EfficientPhys_fixed import EfficientPhys
from datetime import datetime,timezone
def state(s,**kw):save(O/'FOLLOWUP_STATUS.json',dict(state=s,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw))
def main():
    lock=(O/'followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/'FOLLOWUP_COMPLETE.json').exists()
    save(O/'FOLLOWUP_PREFLIGHT.json',guard())
    man=read(B/'dev_manifest.json');split=read(B/'btc_split.json');cohort=read(O/'COHORT.json');videos={r['recording']:r for r in man['videos']}
    assert sha(B/'btc_split.json')==cohort['split_sha256']==man['split_sha256']
    assert all(split['people'][videos[r]['person']]['partition']=='dev' for r in cohort['recordings'])
    pilot=R/'v4_padding_pilot/fold0/seed2027/joint';st=read(pilot/'status.json')
    assert st['state']=='complete' and st['completed_epochs']==30
    expected='e0315095fda595be37e4dc3ce3b693016111cfe7da81ef924a839c5ddf32ab0c'
    assert sha(pilot/'final.pt')==st['checkpoint_sha256']==expected
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ck=torch.load(pilot/'final.pt',map_location='cpu',weights_only=False)
    assert ck['config']['padding']=='replicate_edges_training_and_inference'
    model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));model.load_state_dict(ck['model']);model.eval()
    out=O/'padding205';out.mkdir(exist_ok=True)
    for i,rec in enumerate(cohort['recordings']):
        v=videos[rec];root=B/'dev'/v['drive_id'];results=[]
        for cond in ['native','c5']:
            dest=out/(rec+'_'+cond+'.json')
            if dest.exists():
                prior=read(dest);assert prior['checkpoint_sha256']==expected and prior['split_sha256']==man['split_sha256'];continue
            # Reuse terminal old audit exactly, no repeated first10 inference.
            old=B/'diagnostics_20261108/padding_pilot';oldsummary=read(old/'COMPLETE.json');assert oldsummary['checkpoint_sha256']==expected
            import csv
            prior=[r for r in csv.DictReader((old/'WINDOWS.csv').open()) if r['recording']==rec and r['condition']==cond]
            if prior:
                rr=[]
                for r in prior:
                    rr.append(dict(recording=rec,person=v['person'],condition=cond,start=float(r['start']),scheme='10s',prediction=float(r['prediction']) if r['prediction'] else None,reason=r['reason'] or None))
                save(dest,dict(rows=rr,checkpoint_sha256=expected,split_sha256=man['split_sha256'],reused_source_sha256=sha(old/'WINDOWS.csv')));continue
            d=root/cond;c=read(d/'COMPLETE.json');assert sha(d/'crop.mkv')==c['crop_sha256'] and sha(d/'meta.npz')==c['meta_sha256']
            windows,t,valid,_=windows_base(v,d,cond);rgb=crop_rgb(d,len(t));rr=[]
            for w in windows:
                if w['scheme']!='10s':continue
                p=None;reason=None
                try:
                    x=image_clip(rgb,t,valid,w['start'],10)
                    with torch.inference_mode():y=model(torch.from_numpy(x[None]))[0].numpy()
                    assert np.isfinite(y).all();p=decode_model(y)
                except ValueError as e:reason=str(e)
                rr.append(dict(recording=rec,person=v['person'],condition=cond,start=w['start'],scheme='10s',prediction=p,reason=reason))
            save(dest,dict(rows=rr,checkpoint_sha256=expected,split_sha256=man['split_sha256'],cache_receipt_sha256=sha(d/'COMPLETE.json')))
        state('padding_inference',completed=i+1,total=205,recording=rec)
    # Actual encoded keyframe cadence, distinct from the configured GOP parameter.
    gops=[]
    for rec in cohort['recordings']:
        v=videos[rec];d=B/'dev'/v['drive_id']/'c5_encoded';p=d/'video.mp4'
        z=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_frames','-show_entries','frame=key_frame,pict_type,best_effort_timestamp_time','-of','json',str(p)]))
        frames=z['frames'];times=[float(f['best_effort_timestamp_time']) for f in frames if f['key_frame']==1]
        delta=np.diff(times)
        gops.append(dict(recording=rec,frames=len(frames),keyframes=len(times),key_times_s=times,key_interval_median=float(np.median(delta)) if len(delta) else None,key_interval_min=float(delta.min()) if len(delta) else None,key_interval_max=float(delta.max()) if len(delta) else None,encoding_manifest=read(d/'manifest.json'),encoded_sha256=sha(p)))
    save(O/'GOP.json',gops);save(O/'PADDING_GOP_COMPLETE.json',dict(videos=205,checkpoint_sha256=expected,holdout_evaluations=0,training=False))
    from update_models import update
    import time
    while not update():
        state('padding_GOP_complete_waiting_full15_V4',videos=205);time.sleep(300)
    save(O/'FOLLOWUP_COMPLETE.json',dict(videos=205,checkpoint_sha256=expected,holdout_evaluations=0,training=False));state('complete',videos=205)
if __name__=='__main__':
    try:main()
    except Exception as e:state('error',error=repr(e),traceback=traceback.format_exc());raise
