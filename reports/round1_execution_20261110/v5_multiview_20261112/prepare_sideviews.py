"""Prepare only eligible new view caches with canonical sign and explicit phase shift."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2')
import sys,csv,json,traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
from runtime import *
os.environ['CUDA_VISIBLE_DEVICES']=''
T8=Path('/data_pool/khangle/facevitals/code/t8_oval_20261031');BACK=Path('/data_pool/khangle/facevitals/code/ppg_polarity_backend_20261102')
# Unimported adapter resolved explicitly to globally canonical backend.
sys.path.insert(0,str(T8));sys.path.insert(0,str(BACK))
from data_adapters import MCD
from prepare_recording import prepare
from t8_t1_loader import supported,clip_arrays
def main():
    lease=lock(P/'prep.lock');assert not (P/'PREP_STARTED.json').exists(),'Inspect previous start; no duplicate/restart'
    assert read(P/'OFFSETS_COMPLETE_V2.json')['results_sha256']==sha(P/'OFFSETS_V2.json')
    rows=[r for r in read(P/'OFFSETS_V2.json') if r['eligible']];assert any(r['camera']=='IriunWebcam' for r in rows),'No eligible Iriun; do not run empty V5'
    frames=0;dur=0
    for r in rows:
        g=read(R/'mapping_audit_20261030/dev60_gate_v3'/f"{r['recording']}.json");frames+=g['gate']['candidate_prefix_frames'];dur+=g['gate']['candidate_prefix_frames']/r['fs']
    native_bound=frames*36*36*3*2+frames*700
    # One deterministic codec/lowres per window per train fold, <=361 encoded crop frames.
    job_bound=int(dur//10+len(rows))*4;aug_bound=job_bound*(361*36*36*3+50000)
    alloc=native_bound+aug_bound+450_000_000
    plan=dict(eligible_records=len(rows),eligible_people=len({r['person'] for r in rows}),native_bound_bytes=native_bound,augmented_jobs_upper_bound=job_bound,augmented_bound_bytes=aug_bound,checkpoint_reports_bound_bytes=450_000_000,allocation_bytes=alloc,old_completed_N_unused_reservation_released=True,raw_deleted=False)
    save(P/'STORAGE_PLAN.json',plan);b=budget(fresh=True);save(P/'PREP_STARTED.json',dict(utc=now(),pid=os.getpid(),budget=b,offsets_sha256=sha(P/'OFFSETS_V2.json'),code_sha256=sha(__file__)))
    db=P/'eligible_db.csv';keys=list(rows[0]['raw_row']);
    with db.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(r['raw_row'] for r in rows)
    backend=MCD(R/'datasets/MCD-rPPG',R/'mapping_audit_20261030',db_path=db);result=[]
    for i,r in enumerate(rows):
        budget();rid=r['recording'];rec=backend.load(rid);assert rec.metadata['ppg_polarity']['raw_to_canonical_sign']==-1
        assert sha(rec.frames.path)==r['source_sha256'];ppg=rec.labels['ppg'];shift=r['offset_seconds'];newlabel=replace(ppg,timestamps=ppg.timestamps+shift)
        labels={'ppg':newlabel} # No stale frame-aligned label retained after shifting independent PPG clock.
        rec=replace(rec,labels=labels,metadata={**rec.metadata,'operational_alignment_accepted':True,'synchronization_verified':False,'synchronization_basis':'V5 preregistered POS/PPG phase alignment; not exposure synchronization','v5_offset_seconds':shift,'offset_receipt_sha256':sha(P/'offsets_v2'/f'{rid}.json')})
        dest=P/'side_native'/rid/'native';dest.mkdir(parents=True);assert not (dest/'crop').exists()
        prepared=prepare(rec,dest/'crop');m=dict(np.load(dest/'crop/meta.npz'));rgb=np.load(dest/'crop/rgb.npy',mmap_mode='r');windows=[]
        start=max(float(m['t'][0]),float(m['ppg_t'][0]));last=min(float(m['t'][-1]),float(m['ppg_t'][-1]))
        for s in np.arange(start,last-10+1e-8,10):
            if supported(m['t'],m['valid'],s,s+10) and supported(m['ppg_t'],m['ppg_valid'],s,s+10):
                x,y,_=clip_arrays(rgb,m['t'],m['valid'],m['ppg'],m['ppg_t'],m['ppg_valid'],float(s));assert x.shape==(301,3,36,36) and y.shape==(300,)
                windows.append(float(s))
        a=dict(dataset='MCD_SIDE',recording_id=rid,subject_id=r['person'],fold=r['fold'],camera=r['camera'],source=str(rec.frames.path),source_sha256=r['source_sha256'],offset_seconds=shift,offset_receipt_sha256=sha(P/'offsets_v2'/f'{rid}.json'),polarity=rec.metadata['ppg_polarity'],prepared=prepared,windows=windows,cache_files={n:sha(dest/'crop'/n) for n in ['rgb.npy','meta.npz','manifest.json','windows.json']},frontend_sha256=sha(T8/'oval_crop.py'))
        save(dest/'complete.json',a);result.append(a);save(P/'PREP_STATUS.json',dict(pid=os.getpid(),completed=i+1,total=len(rows),last=rid,usable_windows=len(windows)))
    save(P/'SIDE_INDEX.json',result);save(P/'PREP_COMPLETE.json',dict(utc=now(),records=len(result),windows=sum(len(r['windows']) for r in result),index_sha256=sha(P/'SIDE_INDEX.json'),budget=budget(),holdout_evaluations=0))
if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'PREP_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
