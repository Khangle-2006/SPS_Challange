"""T1-native clocks and full-frame codec augmentation; no training or test access."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time,traceback,fcntl,shutil
from pathlib import Path
import numpy as np
import cv2
from data_adapters import MCD,UBFC2,Recording,VideoFrames
from window_support import recording_windows
from prepare_recording import prepare
from simulate import simulate,sha256,write_json
from t8_t1_loader import supported,load_clip

def main(a):
    cv2.setNumThreads(1);out=a.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    code=Path(__file__).parent
    files=['prepare_t8_cache.py','t8_t1_loader.py','data_adapters.py','window_support.py','prepare_recording.py','video_roi_frontend.py','simulate.py','protocol.json','PROTOCOL.txt','folds_frozen.json','mcd_cohort.json']
    fingerprint={f:sha256(code/f) for f in files}
    if (out/'fingerprint.json').exists():assert json.loads((out/'fingerprint.json').read_text())==fingerprint
    else:write_json(out/'fingerprint.json',fingerprint)
    folds=json.loads((code/'folds_frozen.json').read_text());mcd=json.loads((code/'mcd_cohort.json').read_text())
    if a.dataset=='MCD':
        backend=MCD(a.root,a.audit);ids=[Path(r['video']).stem for r in mcd]
        assert len(ids)==119 and len({r['patient_id'] for r in mcd})==60
        assert all(backend.rows[r]['camera']=='FullHDwebcam' and backend.exclusion(backend.rows[r]) is None for r in ids)
    else:
        backend=UBFC2(a.root);ids=sorted(set(folds['folds'][0]['ubfc_train_subjects']+folds['folds'][0]['ubfc_val_subjects']),key=lambda x:int(x[7:]))
        assert len(ids)==35 and set(ids).isdisjoint(folds['original_ubfc_test_subjects'])
    variants=['native']+[f'f{f}_b{b}' for f in [10,15,30] for b in [300,500,800]]
    inventory={'dataset':a.dataset,'recordings':ids,'variants':variants,'expected_items':len(ids)*len(variants),'no_test_access':True}
    write_json(out/'cohort.json',inventory)
    status={'state':'running','pid':os.getpid(),'completed':0,'total':inventory['expected_items'],'errors':[],'budget_bytes':a.budget_bytes,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(out/'status.json',status)
    selected=ids[:1] if a.smoke else ids
    conditions=['native','f10_b500'] if a.smoke else variants
    for variant in conditions:
        for rid in selected:
            dest=out/rid/variant;receipt=dest/'complete.json'
            status.update(recording=rid,condition=variant);write_json(out/'status.json',status)
            try:
                if receipt.exists():
                    old=json.loads(receipt.read_text());assert old['fingerprint']==fingerprint
                    status['completed']+=1;continue
                used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
                if used>a.budget_bytes-150000000 or shutil.disk_usage(out).free<5e9:raise RuntimeError('Cache/disk budget guard')
                record=backend.load(rid);dest.mkdir(parents=True,exist_ok=True)
                canonical=recording_windows(record)
                cp=out/rid/'canonical_windows.json'
                if cp.exists():assert json.loads(cp.read_text())==canonical
                else:write_json(cp,canonical)
                source_hash=record.metadata.get('source_video_sha256') or sha256(record.frames.path)
                if variant=='native':adapted=record;encoding=None
                else:
                    fps,bitrate=[int(x[1:]) for x in variant.split('_')]
                    directory=dest/'encoded'
                    if (directory/'manifest.json').exists():
                        encoding=json.loads((directory/'manifest.json').read_text());video=Path(encoding['video'])
                    else:
                        video,encoding=simulate(record.frames,record.timestamps,directory,'c5',fps=fps,bitrate_kbps=bitrate,budget_root=out,budget_bytes=a.budget_bytes)
                    if not encoding['bitrate_pass']:raise RuntimeError('Measured bitrate exceeds requested cap; preserve receipt')
                    mapping=np.load(directory/'frame_mapping.npz');tt=mapping['timestamps_s'];ix=mapping['source_frame_indices']
                    vv=np.asarray(record.metadata.get('frame_valid',np.ones(len(record.timestamps),bool)))[ix]
                    adapted=Recording(VideoFrames(video,len(tt)),tt,record.labels,record.subject_id,record.recording_id,{**record.metadata,'frame_valid':vv.tolist(),'T8_variant':variant,'original_source_sha256':source_hash})
                crop=dest/'crop'
                if crop.exists():raise RuntimeError('Partial crop exists without receipt; preserve and inspect')
                prepared=prepare(adapted,crop)
                meta=np.load(crop/'meta.npz');windows=[]
                for w in canonical['valid_windows']:
                    good=supported(meta['t'],meta['valid'],w['start_s'],w['end_s']) and supported(meta['ppg_t'],meta['ppg_valid'],w['start_s'],w['end_s'])
                    windows.append({**w,'usable':bool(good),'subject_id':record.subject_id,'recording_id':rid,'dataset':a.dataset})
                available=[w for w in windows if w['usable']]
                smoke=None
                if available:
                    x,y,ref=load_clip(crop,available[0]['start_s']);assert x.shape==(301,3,36,36) and y.shape==(300,) and ref.shape==(300,)
                    smoke={'start_s':available[0]['start_s'],'image_shape':list(x.shape),'target_shape':list(y.shape),'finite':bool(np.isfinite(x).all() and np.isfinite(y).all())}
                receipt_value={'fingerprint':fingerprint,'dataset':a.dataset,'subject_id':record.subject_id,'recording_id':rid,'variant':variant,'source_sha256':source_hash,'prepared':prepared,'canonical_valid_windows':windows,'usable_windows':len(available),'loader_smoke':smoke,'encoding':encoding,'cache_files':{f:sha256(crop/f) for f in ['rgb.npy','meta.npz','manifest.json','windows.json']}}
                write_json(receipt,receipt_value);status['completed']+=1
                print('DONE',a.dataset,rid,variant,'usable',len(available),flush=True)
            except Exception:
                error={'recording':rid,'condition':variant,'traceback':traceback.format_exc()};status['errors'].append(error);status['state']='blocked_error';write_json(out/'status.json',status);dest.mkdir(parents=True,exist_ok=True);write_json(dest/'error.json',error);raise
            finally:write_json(out/'status.json',status)
    status['state']='smoke_complete' if a.smoke else 'complete';status['bytes']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());write_json(out/'status.json',status)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['MCD','UBFC2'],required=True)
    for k in ['root','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--audit',type=Path);p.add_argument('--budget-bytes',type=int,required=True);p.add_argument('--smoke',action='store_true')
    main(p.parse_args())
