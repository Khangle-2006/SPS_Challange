"""Frozen T3 all-frame cache; one process, lock, resume matching fingerprints only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from pathlib import Path
import argparse
import json
import time
import socket
import hashlib
import traceback
import fcntl
from data_adapters import UBFC2,MCD
from video_landmark_frontend import process_recording,sha256


def write_json(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['UBFC2','MCD'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--audit');p.add_argument('--out',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads(Path(a.protocol).read_text());budget=protocol['budget_bytes_each_server']
    code=Path(__file__).parent
    provenance={str(f.name):sha256(f) for f in [code/'t3_landmark_runner.py',code/'video_landmark_frontend.py',
        code/'data_adapters.py',code/'assets/canonical_face_model.obj',Path(a.protocol)]}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    if (out/'provenance.json').exists():
        if json.loads((out/'provenance.json').read_text())['fingerprint']!=fingerprint:raise RuntimeError('Fingerprint differs; use new version output')
    else:write_json(out/'provenance.json',{'fingerprint':fingerprint,'sha256':provenance,'protocol':protocol})
    backend=UBFC2(a.root) if a.dataset=='UBFC2' else MCD(a.root,a.audit)
    ids=backend.recordings()
    if a.dataset=='MCD':ids=[i for i in ids if backend.rows[i]['camera']=='FullHDwebcam']
    if a.limit:ids=ids[:a.limit]
    qa_ids=set(ids[:5]);(out/'qa').mkdir(exist_ok=True)
    status={'state':'running','dataset':a.dataset,'pid':os.getpid(),'host':socket.gethostname(),
            'fingerprint':fingerprint,'total':len(ids),'completed':0,'failed':0,'started':time.time(),'errors':[]}
    results=[];write_json(out/'status.json',status)
    for rid in ids:
        status['current']=rid;write_json(out/'status.json',status)
        dest=out/rid
        if (dest/'manifest.json').exists():
            result=json.loads((dest/'manifest.json').read_text())
            results.append(result);status['completed']+=1;continue
        current_bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
        if current_bytes>budget-300000000:
            status.update(state='blocked_budget',bytes=current_bytes);write_json(out/'status.json',status);return
        print('START',rid,flush=True)
        try:
            recording=backend.load(rid)
            def progress(done,total):
                status.update(current_frame=done,current_total_frames=total,updated=time.time())
                write_json(out/'status.json',status)
            result=process_recording(recording,dest,config={'fingerprint':fingerprint,'dataset':a.dataset},
              qa_path=out/'qa'/(rid+'.png') if rid in qa_ids else None,progress=progress)
            results.append(result);status['completed']+=1
            print('DONE',rid,'frames',result['n_frames'],'detected',result['detected_fraction'],flush=True)
        except Exception as e:
            err={'recording_id':rid,'error':repr(e),'traceback':traceback.format_exc()}
            status['errors'].append(err);status['failed']+=1
            dest.mkdir(exist_ok=True);write_json(dest/'error.json',err);print('ERROR',rid,repr(e),flush=True)
        write_json(out/'results.json',results);write_json(out/'status.json',status)
    status.update(state='complete' if not status['failed'] else 'complete_with_errors',finished=time.time(),
                  bytes=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()))
    write_json(out/'results.json',results);write_json(out/'status.json',status)
    print('FINISHED',json.dumps(status),flush=True)


if __name__=='__main__':main()
