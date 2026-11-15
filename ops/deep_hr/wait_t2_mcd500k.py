"""Read-only E0 completion watcher, then one bounded T2 trace replay.

Does not start, stop or alter E0. Advisory flock forbids duplicate watchers.
"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path

def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--root',required=True);ap.add_argument('--audit',required=True);ap.add_argument('--e0',required=True);a=ap.parse_args()
    out=Path(a.out);lock=(out/'wait_mcd500k.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statefile=out/'wait_mcd500k_status.json';start=time.time()
    while not (out/'mcd_h264_500k_results.json').exists():
        state=json.loads((Path(a.e0)/'status.json').read_text())
        if state.get('state')=='complete' and state.get('processed')==119:
            save(statefile,dict(state='running_replay',pid=os.getpid(),started_unix=start,updated_unix=time.time()))
            args=[sys.executable,str(Path(__file__).with_name('evaluate_t2_pos.py')),'--out',a.out,'--mcd-root',a.root,'--audit',a.audit,'--e0',a.e0,'--condition','h264_500k']
            with (out/'mcd_h264_500k.log').open('a') as log:code=subprocess.call(args,stdout=log,stderr=subprocess.STDOUT)
            if code:
                save(statefile,dict(state='error',returncode=code,pid=os.getpid(),updated_unix=time.time()));return code
            break
        save(statefile,dict(state='waiting_full_E0',pid=os.getpid(),e0_state=state.get('state'),e0_processed=state.get('processed'),updated_unix=time.time()))
        time.sleep(30)
    save(statefile,dict(state='complete',pid=os.getpid(),updated_unix=time.time()))
    return 0

if __name__=='__main__':raise SystemExit(main())
