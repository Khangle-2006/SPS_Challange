"""Finish the registered GOP/background audit once inputs exist; no training or restart.

This wrapper accounts for every requested recording, including explicit exclusions.
It does not alter codec_association's definitions, thresholds, cached results or cohorts.
"""
from common import *
import fcntl, time, traceback

def main():
    lock=(OUT/'association_watch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prereg()
    assert not (OUT/'ASSOCIATION_FINAL.json').exists(), 'Final audit must not rerun'
    save(OUT/'association_watch_PREFLIGHT.json',guard())
    expected=sha(OUT/'codec_association.py')
    order=read(OUT/'TRACE_ORDER.json')
    selected=set(order['background_top20']+order['background_worst20'])
    cohort=set(read(OLD/'diagnosis205_20261108/COHORT.json')['recordings'])
    assert selected<=cohort
    while True:
        for error in ['codec_audit_ERROR.json','trace_ERROR.json']:
            assert not (OUT/error).exists(), 'Upstream stopped: '+error
        done=[r for r in selected if any((OUT/'traces'/r/'native'/n).exists() for n in ['COMPLETE.json','EXCLUDED.json'])]
        gop_done=(OUT/'codec_audit_COMPLETE.json').exists()
        save(OUT/'association_watch_STATUS.json',dict(utc=now(),pid=os.getpid(),state='waiting',GOP_complete=gop_done,background_inputs_ready=len(done),background_requested=len(selected)))
        if gop_done and len(done)==len(selected):break
        time.sleep(30)
    assert sha(OUT/'codec_association.py')==expected, 'Definition changed while waiting'
    guard(False)
    probe=subprocess.run(['pgrep','-f','^'+sys.executable+' -u codec_association.py'],capture_output=True,text=True)
    assert not probe.stdout.strip(), 'Snapshot is active; do not duplicate'
    with (OUT/'association_final.log').open('wb') as log:
        subprocess.run([sys.executable,'-u','codec_association.py'],cwd=OUT,stdout=log,stderr=subprocess.STDOUT,check=True)
    associations=rows(OUT/'CODEC_ASSOCIATION.csv')
    backgrounds=rows(OUT/'BACKGROUND_ASSOCIATION.csv') if (OUT/'BACKGROUND_ASSOCIATION.csv').exists() else []
    exclusions=rows(OUT/'ASSOCIATION_EXCLUSIONS.csv') if (OUT/'ASSOCIATION_EXCLUSIONS.csv').exists() else []
    valid={r['recording'] for r in associations}
    excluded={r['recording'] for r in exclusions if not r['reason'].startswith('background:')}
    assert valid|excluded==cohort, 'Native cohort not fully accounted for'
    background_valid={r['recording'] for r in backgrounds}
    background_excluded={r['recording'] for r in exclusions}&selected
    trace_excluded={r for r in selected if (OUT/'traces'/r/'native/EXCLUDED.json').exists()}
    assert background_valid|background_excluded|trace_excluded==selected, 'Background cohort not fully accounted for'
    receipt=dict(utc=now(),requested_native=len(cohort),valid_native=len(valid),excluded_native=len(excluded),requested_background=len(selected),valid_background=len(background_valid),excluded_background=len(selected-background_valid),cohort_accounting_complete=True,code_sha256=expected,holdout_evaluations=0,causal_claim=False,files={n:sha(OUT/n) for n in ['CODEC_ASSOCIATION.csv','BACKGROUND_ASSOCIATION.csv','ASSOCIATION_EXCLUSIONS.csv','ASSOCIATION_STATUS.json'] if (OUT/n).exists()})
    save(OUT/'ASSOCIATION_FINAL.json',receipt)
    save(OUT/'association_watch_STATUS.json',dict(utc=now(),pid=os.getpid(),state='complete',**{k:v for k,v in receipt.items() if k!='utc'}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'association_watch_ERROR.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
