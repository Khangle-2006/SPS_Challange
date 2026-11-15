"""Cached 10-second public features. All ablations derive from the same blind spectra."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,gzip,time,subprocess,traceback,copy,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from clock_contract import align,window
from pipeline_frozen import trace_candidates
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(a,allow_nan=False));t.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
def derive(record,ablation):
    r=copy.deepcopy(record)
    for c in r['candidates']:
        assert c['method']!='Viterbi', '10-second primary comparison only'
        for p in c['peaks']:
            p['weight']=p['power']*(.25 if p['background_flag'] and ablation!='no_background' else 1)*(.25 if p['codec_flag'] and ablation!='no_codec' else 1)
        if ablation=='K1':c['peaks']=c['peaks'][:1]
    return r
def gop(path,clock,tail_drop):
    cmd=['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_frames','-show_entries','frame=pict_type','-of','json',str(path)]
    ff=json.loads(subprocess.check_output(cmd))['frames'];assert len(ff)==len(clock)+tail_drop
    types=np.array([r.get('pict_type','?') for r in ff[:len(clock)]]);t=np.asarray(clock)
    dt=np.diff(t);fs=1/np.median(dt[dt>0]);I=np.where(types=='I')[0];nonB=np.where(types!='B')[0]
    key=float(np.median(np.diff(I))) if len(I)>1 else None
    mini=float(np.median(np.diff(nonB))) if (types=='B').any() and len(nonB)>=4 else None
    frequencies=[]
    for kind,p in [('I',key),('miniGOP',mini)]:
        if p is None:continue
        for k in range(1,7):
            hz=abs((k*fs/p+fs/2)%fs-fs/2)
            if .7<=hz<=3:frequencies.append(dict(kind=kind,harmonic=k,hz=float(hz)))
    return dict(fs_actual_clock=float(fs),keyint_frames=key,miniGOP_frames=mini,artifact_frequencies=frequencies,clock_basis='frozen original_t, no container FPS substitution')
def main(server):
    import fcntl
    # Reuse extraction guard without launching or touching its lock.
    import worker_budget_v3 as worker
    lock=(P/('features_'+server+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (P/('FEATURES_COMPLETE_'+server+'.json')).exists()
    for name,h in read(P/'FEATURE_CODE_HASHES_BUDGET_V3.json').items():assert sha(P/name)==h
    save(P/('FEATURE_PREFLIGHT_'+server+'.json'),worker.guard(server,True))
    records=[r for r in read(P/'MANIFEST.json')['records'] if r['server']==server]
    frozen=read(P/'REFERENCE_COMPACT.json');refs={(r['dataset'],r['recording_id'],r['variant']):r for r in frozen['records']}
    for r in records:
        for variant in ['native','f10_b500']:
            d=P/'traces'/r['dataset']/r['recording']/variant;cp=d/'PUBLIC_COMPLETE.json';out=P/'features'/r['dataset']/(r['recording']+'_'+variant+'.json.gz')
            if out.exists():continue
            while not cp.exists():
                if (P/('ERROR_'+server+'.json')).exists():raise RuntimeError('Extraction failed; no automatic restart')
                if (P/('COMPLETE_'+server+'.json')).exists():raise RuntimeError('Extraction complete but receipt missing')
                save(P/('FEATURE_STATUS_'+server+'.json'),dict(state='waiting_trace',recording=r['recording'],variant=variant,pid=os.getpid(),utc=now()));time.sleep(15)
            worker.guard(server)
            receipt=read(cp);assert sha(d/'trace.npz')==receipt['trace_sha256'] and sha(d/'clock.npz')==receipt['clock_sha256']
            with np.load(d/'trace.npz') as z:rgb=z['rgb']
            with np.load(d/'clock.npz') as z:original=z['original_t'];canonical=z['canonical_t']
            t,rgb=align(original,canonical,rgb,receipt['allowed_tail_drop'])
            cache_receipt=read(Path(r['cache'])/variant/'complete.json')
            source=Path(r['source'] if variant=='native' else cache_receipt['encoding']['video'])
            expected=cache_receipt['source_sha256'] if variant=='native' else cache_receipt['encoding']['video_sha256_at_reuse_or_creation']
            assert sha(source)==expected==receipt['source_sha256'], 'source receipt mismatch'
            gp=d/'GOP.json'
            if gp.exists():g=read(gp)
            else:g=gop(source,original,receipt['allowed_tail_drop']);save(gp,g)
            results=[];ref=refs[r['dataset'],r['recording'],variant]
            for b in ref['windows']:
                if not b['reference_eligible']:continue
                assert b['fold']==r['fold'] and b['subject_id']==r['person']
                try:
                    tw,xw=window(t,rgb,b['window_start_s']);a=trace_candidates(tw,xw,[x['hz'] for x in g['artifact_frequencies']])
                except (ValueError,AssertionError) as e:a=dict(candidates=[],background=[],failures=[dict(reason=str(e),action='prior_only_no_extrapolation')])
                results.append(dict(person=r['person'],recording=r['recording']+'@'+str(b['window_start_s']),video=r['recording'],dataset=r['dataset'],condition=variant,fold=r['fold'],truth=b['reference_hr_bpm'],start=b['window_start_s'],**a))
            obj=dict(protocol_sha256=sha(P/'PROTOCOL.md'),trace_sha256=receipt['trace_sha256'],clock_sha256=receipt['clock_sha256'],reference_sha256=sha(P/'REFERENCE_COMPACT.json'),rows=results)
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.tmp')
            with gzip.open(tmp,'wt',encoding='utf8') as f:json.dump(obj,f,allow_nan=False)
            tmp.replace(out)
            save(P/('FEATURE_STATUS_'+server+'.json'),dict(state='features',recording=r['recording'],variant=variant,completed=len(list((P/'features').rglob('*.gz'))),total=2*len(records),pid=os.getpid(),utc=now()))
    save(P/('FEATURES_COMPLETE_'+server+'.json'),dict(utc=now(),records=len(records),conditions=2,holdout_evaluations=0))
if __name__=='__main__':
    server=sys.argv[1]
    try:main(server)
    except Exception as e:save(P/('FEATURE_ERROR_'+server+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
