"""CPU-only zero-shot, per-checkpoint outputs; no BTC weight updates or holdout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import ast,fcntl,hashlib,json,math,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy import signal,sparse
from scipy.sparse.linalg import spsolve
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/samsung_t5/khangle/facevitals')
V3=Path('/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102')
sys.path.insert(0,str(V3))
from t8_t1_loader import supported
from score_round1 import reference_hr
from t8_evaluate import decode_difference

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def image_clip(rgb,t,valid,start,seconds):
    q=start+np.arange(seconds*30+1)/30
    if not supported(t,valid,q[0],q[-1]):raise ValueError('image_support_gap_or_tracking_failure')
    j=np.searchsorted(t,q,side='right').clip(1,len(t)-1)
    w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
    x=np.asarray(rgb[j-1],np.float32)*(1-w[:,None,None,None])+np.asarray(rgb[j],np.float32)*w[:,None,None,None]
    if not np.isfinite(x).all() or x.std()<1e-6:raise ValueError('invalid_or_constant_image_clip')
    x=(x-x.mean())/x.std()
    return x.transpose(0,3,1,2).copy()
def decode_wave(w):
    w=np.asarray(w,float)
    if not np.isfinite(w).all() or w.std()<1e-10:raise ValueError('flat_or_nonfinite_waveform')
    if len(w)==300:
        return reference_hr(dict(window_start_s=0.,ppg_timestamps_json=json.dumps((np.arange(301)/30).tolist()),ppg_values_json=json.dumps(np.r_[w,w[-1]].tolist())))
    assert len(w)==900
    # Same frozen operator, generalized only to the predeclared 30s sensitivity length.
    n=len(w);d=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    trend=spsolve(sparse.eye(n,format='csc')+100**2*(d.T@d),w)
    y=signal.sosfiltfilt(signal.butter(1,[.7,3.],btype='bandpass',fs=30,output='sos'),w-trend)
    f,p=signal.periodogram(y,fs=30,window='hann',nfft=4096,detrend=False);sel=(f>=.7)&(f<=3.)
    if not np.isfinite(p).all() or p[sel].sum()<=1e-30:raise ValueError('invalid_spectrum')
    return float(60*f[sel][np.argmax(p[sel])])
def decode_model(y):
    y=np.asarray(y,float)
    if len(y)==300:return decode_difference(y)
    if len(y)!=900 or not np.isfinite(y).all() or y.std()<1e-10:raise ValueError('invalid_prediction_waveform')
    return decode_wave(np.cumsum(y))
def classical_functions():
    namespace=dict(np=np,json=json,math=math,signal=signal)
    receipts={}
    for name,file in [('pos','hr_baseline.py'),('chrom','t7_pipeline.py')]:
        p=HERE/'vendor'/file;src=p.read_text();fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),namespace);receipts[file]=sha(p)
    return namespace,receipts
def candidate_windows(span):
    return [(name,float(start),seconds) for name,seconds,stride in [('10s',10,10),('10s_stride5',10,5),('30s',30,30)]
        for start in np.arange(0,max(0,span-seconds+1e-8),stride)]
def model_inventory():
    entries=[]
    frozen=read(HERE.parent.parent/'round1_final_20261105/revision_02/frozen_data/CHECKPOINT_MANIFEST.json')
    old={(e['branch'],e['fold'],e['seed']):e for e in frozen}
    for branch,base,arm in [('ubfc_only',ROOT/'t8_oval_20261031/runs','ubfc_only'),('v3_joint',ROOT/'t8_polarity_v3_20261102','joint'),('v4_repaired',HERE.parent/'v4_repaired','joint')]:
        for f in range(5):
            for s in [2027,2028,2029]:
                d=base/f'fold{f}'/f'seed{s}'/arm
                if not (d/'status.json').exists():continue
                st=read(d/'status.json')
                if st['state']!='complete':continue
                assert st['completed_epochs']==30 and sha(d/'final.pt')==st['checkpoint_sha256']
                assert [h['epoch'] for h in read(d/'history.json')]==list(range(1,31))
                if branch!='v4_repaired':
                    prior=old[branch,f,s]
                    assert st['checkpoint_sha256']==prior['checkpoint_sha256'] and sha(d/'provenance.json')==prior['provenance_sha256']
                entries.append(dict(branch=branch,fold=f,seed=s,path=str(d/'final.pt'),sha256=st['checkpoint_sha256'],name=f'{branch}_f{f}_s{s}'))
    assert sum(e['branch']=='ubfc_only' for e in entries)==15 and sum(e['branch']=='v3_joint' for e in entries)==15
    return entries
def crop_rgb(d,n):
    b=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(d/'crop.mkv'),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    assert len(b)==n*36*36*3
    return np.frombuffer(b,np.uint8).reshape(n,36,36,3)
def windows_base(row,d,condition):
    src=read(d.parent/'SOURCE.json')
    with np.load(d/'meta.npz') as m:t=m['t'].copy();valid=m['valid'].copy();mean=m['rgb_mean'].copy()
    rows=[]
    for scheme,start,seconds in candidate_windows(src['source_span_s']):
        rows.append(dict(person=row['person'],recording=row['recording'],condition=condition,scheme=scheme,start=start,seconds=seconds,
            support_valid=bool(supported(t,valid,start,start+seconds))))
    return rows,t,valid,mean
def run_classical(row,d,condition,ns,sources):
    dest=HERE/'predictions'/row['drive_id']/condition/'classical.json'
    if dest.exists():return
    rows,t,valid,rgb=windows_base(row,d,condition);result=[]
    for w in rows:
        for method in ['POS','CHROM']:
            p=None;reason=None
            try:
                if not w['support_valid']:raise ValueError('image_support_gap_or_tracking_failure')
                q=w['start']+np.arange(w['seconds']*30)/30
                x=np.stack([np.interp(q,t,rgb[:,c]) for c in range(3)],axis=1)
                p=decode_wave(ns[method.lower()](x,30))
            except ValueError as e:reason=str(e)
            result.append(dict(w,method=method,prediction=p,reason=reason))
    save(dest,dict(rows=result,sources=sources,cache_sha256=sha(d/'COMPLETE.json'),code_sha256=sha(__file__),split_sha256=sha(HERE/'btc_split.json')))
    control=read(HERE/'BTC_CONTROL_SAMPLE.json')
    if row['person'] in control['people']:
        selected=next((w for w in rows if w['scheme']=='10s' and w['support_valid']),None)
        for method in ['POS','CHROM']:
            p=None;reason='no_valid_main_window'
            if selected is not None:
                i=int(np.searchsorted(t,selected['start']))
                try:p=decode_wave(ns[method.lower()](np.repeat(rgb[i:i+1],300,axis=0),30));reason=None
                except ValueError as e:reason=str(e)
            save(HERE/'btc_controls'/row['drive_id']/condition/(method+'.json'),dict(person=row['person'],recording=row['recording'],method=method,
                condition=condition,static_hr=p,static_reason=reason,static_pass=p is None and selected is not None,sample_sha256=sha(HERE/'BTC_CONTROL_SAMPLE.json')))
def run_model(row,d,condition,entry,torch):
    dest=HERE/'predictions'/row['drive_id']/condition/(entry['name']+'.json')
    if dest.exists():assert read(dest)['checkpoint']['sha256']==entry['sha256'];return
    from EfficientPhys_fixed import EfficientPhys
    model=EfficientPhys(frame_depth=10,img_size=36)
    ck=torch.load(entry['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval()
    rows,t,valid,mean=windows_base(row,d,condition)
    c=read(d/'COMPLETE.json');assert sha(d/'crop.mkv')==c['crop_sha256'] and sha(d/'meta.npz')==c['meta_sha256']
    rgb=crop_rgb(d,len(t));result=[];cache={}
    for w in rows:
        ident=w['start'],w['seconds']
        if ident not in cache:
            p=None;reason=None
            try:
                x=image_clip(rgb,t,valid,*ident)
                with torch.inference_mode():y=model(torch.from_numpy(x[None]))[0].numpy()
                assert np.isfinite(y).all(),'nonfinite model output is a pipeline error'
                p=decode_model(y)
            except ValueError as e:reason=str(e)
            cache[ident]=(p,reason)
        p,reason=cache[ident];result.append(dict(w,method=entry['name'],prediction=p,reason=reason))
    save(dest,dict(rows=result,checkpoint=entry,cache_sha256=sha(d/'COMPLETE.json'),code_sha256=sha(__file__),split_sha256=sha(HERE/'btc_split.json')))
    controls=read(HERE/'BTC_CONTROL_SAMPLE.json')
    if row['person'] in controls['people']:
        selected=next((w for w in rows if w['scheme']=='10s' and w['support_valid']),None)
        result=None;reason='no_valid_main_window'
        if selected is not None:
            x=image_clip(rgb,t,valid,selected['start'],10);x=np.repeat(x[:1],len(x),axis=0)
            with torch.inference_mode():wave=model(torch.from_numpy(x[None]))[0].numpy()
            try:result=decode_model(wave);reason=None
            except ValueError as e:reason=str(e)
        save(HERE/'btc_controls'/row['drive_id']/condition/(entry['name']+'.json'),dict(person=row['person'],recording=row['recording'],method=entry['name'],
            condition=condition,static_hr=result,static_reason=reason,static_pass=result is None and selected is not None,
            checkpoint_sha256=entry['sha256'],sample_sha256=sha(HERE/'BTC_CONTROL_SAMPLE.json')))
def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    lock=(HERE/'inference.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    man=read(HERE/'dev_manifest.json');split=read(HERE/'btc_split.json');assert man['split_sha256']==sha(HERE/'btc_split.json')
    for row in man['videos']:assert split['people'][row['person']]['partition']=='dev'
    ns,sources=classical_functions()
    frozen={str(p):sha(p) for p in [Path(__file__),V3/'EfficientPhys_fixed.py',V3/'t8_t1_loader.py',V3/'score_round1.py',V3/'t8_evaluate.py',HERE/'BTC_CONTROL_SAMPLE.json',HERE/'DECISIONS.md']}
    if (HERE/'INFERENCE_SOURCES.json').exists():assert read(HERE/'INFERENCE_SOURCES.json')==frozen
    else:save(HERE/'INFERENCE_SOURCES.json',frozen)
    while True:
        entries=model_inventory();save(HERE/'CHECKPOINTS.json',entries)
        ready=[]
        # Classical baselines for every currently ready clip before slower learned models.
        for row in man['videos']:
            root=HERE/'dev'/row['drive_id']
            if not (root/'COMPLETE.json').exists():continue
            info=read(root/'COMPLETE.json')
            for condition in ['native','c5']:
                if condition not in info:continue
                d=root/condition;run_classical(row,d,condition,ns,sources);ready.append((row,d,condition))
        todo=0
        for entry in entries:
            for row,d,condition in ready:
                out=HERE/'predictions'/row['drive_id']/condition/(entry['name']+'.json')
                if out.exists():continue
                todo+=1;save(HERE/'INFERENCE_STATUS.json',dict(state='running',pid=os.getpid(),recording=row['recording'],condition=condition,checkpoint=entry['name'],available_checkpoints=len(entries)))
                run_model(row,d,condition,entry,torch)
        if read(HERE/'PREPROCESS_STATUS.json').get('state')=='complete' and len(entries)==45:
            save(HERE/'INFERENCE_STATUS.json',dict(state='complete',pid=os.getpid(),checkpoints=45));return
        save(HERE/'INFERENCE_STATUS.json',dict(state='waiting_more_cache_or_checkpoints',pid=os.getpid(),available_checkpoints=len(entries),ready_video_conditions=len(ready)))
        time.sleep(30)
if __name__=='__main__':
    try:main()
    except Exception as e:save(HERE/'INFERENCE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
