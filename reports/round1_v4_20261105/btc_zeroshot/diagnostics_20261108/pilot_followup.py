"""Single CPU follow-up after the predeclared pilot reaches a terminal checkpoint."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,time,json,fcntl,traceback,datetime
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REPORT=BASE.parent
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPORT/'padding_pilot_20261107'))
from infer_dev import image_clip,decode_model,crop_rgb,windows_base
from distribution import read,save,sha,csvout,describe,num
from replicate_tsm import use_replicate_padding
from test_static_tsm import static_result
from EfficientPhys_fixed import EfficientPhys
OUT=HERE/'padding_pilot';OUT.mkdir(exist_ok=True)
def state(s,**kw):save(OUT/'STATUS.json',dict(state=s,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def main():
    lock=(OUT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'COMPLETE.json').exists():return
    pilot=REPORT/'v4_padding_pilot/fold0/seed2027/joint'
    source_paths=[Path(__file__),HERE/'distribution.py',BASE/'infer_dev.py',REPORT/'padding_pilot_20261107/replicate_tsm.py',REPORT/'padding_pilot_20261107/test_static_tsm.py']
    sources={str(p):sha(p) for p in source_paths};save(OUT/'SOURCES.json',sources)
    while True:
        st=read(pilot/'status.json') if (pilot/'status.json').exists() else {}
        if st.get('state') in ['complete','stopped_epoch5_loss']:break
        state('waiting_terminal_pilot',pilot_status=st);time.sleep(30)
    assert {str(p):sha(p) for p in source_paths}==sources
    final=st['state']=='complete';ckpath=pilot/('final.pt' if final else 'last.pt');cksha=sha(ckpath)
    if final:assert st['completed_epochs']==30 and st['checkpoint_sha256']==cksha
    else:assert st['completed_epochs']==5 and not st['gate']['passed']
    state('cpu_static_and_BTC',checkpoint_sha256=cksha,epoch=30 if final else 5)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['epoch']==(30 if final else 5)
    assert ck['config']['padding']=='replicate_edges_training_and_inference'
    model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));model.load_state_dict(ck['model']);model.eval()
    synth=static_result(model);save(OUT/'STATIC_SYNTHETIC.json',dict(result=synth,pass_static=synth['hr'] is None,checkpoint_sha256=cksha))
    man=read(BASE/'dev_manifest.json');split=read(BASE/'btc_split.json');assert sha(BASE/'btc_split.json')==man['split_sha256']
    fixed=read(BASE/'PARTIAL_MATCHED_20261108_0208.json')['videos'];videos={v['recording']:v for v in man['videos']};rows=[];statics=[];cachehashes={}
    method='padding_pilot_f0_s2027_epoch'+str(ck['epoch'])
    for rec in fixed:
        v=videos[rec];assert split['people'][v['person']]['partition']=='dev'
        for cond in ['native','c5']:
            d=BASE/'dev'/v['drive_id']/cond;receipt=read(d/'COMPLETE.json')
            assert sha(d/'crop.mkv')==receipt['crop_sha256'] and sha(d/'meta.npz')==receipt['meta_sha256']
            cachehashes[str(d)]=sha(d/'COMPLETE.json')
            windows,t,valid,_=windows_base(v,d,cond);rgb=crop_rgb(d,len(t));done_static=False
            for w in windows:
                if w['scheme']!='10s':continue
                p=None;reason=None
                try:
                    x=image_clip(rgb,t,valid,w['start'],10)
                    with torch.inference_mode():y=model(torch.from_numpy(x[None]))[0].numpy()
                    assert np.isfinite(y).all();p=decode_model(y)
                    if not done_static:
                        static=np.repeat(x[:1],len(x),axis=0)
                        with torch.inference_mode():sy=model(torch.from_numpy(static[None]))[0].numpy()
                        assert np.isfinite(sy).all();hr=None;why=None
                        try:hr=decode_model(sy)
                        except ValueError as e:why=str(e)
                        statics.append(dict(recording=rec,condition=cond,start_s=w['start'],static_hr=hr,reason=why,pass_static=hr is None,output_std=float(sy.std())))
                        done_static=True
                except ValueError as e:reason=str(e)
                rows.append(dict(w,method=method,prediction=p,reason=reason,truth=v['pr_bpm']))
            state('cpu_static_and_BTC',last_recording=rec,condition=cond,checkpoint_sha256=cksha)
    metrics=[];vr=[]
    for c in ['native','c5']:
        m,r=describe(method,c,[r for r in rows if r['condition']==c],videos);metrics.append(m);vr.extend(dict(method=method,condition=c,**x) for x in r)
    assert sha(ckpath)==cksha
    csvout(OUT/'WINDOWS.csv',rows);csvout(OUT/'VIDEO.csv',vr);csvout(OUT/'STATIC_BTC.csv',statics)
    save(OUT/'COMPLETE.json',dict(checkpoint_sha256=cksha,epoch=ck['epoch'],synthetic_static=synth,btc_static=statics,metrics=metrics,cache_receipt_hashes=cachehashes,holdout_evaluations=0))
    # Include the exact same cohort's OOF constant and main models, never compare unequal samples silently.
    base=read(HERE/'fixed10/SUMMARY.json');primary=['constant','POS','CHROM','ubfc_only_median15','v3_joint_median15','v4_repaired_median11']
    allmetrics=[m for m in base['metrics'] if m['method'] in primary]+metrics
    csvout(OUT/'DISTRIBUTION_WITH_BASELINES.csv',allmetrics)
    txt=['# Pilot sửa padding trên BTC — mẫu cố định 10 video / 7 người','','Một checkpoint fold0/seed2027; không phải ensemble 15. BTC chỉ dùng để đánh giá, không chọn epoch.','',f"Static tổng hợp: {'PASS' if synth['hr'] is None else 'FAIL'}; BTC static: {sum(s['pass_static'] for s in statics)}/{len(statics)} điều kiện-video abstain.",'','|Phương pháp|Điều kiện|MAE|Video/người/cửa sổ|HR trung bình ± SD|Gần 42|Gần 180|≥175|SD trong video|Spearman|','|---|---|---:|---|---|---|---|---|---:|---:|']
    for m in allmetrics:txt.append(f"|{m['method']}|{m['condition']}|{num(m['MAE'])}|{m['n_videos']}/{m['n_people']}/{m['n_windows']}|{num(m['prediction_mean'])} ± {num(m['prediction_sd'])}|{m['near42_count']}/{m['n_windows']}|{m['near180_count']}/{m['n_windows']}|{m['ge175_count']}/{m['n_windows']}|{num(m['mean_within_video_sd'])}|{num(m['spearman_person'])}|")
    (OUT/'REPORT.md').write_text('\n'.join(txt)+'\n',encoding='utf-8');state('complete',checkpoint_sha256=cksha)
if __name__=='__main__':
    try:main()
    except Exception as e:state('error',error=repr(e),traceback=traceback.format_exc());raise
