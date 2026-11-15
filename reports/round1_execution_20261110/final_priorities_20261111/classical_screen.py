"""Blind OMIT + preregistered nuisance regression on existing dev traces only."""
import os,sys,json,csv,traceback,subprocess
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P.parent;R=E.parent.parent;V=R/'reports/round1_v4_20261105';B=R/'reports/round1_bayes_20261108'
sys.path.insert(0,str(V/'btc_zeroshot'));sys.path.insert(0,str(E))
from infer_dev import read,sha,save,supported,decode_wave,classical_functions,candidate_windows
from report_metrics import metric,belongs
from nuisance import regress,omit
import padding_queue as guard
os.environ['CUDA_VISIBLE_DEVICES']=''
NS,_=classical_functions();OUT=P/'classical_screen';OUT.mkdir(exist_ok=True)
def frames(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,pict_type','-of','json',str(path)]))['frames']
def types_for(j,condition,d):
    if condition=='native':
        file=B/'native_gop'/f"{j['recording']}_frames.csv"
        a=list(csv.DictReader(file.open()));return [v['type'] for v in a],dict(path=str(file),sha256=sha(file))
    if condition=='c5':
        manifest=d.parent/'c5_encoded/manifest.json';m=read(manifest);video=Path(m['video']);g=frames(video)
        return [v['pict_type'] for v in g],dict(path=str(video),sha256=sha(video),manifest_sha256=sha(manifest))
    file=d.parent/'GOP.json';g=read(file)['frames'];return [v['pict_type'] for v in g],dict(path=str(file),sha256=sha(file),receipt_sha256=sha(d.parent/'COMPLETE.json'))
def one(j,condition,d,windows):
    c=read(d/'COMPLETE.json');assert sha(d/'meta.npz')==c['meta_sha256']
    with np.load(d/'meta.npz') as z:t=z['t'].copy();valid=z['valid'].copy();rgb=z['rgb_mean'].copy();boxes=z['boxes'].copy()
    types,evidence=types_for(j,condition,d);assert len(types)==len(t),'GOP/crop ordinal mismatch'
    clean,fit=regress(rgb,types,boxes,valid)
    result=[]
    for w in windows:
        start=w['start'];elig=w.get('eligible',True)
        for method in ['POS','CHROM','G','OMIT','POS_schedule','CHROM_schedule','G_schedule']:
            pred=None;reason=None
            try:
                if not elig:raise ValueError('reference_ineligible')
                if not supported(t,valid,start,start+10):raise ValueError('image_support_gap_or_tracking_failure')
                grid=start+np.arange(300)/30;data=clean if method.endswith('_schedule') else rgb
                x=np.stack([np.interp(grid,t,data[:,i]) for i in range(3)],axis=1);name=method.split('_')[0]
                wave=-x[:,1] if name=='G' else omit(x) if name=='OMIT' else NS[name.lower()](x,30)
                pred=decode_wave(wave)
            except ValueError as e:reason=str(e)
            result.append(dict(dataset=j['dataset'],recording=j['recording'],person=j['person'],fold=j['fold'],condition=condition,method=method,start=start,truth=w['truth'],prediction=pred,reason=reason,group=j.get('group'),task=j.get('task')))
    return dict(rows=result,fit=fit,evidence=evidence,meta_sha256=sha(d/'meta.npz'),code_sha256=sha(__file__),nuisance_sha256=sha(P/'nuisance.py'),roi='existing oval36 RGB mean',deployment_allowed=False)
def report():
    from report_c5 import public_score
    rows=[r for p in sorted((OUT/'predictions').glob('*/*/*.json')) for r in read(p)['rows']]
    baseline=[r for r in read(E/'C5_BTC_VIDEO.json') if r['method'] in ['constant','calibrated']]
    video=[]
    for rid,c,m in sorted({(r['recording'],r['condition'],r['method']) for r in rows if r['dataset']=='BTC'}):
        rr=[r for r in rows if (r['recording'],r['condition'],r['method'])==(rid,c,m) and r['prediction'] is not None]
        if rr:video.append(dict(rr[0],prediction=float(np.median([r['prediction'] for r in rr]))))
    methods=['constant','calibrated','POS','CHROM','G','OMIT','POS_schedule','CHROM_schedule','G_schedule'];tables=[]
    for c in sorted({r['condition'] for r in video}):
        for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
            paired=[r for r in baseline+video if r['condition']==c and belongs(r,group)]
            tables.extend(dict(condition=c,group=group,**r) for r in metric(paired,methods))
    public=[]
    for ds,c in sorted({(r['dataset'],r['condition']) for r in rows if r['dataset']!='BTC'}):
        rr=[r for r in rows if r['dataset']==ds and r['condition']==c];look={m:{(r['recording'],r['start']):r for r in rr if r['method']==m and r['prediction'] is not None and r['truth'] is not None} for m in methods[2:]}
        common=set.intersection(*(set(v) for v in look.values()))
        for m in look:
            public.append(dict(dataset=ds,condition=c,method=m,**public_score([look[m][k] for k in sorted(common)])))
    save(OUT/'BTC_METRICS.json',tables);save(OUT/'PUBLIC_METRICS.json',public);save(OUT/'VIDEO_PREDICTIONS.json',video)
    save(OUT/'COMPLETE.json',dict(rows=len(rows),BTC_valid_videos=len({r['recording'] for r in video}),files=len(list((OUT/'predictions').glob('*/*/*.json'))),holdout_evaluations=0,diagnostic_only=True))
    lines=['# OMIT và hồi quy lịch codec — sàng lọc dev','','Hồi quy chỉ là đối chứng; không được dùng để nộp nếu probe nhịp thật bị triệt. Các hàng ghép cùng video; MAE tính theo người; bootstrap 10000 seed0. Mốc hiệu chỉnh là cấu hình đã lưu, tính lại trên đúng giao video của từng điều kiện. Không diễn giải 11,37 như mốc của mọi nhóm/cohort.','','| Điều kiện | Nhóm | Phương pháp | Người/video | MAE | Δ hiệu chỉnh [CI95%] |','|---|---|---|---:|---:|---|']
    for r in tables:lines.append(f"|{r['condition']}|{r['group']}|{r['method']}|{r['people']}/{r['videos']}|{r['MAE']:.4f}|{r['delta_calibrated']:+.4f} {r['delta_calibrated_CI']}|")
    lines+=['','Các lỗi/thiếu hỗ trợ được giữ trong từng file predictions; chưa có kết luận cơ chế hoặc cải thiện sinh lý. C5′ là proxy ffmpeg, chưa xác minh MATLAB.']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
def main():
    lease=guard.lock(OUT/'worker.lock');assert not (OUT/'LAUNCH.json').exists()
    save(OUT/'LAUNCH.json',dict(pid=os.getpid(),utc=guard.now(),budget=guard.budget(fresh=True),nuisance_sha256=sha(P/'nuisance.py'),code_sha256=sha(__file__)))
    dev=read(V/'btc_zeroshot/dev_manifest.json')['videos'];split=read(V/'btc_zeroshot/btc_split.json');jobs=[]
    for v in dev:
        assert split['people'][v['person']]['partition']=='dev';j=dict(v,dataset='BTC')
        span=read(V/'btc_zeroshot/dev'/v['drive_id']/'SOURCE.json')['source_span_s'];windows=[dict(start=s,truth=v['pr_bpm']) for scheme,s,seconds in candidate_windows(span) if scheme=='10s']
        for c in ['native','c5']:
            d=V/'btc_zeroshot/dev'/v['drive_id']/c
            if (d/'COMPLETE.json').exists():jobs.append((j,c,d,windows))
    refs=read(B/'public_comparison/REFERENCE_COMPACT.json');meta={v['recording']:v for v in dev}
    for cp in sorted((E/'c5prime').glob('*/*/*/COMPLETE.json')):
        a=read(cp)
        if a['state']!='complete' or not a['bitrate_pass']:continue
        if a['dataset']=='BTC':
            v=meta[a['recording']];j=dict(v,dataset='BTC');span=read(V/'btc_zeroshot/dev'/v['drive_id']/'SOURCE.json')['source_span_s'];windows=[dict(start=s,truth=v['pr_bpm']) for scheme,s,seconds in candidate_windows(span) if scheme=='10s']
        else:
            j=a;r=next(r for r in refs['records'] if r['dataset']==a['dataset'] and r['recording_id']==a['recording'] and r['variant']=='native');windows=[dict(start=w['window_start_s'],truth=w['reference_hr_bpm'],eligible=w['reference_eligible']) for w in r['windows']]
        jobs.append((j,'c5prime_'+a['variant'],cp.parent/'crop',windows))
    for i,(j,c,d,windows) in enumerate(jobs):
        if i%40==0:guard.budget()
        a=one(j,c,d,windows);save(OUT/'predictions'/j['dataset']/j['recording']/(c+'.json'),a)
        save(OUT/'STATUS.json',dict(completed=i+1,total=len(jobs),recording=j['recording'],condition=c,pid=os.getpid()))
    report()
if __name__=='__main__':
    try:main()
    except Exception as e:save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
