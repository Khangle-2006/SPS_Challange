"""Read-only snapshot of BTC dev predictions; never inspect holdout labels/media."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,hashlib,sys,datetime,warnings
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import spearmanr
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
sys.path.insert(0,str(BASE))
from summarize_dev import boot
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def csvout(p,rows):
    if not rows:return
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def num(v):return '—' if v is None else f'{v:.3f}'
def describe(method,condition,ww,videos):
    by=defaultdict(list)
    for r in ww:by[r['recording']].append(r)
    vr=[];within=[]
    for rec,rr in sorted(by.items()):
        v=videos[rec];pp=[r['prediction'] for r in rr if r['prediction'] is not None]
        if not pp:continue
        sd=float(np.std(pp,ddof=1)) if len(pp)>1 else None
        vr.append(dict(recording=rec,person=v['person'],fold=v['fold'],truth=v['pr_bpm'],prediction=float(np.median(pp)),windows=len(pp),within_video_sd=sd))
        if sd is not None:within.append(sd)
    pp=np.array([r['prediction'] for r in vr]);wp=np.array([r['prediction'] for r in ww if r['prediction'] is not None]);people=defaultdict(list)
    for r in vr:people[r['person']].append(r)
    x=[np.mean([r['prediction'] for r in rr]) for rr in people.values()];y=[np.mean([r['truth'] for r in rr]) for rr in people.values()]
    rho=float(spearmanr(x,y).statistic) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None
    if rho is not None and not np.isfinite(rho):rho=None
    m=boot(vr)
    low=int(np.sum(np.abs(wp-42)<=2));high=int(np.sum(np.abs(wp-180)<=2));top=int(np.sum(wp>=175))
    result=dict(method=method,condition=condition,n_videos=len(vr),n_people=len(people),n_windows=len(wp),expected_windows=len(ww),
        MAE=m['MAE'],MAE_CI95=m['CI95'][0] if m['CI95'] else None,bias=m['bias'],prediction_mean=float(pp.mean()) if len(pp) else None,
        prediction_sd=float(pp.std(ddof=1)) if len(pp)>1 else None,near42_count=low,near180_count=high,ge175_count=top,
        near42_fraction=low/len(wp) if len(wp) else None,near180_fraction=high/len(wp) if len(wp) else None,ge175_fraction=top/len(wp) if len(wp) else None,
        mean_within_video_sd=float(np.mean(within)) if within else None,n_videos_with_sd=len(within),n_videos_exactly3=sum(r['windows']==3 for r in vr),spearman_person=rho)
    return result,vr
def main():
    man=read(BASE/'dev_manifest.json');split=read(BASE/'btc_split.json');assert sha(BASE/'btc_split.json')==man['split_sha256']
    videos={v['recording']:v for v in man['videos']};assert all(split['people'][v['person']]['partition']=='dev' for v in videos.values())
    fixed=read(BASE/'PARTIAL_MATCHED_20261108_0208.json')['videos']
    inventory=OUT/'CHECKPOINTS_FIXED.json'
    if not inventory.exists():
        ee=[e for e in read(BASE/'CHECKPOINTS.json') if e['branch']!='v4_repaired' or e['fold']<3 or (e['fold']==3 and e['seed'] in [2027,2028])]
        assert Counter(e['branch'] for e in ee)=={'ubfc_only':15,'v3_joint':15,'v4_repaired':11};save(inventory,ee)
    entries=read(inventory);names={e['name']:e for e in entries};rows=defaultdict(dict);sources={}
    for v in videos.values():
        for cond in ['native','c5']:
            folder=BASE/'predictions'/v['drive_id']/cond
            for name in ['classical',*names]:
                p=folder/(name+'.json')
                if not p.exists():continue
                d=read(p);assert d['split_sha256']==man['split_sha256']
                if name!='classical':assert d['checkpoint']['sha256']==names[name]['sha256']
                sources[str(p.relative_to(BASE))]=sha(p)
                for r in d['rows']:
                    if r['scheme']=='10s':rows[r['method'],cond][r['recording'],r['start']]=r
    labels=[]
    for branch in ['ubfc_only','v3_joint','v4_repaired']:
        members=[e['name'] for e in entries if e['branch']==branch];label=f'{branch}_median{len(members)}';labels.append(label)
        for cond in ['native','c5']:
            keys=set().union(*(rows[m,cond] for m in members))
            for k in keys:
                rr=[rows[m,cond].get(k) for m in members]
                if any(r is None for r in rr):continue
                p=float(np.median([r['prediction'] for r in rr])) if all(r['prediction'] is not None for r in rr) else None
                rows[label,cond][k]=dict(rr[0],method=label,prediction=p)
    constants={};constreceipt=[]
    for fold in range(5):
        train=[v for v in videos.values() if v['fold']!=fold];val=[v for v in videos.values() if v['fold']==fold]
        assert not {v['person'] for v in train}&{v['person'] for v in val}
        constants[fold]=float(np.median([v['pr_bpm'] for v in train]));constreceipt.append(dict(fold=fold,PR_median=constants[fold],train_videos=len(train),train_people=len({v['person'] for v in train})))
    primary=['constant','POS','CHROM',*labels]
    for cond in ['native','c5']:
        for k,r in rows['POS',cond].items():rows['constant',cond][k]=dict(r,method='constant',prediction=constants[videos[k[0]]['fold']])
    matched=[]
    for rec,v in videos.items():
        rr=[r for (rrec,start),r in rows['POS','native'].items() if rrec==rec]
        if not rr:continue
        keys=[(rec,r['start']) for r in rr]
        if all(k in rows[m,c] and rows[m,c][k]['prediction'] is not None for m in primary for c in ['native','c5'] for k in keys):matched.append(rec)
    assert set(fixed)<=set(matched),'Original fixed cohort no longer matches complete predictions'
    summaries={}
    for scope,cohort in [('fixed10',fixed),('matched_current',matched)]:
        dest=OUT/scope;dest.mkdir(exist_ok=True);metrics=[];long=[];vlong=[];by_method={}
        for (method,cond),rr in sorted(rows.items()):
            ww=[r for (rec,_),r in rr.items() if rec in cohort]
            if not ww:continue
            m,vr=describe(method,cond,ww,videos);metrics.append(m);by_method[method,cond]=vr
            vlong += [dict(method=method,condition=cond,**r) for r in vr]
            for r in sorted(ww,key=lambda r:(r['recording'],r['start'])):
                v=videos[r['recording']];long.append(dict(recording=r['recording'],person=v['person'],fold=v['fold'],condition=cond,method=method,PR=v['pr_bpm'],start_s=r['start'],seconds=r['seconds'],prediction=r['prediction'],reason=r.get('reason')))
        paired=[]
        for m in primary[1:]:
            for c in ['native','c5']:paired.append(dict(method=m,condition=c,delta_vs_constant=boot(by_method[m,c],by_method['constant',c])))
        save(dest/'SUMMARY.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='partial_no_selection',videos=cohort,metrics=metrics,paired=paired,constant_fit=constreceipt,holdout_evaluations=0))
        csvout(dest/'DISTRIBUTION.csv',metrics);csvout(dest/'WINDOWS_ALL_CHECKPOINTS.csv',long);csvout(dest/'VIDEO_ALL_CHECKPOINTS.csv',vlong)
        wide=[]
        for rec in cohort:
            for c in ['native','c5']:
                keys=sorted(k for k in rows['POS',c] if k[0]==rec)
                for k in keys:wide.append(dict(recording=rec,person=videos[rec]['person'],PR=videos[rec]['pr_bpm'],condition=c,start_s=k[1],**{m:rows[m,c][k]['prediction'] for m in primary}))
        csvout(dest/'WINDOWS_MAIN.csv',wide)
        text=['# BTC: kiểm tra phân bố dự đoán ('+scope+')','','**Mẫu tạm thời**, '+str(len(cohort))+' video; không chọn mô hình, không fine-tune, không mở holdout.','',
        'MAE: sai số tuyệt đối trung bình trong người rồi giữa người. Trung bình ± SD: phân bố HR dự đoán cấp video (bpm), không phải sai số. Gần 42/180 và ≥175 đếm ở cửa sổ 10 giây; mẫu số là số cửa sổ có dự đoán. Spearman đo quan hệ thứ hạng giữa dự đoán và PR sau khi gộp theo người; chưa phải bằng chứng đo được mạch.','',
        '|Phương pháp|Điều kiện|Video/người/cửa sổ|MAE|HR trung bình ± SD|Gần 42|Gần 180|≥175|SD trong video|Spearman người|','|---|---|---|---:|---|---|---|---|---:|---:|']
        for r in metrics:
            if r['method'] in primary:text.append(f"|{r['method']}|{r['condition']}|{r['n_videos']}/{r['n_people']}/{r['n_windows']}|{num(r['MAE'])}|{num(r['prediction_mean'])} ± {num(r['prediction_sd'])}|{r['near42_count']}/{r['n_windows']}|{r['near180_count']}/{r['n_windows']}|{r['ge175_count']}/{r['n_windows']}|{num(r['mean_within_video_sd'])}|{num(r['spearman_person'])}|")
        text+=['','## Từng video, từng cửa sổ','','|Video|PR|Điều kiện|Bắt đầu (s)|Hằng|POS|CHROM|UBFC|V3|V4 (11)|','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in wide:text.append('|'+ '|'.join([r['recording'],num(r['PR']),r['condition'],num(r['start_s']),*[num(r[m]) for m in primary]])+'|')
        text+=['','Các file CSV đi kèm có toàn bộ từng checkpoint, mẫu số, SD và dự đoán. SUMMARY.json lưu CI95 MAE và CI ghép cặp với hằng số. SD dùng ddof=1. SD trong video là trung bình SD của các video có ≥2 cửa sổ; số video đúng 3 cửa sổ lưu riêng. Không coi SD thấp tự nó là chứng cứ hồi quy về hằng số.']
        (dest/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8');summaries[scope]=dict(videos=len(cohort),people=len({videos[r]['person'] for r in cohort}))
    save(OUT/'SOURCE_SNAPSHOT.json',dict(split_sha256=man['split_sha256'],manifest_sha256=sha(BASE/'dev_manifest.json'),fixed_checkpoints_sha256=sha(inventory),code_sha256=sha(__file__),prediction_hashes=sources,cohorts=summaries))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
