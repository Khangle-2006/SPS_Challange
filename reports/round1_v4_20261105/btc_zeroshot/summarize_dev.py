"""Person-first dev tables; no training, no holdout access, no automatic promotion."""
import csv,json,hashlib,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def person_metrics(rows):
    people=defaultdict(list)
    for r in rows:
        if r['prediction'] is not None:people[r['person']].append(r['prediction']-r['truth'])
    return {p:np.array([np.abs(e).mean(),np.quantile(np.abs(e),.95),np.mean(e)]) for p,e in people.items()}
def boot(rows,other=None):
    if other is not None:
        # Pair at video first, then people. Do not compare different accepted video sets.
        a={r['recording']:r for r in rows if r['prediction'] is not None};b={r['recording']:r for r in other if r['prediction'] is not None}
        keys=sorted(a.keys()&b.keys());x=person_metrics([a[k] for k in keys]);y=person_metrics([b[k] for k in keys])
        values=np.array([x[p]-y[p] for p in sorted(x.keys()&y.keys())])
    else:values=np.array(list(person_metrics(rows).values()))
    if not len(values):return dict(people=0,MAE=None,P95=None,bias=None,CI95=None)
    rng=np.random.default_rng(0);samples=rng.integers(0,len(values),(10000,len(values)))
    ci=np.quantile(values[samples].mean(1),[.025,.975],axis=0).T.tolist()
    return dict(people=len(values),MAE=float(values[:,0].mean()),P95=float(values[:,1].mean()),bias=float(values[:,2].mean()),CI95=ci)
def calibration(rows):
    result=[];fits=[]
    for fold in range(5):
        tr=[r for r in rows if r['fold']!=fold and r['prediction'] is not None]
        va=[r for r in rows if r['fold']==fold]
        train_ids={r['person'] for r in tr};assert not train_ids&{r['person'] for r in va}
        if len(train_ids)<2:
            result += [dict(r,prediction=None) for r in va];continue
        counts=Counter(r['person'] for r in tr);w=np.array([1/counts[r['person']] for r in tr])
        x=np.array([[r['prediction'],1] for r in tr]);y=np.array([r['truth'] for r in tr])
        if np.linalg.matrix_rank(x)<2:
            result += [dict(r,prediction=None) for r in va];continue
        coef=np.linalg.lstsq(x*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
        fits.append(dict(fold=fold,a=float(coef[0]),b=float(coef[1]),train_people=sorted(train_ids)))
        result += [dict(r,prediction=float(coef[0]*r['prediction']+coef[1]) if r['prediction'] is not None else None) for r in va]
    return result,fits
def strata(v):
    m=v['media'];pr=v['pr_bpm'];task=v['task'].casefold()
    activity='Resting' if task=='resting' else 'Walking' if task=='walking' else 'Task' if task.startswith('task') else 'Other'
    res=f"{m['width']}x{m['height']}"
    return ['all','group='+v['group'],'activity='+activity,'skin='+v['skin'],'lighting='+v['lighting'],'resolution='+res,
        'codec='+m['codec'],'HDR='+('yes' if m['transfer'] in ['arib-std-b67','smpte2084'] else 'no'),
        'PR='+('<60' if pr<60 else '60-<100' if pr<100 else '>=100')]
def format_value(x):return '—' if x is None else f'{x:.3f}'
def main():
    manifest=read(HERE/'dev_manifest.json');videos=manifest['videos'];byrec={v['recording']:v for v in videos}
    windows=defaultdict(list);checkpoint_counts=Counter();spans={};exclusions=[];qc=[]
    for v in videos:
        root=HERE/'dev'/v['drive_id'];src=root/'SOURCE.json'
        if src.exists():spans[v['recording']]=read(src)['source_span_s']
        if (root/'COMPLETE.json').exists():
            info=read(root/'COMPLETE.json')
            if info['state']!='complete':exclusions.append(dict(recording=v['recording'],person=v['person'],reason=info.get('reason',info['state'])))
            for condition in ['native','c5']:
                if condition in info:qc.append(dict(recording=v['recording'],person=v['person'],condition=condition,**info[condition]))
    for r in manifest['missing']:exclusions.append(dict(recording=r['recording'],person=r['person'],reason=r['reason']))
    if (HERE/'CHECKPOINTS.json').exists():checkpoint_counts=Counter(e['branch'] for e in read(HERE/'CHECKPOINTS.json'))
    for file in (HERE/'predictions').glob('*/*/*.json'):
        data=read(file);assert data['split_sha256']==manifest['split_sha256']
        for r in data['rows']:
            assert r['recording'] in byrec
            windows[r['method'],r['condition'],r['scheme']].append(r)
    # Freeze membership by available checkpoint set; a partial ensemble is explicitly named.
    for condition in ['native','c5']:
        for scheme in ['10s','10s_stride5','30s']:
            for branch in ['ubfc_only','v3_joint','v4_repaired']:
                names=sorted(k[0] for k in windows if k[1:]==(condition,scheme) and k[0].startswith(branch+'_f'))
                expected=checkpoint_counts[branch]
                if not expected or len(names)!=expected:continue
                grouped=defaultdict(list)
                for name in names:
                    for r in windows[name,condition,scheme]:grouped[r['recording'],r['start']].append(r)
                label=branch+'_median'+str(expected)
                for rr in grouped.values():
                    good=len(rr)==expected and all(r['prediction'] is not None for r in rr)
                    windows[label,condition,scheme].append(dict(rr[0],method=label,prediction=float(np.median([r['prediction'] for r in rr])) if good else None))
            v4='v4_repaired_median'+str(checkpoint_counts['v4_repaired'])
            names=['POS','CHROM',v4];grouped=defaultdict(dict)
            for name in names:
                for r in windows.get((name,condition,scheme),[]):grouped[r['recording'],r['start']][name]=r
            for rr in grouped.values():
                if not all(n in rr for n in names):continue
                good=all(rr[n]['prediction'] is not None for n in names);label='median_POS_CHROM_'+v4
                windows[label,condition,scheme].append(dict(rr['POS'],method=label,prediction=float(np.median([rr[n]['prediction'] for n in names])) if good else None))
    records={};denominators={}
    for k,ww in windows.items():
        grouped=defaultdict(list)
        for r in ww:grouped[r['recording']].append(r)
        rows=[]
        for v in videos:
            rr=grouped.get(v['recording'],[]);pred=[r['prediction'] for r in rr if r['prediction'] is not None]
            rows.append(dict(recording=v['recording'],person=v['person'],fold=v['fold'],truth=v['pr_bpm'],prediction=float(np.median(pred)) if pred else None))
        records[k]=rows;denominators[k]=ww
    # Constant is fitted exclusively on the other four folds, no validation-label fitting.
    for condition in ['native','c5']:
        for scheme in ['10s','10s_stride5','30s']:
            const={f:float(np.median([v['pr_bpm'] for v in videos if v['fold']!=f])) for f in range(5)}
            duration=30 if scheme=='30s' else 10
            records['constant',condition,scheme]=[dict(recording=v['recording'],person=v['person'],fold=v['fold'],truth=v['pr_bpm'],prediction=const[v['fold']] if spans.get(v['recording'],-1)>=duration else None) for v in videos]
    fits={}
    for k,rows in list(records.items()):
        if '_median' in k[0] or k[0] in ['POS','CHROM']:
            cal,f=calibration(rows);records['calibrated_'+k[0],k[1],k[2]]=cal;fits['|'.join(k)]=f
    save(HERE/'CALIBRATION_OOF.json',fits)
    summary=[];allstrata=sorted({s for v in videos for s in strata(v)})
    for k,rows in sorted(records.items()):
        for group in allstrata:
            subset=[r for r in rows if group in strata(byrec[r['recording']])]
            if not subset:continue
            mm=boot(subset);accepted=[r for r in subset if r['prediction'] is not None]
            wr=[r for r in denominators.get(k,[]) if group in strata(byrec[r['recording']])];numeric=[r for r in wr if r['prediction'] is not None]
            near=sum(abs(r['prediction']-42)<=2 or abs(r['prediction']-180)<=2 for r in accepted)
            duration=30 if k[2]=='30s' else 10;stride=5 if k[2]=='10s_stride5' else duration
            exact_windows=sum(max(0,math.floor((spans[r['recording']]-duration+1e-8)/stride)+1) for r in subset if r['recording'] in spans)
            denominators_complete=all(r['recording'] in spans for r in subset)
            summary.append(dict(method=k[0],condition=k[1],scheme=k[2],group=group,**mm,video_total=len(subset),video_accepted=len(accepted),
                video_coverage=len(accepted)/len(subset),boundary_videos=near,boundary_fraction=near/len(accepted) if accepted else None,
                emitted_windows=len(wr),accepted_windows=len(numeric),expected_windows=exact_windows if denominators_complete else None,
                window_coverage=len(numeric)/exact_windows if denominators_complete and exact_windows and k[0] in {x[0] for x in denominators} else None,
                window_coverage_emitted=len(numeric)/len(wr) if wr else None))
    # A report is allowed early; every selection/gate remains pending until all cache and45models finish.
    completed=(read(HERE/'INFERENCE_STATUS.json').get('state')=='complete') if (HERE/'INFERENCE_STATUS.json').exists() else False
    paired=[]
    for k,rows in sorted(records.items()):
        if k[0].startswith(('calibrated_','median_')) or k[0].endswith('median15') or k[0] in ['POS','CHROM']:
            for baseline in ['constant','POS','CHROM','v3_joint_median15']:
                other=records.get((baseline,k[1],k[2]));
                if other is None or baseline==k[0]:continue
                for group in ['all','activity=Resting']:
                    a=[r for r in rows if group in strata(byrec[r['recording']])];b=[r for r in other if group in strata(byrec[r['recording']])]
                    paired.append(dict(method=k[0],baseline=baseline,condition=k[1],scheme=k[2],group=group,delta=boot(a,b)))
    save(HERE/'SUMMARY.json',dict(state='complete_dev' if completed else 'partial_do_not_select',checkpoint_counts=dict(checkpoint_counts),metrics=summary,paired=paired,
        selected_candidate=None,fine_tune_gate='pending_complete_matched_dev_and_rules',holdout_evaluations=0,split_sha256=manifest['split_sha256']))
    save(HERE/'EXCLUSIONS.json',exclusions)
    qctables=[]
    for condition in ['native','c5']:
        for group in allstrata:
            rr=[r for r in qc if r['condition']==condition and group in strata(byrec[r['recording']])]
            if rr:qctables.append(dict(condition=condition,group=group,videos=len(rr),frames=sum(r['frames'] for r in rr),valid_frames=sum(r['valid_frames'] for r in rr),
                frame_weighted_success=sum(r['valid_frames'] for r in rr)/sum(r['frames'] for r in rr),mean_video_success=float(np.mean([r['face_success_fraction'] for r in rr]))))
    save(HERE/'QC_TABLES.json',qctables)
    for filename,rows in [('ALL_METRICS.csv',summary),('VIDEO_PREDICTIONS.csv',[dict(method=k[0],condition=k[1],scheme=k[2],**r) for k,rr in records.items() for r in rr])]:
        with (HERE/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# BTC zero-shot — báo cáo dev','',('ĐÃ ĐỦ SUY LUẬN DEV; chưa mở holdout.' if completed else 'ĐANG CHẠY; số dưới đây chỉ là phần đã có, chưa chọn ứng viên và chưa xét fine-tune.'),'',
        '161 người dev; 41 người holdout khóa. Độ phủ video luôn chia cho toàn bộ221 video dev có sẵn; hai dòng thiếu video được ghi riêng. Video dưới10s không chấm mô hình; constant chỉ dùng nhãn nên độ phủ không tương đương.','',
        'MAE: mỗi video lấy trung vị HR các cửa sổ, tính sai số tuyệt đối so với scalar PR, trung bình video trong người rồi trung bình người. P95: tính phân vị95 sai số video trong mỗi người rồi trung bình người. Bias dương nghĩa là đo cao hơn PR. CI95 bootstrap10.000 lần theo người. PR không có timestamp đồng bộ nên sai số còn chứa bất định nhãn.','',
        '|Phương pháp|Điều kiện|MAE bpm|P95 bpm|Bias bpm|Video chấm/221|CI95 MAE|','|---|---|---:|---:|---:|---:|---|']
    for r in summary:
        if r['group']=='all' and r['scheme']=='10s' and '_f' not in r['method']:
            text.append(f"|{r['method']}|{r['condition']}|{format_value(r['MAE'])}|{format_value(r['P95'])}|{format_value(r['bias'])}|{r['video_accepted']}/{r['video_total']}|{r['CI95'][0] if r['CI95'] else '—'}|")
    text+=['','Bảng đầy đủ mọi checkpoint × điều kiện × nhóm × cách cắt cửa sổ: ALL_METRICS.csv. Chi tiết từng video: VIDEO_PREDICTIONS.csv. CI ghép cặp lưu trong SUMMARY.json; chỉ so trên cùng video có dự đoán ở cả hai phía.','',
        'Giả thuyết zero-shot, ensemble và hiệu chỉnh: đang chờ đầy đủ. Fine-tune: chưa chạy. Holdout: chưa mở. Static BTC và donor-label controls: chưa hoàn tất. Nhánh nộp: giữ V3; chưa xác định nguồn test ẩn.','',
        'Độ phủ cửa sổ cuối cùng chia cho mọi cửa sổ ứng viên theo PTS, kể cả lỗi dò mặt; trong lúc chưa đọc hết PTS để null thay vì lấy số đã phát ra làm toàn bộ mẫu số. Hiệu chỉnh hoạt động trên HR cấp video nên không có độ phủ cửa sổ riêng. Độ phủ video cả các dòng bị thiếu: mẫu số dev223; bảng chính221 video đã tải.','',
        '## Video không chấm được / bị thiếu','', '|Video|Lý do|','|---|---|']
    text += [f"|{r['recording']}|{r['reason']}|" for r in exclusions]
    (HERE/'BTC_ZEROSHOT_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
if __name__=='__main__':main()
