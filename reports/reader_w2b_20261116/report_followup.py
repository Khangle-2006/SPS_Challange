"""Read completed receipts; generate paired reports without training or inference."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import json, sys, time, fcntl, traceback
from pathlib import Path
import numpy as np
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent/'round1_execution_20261110'))
from report_metrics import metric, belongs

def read(p):
    return json.loads(p.read_text())

def save(name, value):
    p = P/name
    q = p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(value, indent=2))
    q.replace(p)

def contrast(rows, method, reference, group):
    a = {r['recording']:r for r in rows if r['method']==method and belongs(r,group) and r['prediction'] is not None}
    b = {r['recording']:r for r in rows if r['method']==reference and belongs(r,group) and r['prediction'] is not None}
    keys = sorted(a.keys() & b.keys())
    people = sorted({a[k]['person'] for k in keys})
    if not people:return None
    for k in keys:
        assert (a[k]['person'],a[k]['truth']) == (b[k]['person'],b[k]['truth'])
    d = np.array([np.mean([abs(a[k]['prediction']-a[k]['truth'])-abs(b[k]['prediction']-b[k]['truth']) for k in keys if a[k]['person']==p]) for p in people])
    ix = np.random.default_rng(0).integers(len(people),size=(10000,len(people)))
    return dict(delta=float(d.mean()),CI95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),videos=len(keys),people=len(people))

def report_w2b():
    old = read(P.parent/'reader_v1_20261116/PREDICTIONS.json')
    lookup = {(r['recording'],r['method']):r for r in old if r['condition']=='native'}
    rows=[]; histories={}; completed=[]
    for d in sorted((P/'W2b/runs').glob('fold*')):
        if (d/'history.json').exists():histories[d.name]=read(d/'history.json')
        if not (d/'COMPLETE.json').exists():continue
        completed.append(d.name)
        method='W2b_shuffle' if d.name=='fold2_shuffle' else 'W2b_true'
        for r in read(d/'NATIVE.json'):rows.append(dict(r,method=method))
    for recording in sorted({r['recording'] for r in rows}):
        for method in ['constant','calibrated','reader_v1']:
            rows.append(dict(lookup[recording,method]))
    metrics=[]; groups=['all','Normal-Resting','Abnormal-Resting','Task','Walking']
    for method in ['W2b_true','W2b_shuffle']:
        for g in groups:
            a=[r for r in rows if belongs(r,g)]
            stats=metric(a,['constant','calibrated','reader_v1',method])
            matched={r['recording'] for r in a if r['method']==method and r['prediction'] is not None}
            for r in stats:
                values=[v['prediction'] for v in a if v['method']==r['method'] and v['recording'] in matched and v['prediction'] is not None]
                r['video_SD']=float(np.std(values)) if values else None
            metrics.extend(dict(r,group=g,comparison=method) for r in stats)
    paired={g:contrast(rows,'W2b_true','W2b_shuffle',g) for g in groups}
    disposition=read(P/'W2B_DISPOSITION.json') if (P/'W2B_DISPOSITION.json').exists() else None
    gate=read(P/'W2B_EXPANSION_GATE.json') if (P/'W2B_EXPANSION_GATE.json').exists() else None
    save('W2B_REPORT.json',dict(completed=completed,histories=histories,metrics=metrics,true_minus_shuffle=paired,gate=gate,disposition=disposition,rows=rows,holdout_evaluations=0))
    lines=['# W2b — kết quả đã hoàn tất', '', 'Chỉ dùng epoch cuối. MAE tính trong từng người rồi trung bình; bootstrap ghép cặp theo người 10.000 lần, seed 0. Kết quả một fold không phải kết quả toàn bộ BTC dev.', '', 'Hoàn tất: '+', '.join(completed), '', '|Nhóm|Phương pháp|Video/người|MAE|Bias|Spearman|Δ hằng số [CI95]|Δ hiệu chỉnh [CI95]|','|---|---|---:|---:|---:|---:|---|---|']
    for r in metrics:lines.append(f"|{r['group']}|{r['method']}|{r['videos']}/{r['people']}|{r['MAE']:.5f}|{r['bias']:.5f}|{r['Spearman']}|{r['delta_constant']:.5f} {r['delta_constant_CI']}|{r['delta_calibrated']:.5f} {r['delta_calibrated_CI']}|")
    lines += ['', '## Học trên 40 video train cố định', '|Lượt|Epoch|Weak CE train|MAE train40|Ảnh tĩnh|', '|---|---:|---:|---:|---|']
    for name,h in histories.items():
        for r in h:lines.append(f"|{name}|{r['epoch']}|{r['weak_ce']:.6f}|{r['train40']['MAE']}|{r['static_pass']}|")
    lines += ['', 'SD dự đoán cấp video nằm trong W2B_REPORT.json (video_SD).', '', '## Fold 2: nhãn thật trừ nhãn xáo', 'Số âm: nhãn thật tốt hơn. Chỉ giao video validation chung.', '```json',json.dumps(paired,indent=2),'```','','## Cổng mở rộng','```json',json.dumps(gate,indent=2),'```','','Trạng thái: '+json.dumps(disposition)]
    (P/'W2B_REPORT.md').write_text('\n'.join(lines))

def review_c():
    results=read(P/'C_RESULTS.json')
    # The extraction-time formatter attaches the reader's delta-native to all
    # rows. Recompute each method's contrast from saved predictions explicitly.
    for a in results:
        for r in a['metrics']:
            d=contrast(a['rows'],r['method'],'reader_native',r['group'])
            assert d is not None
            r.update(delta_native=d['delta'],delta_native_CI=d['CI95'])
        for r in a['rows']:
            if r['method']=='reader_gated':r['prior_only']=r['beta']==0 or r['abstain']
    save('C_REVIEWED_RESULTS.json',results)
    lines=['# Reader dưới C1/C4', '', 'Full-frame simulate.py cố định; C1 gồm nhiễu và codec lưu trữ x264, không phải phép thử nhiễu cô lập. Không có holdout. Mỗi điều kiện fit prior/beta trên train-fold; beta chỉ được giữ khi giảm MAE train ít nhất 0,5 bpm.', '', 'MAE theo người; CI bootstrap ghép cặp 10.000 lần, seed 0. Hàng calibrated là dự đoán native lịch sử trên cùng video, không phải bản fit lại theo điều kiện.', '', '|Điều kiện|Nhóm|Phương pháp|Video/người|MAE|Bias|Spearman|Δ hằng số [CI95]|Δ reader native [CI95]|','|---|---|---|---:|---:|---:|---:|---|---|']
    for a in results:
        for r in a['metrics']:lines.append(f"|{a['condition']}|{r['group']}|{r['method']}|{r['videos']}/{r['people']}|{r['MAE']:.5f}|{r['bias']:.5f}|{r['Spearman']}|{r['delta_constant']:.5f} {r['delta_constant_CI']}|{r['delta_native']:.5f} {r['delta_native_CI']}|")
    lines += ['', '## Độ phủ và phổ hợp lệ','```json',json.dumps({a['condition']:a['coverage'] for a in results},indent=2),'```']
    (P/'C_REVIEWED_REPORT.md').write_text('\n'.join(lines))
    save('C_REVIEWED_COMPLETE.json',dict(passed=True,conditions=5,holdout_evaluations=0,source='C_RESULTS.json',correction='per-method paired native contrast in report only; no predictions or fits changed'))

def main():
    lock=(P/'report_followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    last=None
    while True:
        paths=list((P/'W2b/runs').glob('fold*/history.json'))+list((P/'W2b/runs').glob('fold*/COMPLETE.json'))+[p for p in [P/'W2B_DISPOSITION.json',P/'W2B_EXPANSION_GATE.json'] if p.exists()]
        signature=[(str(p),p.stat().st_mtime_ns) for p in paths]
        if signature!=last:report_w2b();last=signature
        if (P/'C_COMPLETE.json').exists() and not (P/'C_REVIEWED_COMPLETE.json').exists():review_c()
        wt=(P/'W2B_DISPOSITION.json').exists() or (P/'W2B_ERROR.json').exists()
        ct=(P/'C_REVIEWED_COMPLETE.json').exists() or any(P.glob('C_ERROR_SHARD*.json'))
        save('REPORT_STATUS.json',dict(pid=os.getpid(),W2b_terminal=wt,C_terminal=ct,updated_epoch=time.time()))
        if wt and ct:return
        time.sleep(30)

if __name__=='__main__':
    try:main()
    except Exception:save('REPORT_ERROR.json',dict(traceback=traceback.format_exc()));raise
