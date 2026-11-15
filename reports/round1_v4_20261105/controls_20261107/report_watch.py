"""Publish durable control tables; refresh final gate after all15 audits finish."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import time,csv,json,subprocess,fcntl
from pathlib import Path
from datetime import datetime,timezone
from common import *
HERE=Path(__file__).resolve().parent
def make_report():
    frozen=read(FROZEN);output=[];status=[]
    for run in completed():
        name=run.parent.parent.name+'_'+run.parent.name;dest=HERE/name
        if not (dest/'STATIC.json').exists():continue
        static=read(dest/'STATIC.json');assert static['checkpoint_sha256']==sha(run/'final.pt')
        done=read(dest/'COMPLETE.json') if (dest/'COMPLETE.json').exists() else None
        if done:
            assert done['checkpoint_sha256']==static['checkpoint_sha256'] and done['windows_sha256']==sha(dest/'windows.jsonl')
        data=load_run(run,frozen)
        status.append(dict(checkpoint=name,pass_static=static['pass_static'],static_hr=static['result']['hr'],controls_complete=done is not None))
        for cond in CONDITIONS:
            rr=[r for r in data if r['condition']==cond and r['accepted']]
            main=metrics(rr)
            sr=[dict(r,prediction_hr_bpm=static['result']['hr']) for r in rr]
            for test in ('main','static','static_noise','shuffled','wrong_person_label'):
                m=done['conditions'][cond][test] if done else main if test=='main' else metrics(sr) if test=='static' else None
                output.append(dict(checkpoint=name,condition=cond,test=test,state='complete' if m else 'pending',main_MAE=main['MAE'],**(m or {k:None for k in main})))
    with (HERE/'CONTROL_TABLE.csv').open('w',newline='') as f:
        if output:
            w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    save(HERE/'SUMMARY.json',dict(utc=datetime.now(timezone.utc).isoformat(),checkpoint_screens=len(status),
        static_failed=sum(not r['pass_static'] for r in status),controls_complete=sum(r['controls_complete'] for r in status),total=15,checkpoints=status,
        static_scope='one invariant zero-difference forward per checkpoint; first five real static clips cross-checked within full audit',
        adoption='Keep V3: any original-padding static failure prevents V4 adoption under user gate'))
    lines=['# Kiểm soát V4 — báo cáo đang cập nhật',
        f"\nĐã kiểm tra ảnh tĩnh {len(status)}/15 checkpoint; {sum(not r['pass_static'] for r in status)} FAIL. Đã hoàn tất tất cả kiểm soát {sum(r['controls_complete'] for r in status)}/15 checkpoint.",
        '\nMỗi hàng dùng cùng các cửa sổ được chấp nhận của nhánh chính. MAE bpm theo người; cột >=179 là số cửa sổ vượt ngưỡng / số dự đoán numeric. Nếu tất cả abstain, MAE và tỷ lệ numeric=N/A. MAE ảnh tĩnh/nhiễu so với nhãn video gốc chỉ là chẩn đoán, không phải độ chính xác HR sinh lý. Các phép thử kiểm soát không chứng minh bao nhiêu phần cải thiện thực đến từ tín hiệu giả.',
        '\nTham số đã ghi trong PROTOCOL.md. Không gán giá trị0 cho abstention, không điều chỉnh ngưỡng sau kết quả.',
        '\n| Checkpoint | Điều kiện | Kiểm soát | MAE chính | MAE kiểm soát | >=179 / numeric | Abstain / tổng |','|---|---|---|---:|---:|---:|---:|']
    for r in output:
        if r['state']!='complete' or r['test']=='main':continue
        value='N/A' if r['MAE'] is None else f"{r['MAE']:.3f}"
        lines.append(f"|{r['checkpoint']}|{r['condition']}|{r['test']}|{r['main_MAE']:.3f}|{value}|{r['ceilings']}/{r['numeric']}|{r['abstentions']}/{r['windows']}|")
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return len(status)==15 and all(r['controls_complete'] for r in status)
if __name__=='__main__':
    lock=(HERE/'reporter.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None
    while True:
        signature=tuple((str(p),p.stat().st_mtime_ns) for pattern in ('fold*_seed*/STATIC.json','fold*_seed*/COMPLETE.json') for p in sorted(HERE.glob(pattern)))
        if signature and signature!=previous:
            done=make_report();previous=signature
            if done:
                subprocess.run(['/data_pool/khangle/facevitals/envs/hr/bin/python',str(REPORT/'aggregation_20261107/aggregate.py')],check=True)
                break
        time.sleep(30)
