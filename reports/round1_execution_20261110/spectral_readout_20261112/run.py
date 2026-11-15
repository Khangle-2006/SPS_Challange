import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,csv,hashlib,traceback,fcntl,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from core import extract,permute_blocks,stats
P=Path(__file__).resolve().parent;E=P.parent;R=E.parent.parent;B=R/'reports/round1_bayes_20261108';OLD=R/'reports/round1_v4_20261105/btc_zeroshot'
SPLIT='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def frozen():return {str(x):sha(x) for x in [P/'GATES.md',P/'core.py',P/'run.py',P/'tests.py',B/'pipeline.py',OLD/'btc_split.json',OLD/'dev_manifest.json',B/'VIDEO_PREDICTIONS.csv']}

def prepare():
    assert sha(OLD/'btc_split.json')==SPLIT
    manifest=read(OLD/'dev_manifest.json');assert manifest['split_sha256']==SPLIT
    dev={v['recording']:v for v in manifest['videos']}
    rr=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()))
    cal={r['recording']:r for r in rr if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'}
    const={r['recording']:r for r in rr if r['condition']=='native' and r['method']=='constant'}
    assert len(cal)==218 and len({r['person'] for r in cal.values()})==158
    labels=[];inputs=[]
    for i,k in enumerate(sorted(cal)):
        v=dev[k];c=cal[k];b=const[k];assert v['person']==c['person']==b['person'] and float(v['pr_bpm'])==float(c['truth'])==float(b['truth'])
        d=B/'traces'/k/'native';receipt=read(d/'COMPLETE.json');assert sha(d/'trace.npz')==receipt['trace_sha256']
        labels.append(dict(index=i,recording=k,person=v['person'],fold=v['fold'],truth=float(c['truth']),constant=float(b['prediction']),calibrated=float(c['prediction']),group=v['group'],task=v['task']))
        inputs.append(dict(index=i,path=str(d/'trace.npz'),sha256=receipt['trace_sha256']))
    save(P/'COHORT_LABELS.json',labels);save(P/'EXTRACT_INPUT.json',inputs)

def extraction():
    # Independent child: only label-free manifest, cached RGB, and fixed pipeline.
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_pulse',B/'pipeline.py');pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)
    results=[]
    for item in read(P/'EXTRACT_INPUT.json'):
        assert set(item)=={'index','path','sha256'} and sha(item['path'])==item['sha256']
        with np.load(item['path']) as z:result,powers,masks=extract(z['t'],z['rgb'],pipeline)
        dest=P/'spectra'/f"{item['index']:03d}.npz";dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,score=powers.astype(np.float32),available=masks)
        results.append(dict(index=item['index'],trace_sha256=item['sha256'],spectra_sha256=sha(dest),**result))
        save(P/'EXTRACT_STATUS.json',dict(utc=now(),pid=os.getpid(),completed=len(results),total=218))
    save(P/'CANDIDATES.json',results)
    save(P/'CANDIDATES_FROZEN.json',dict(utc=now(),sha256=sha(P/'CANDIDATES.json'),input_sha256=sha(P/'EXTRACT_INPUT.json'),code_sha256=sha(P/'core.py'),labels_read_by_extractor=False))

def belongs(r,g):
    if g=='all':return True
    if g=='Task':return r['task'].lower().startswith('task')
    if g=='Walking':return r['task'].lower()=='walking'
    return r['task'].lower()=='resting' and r['group']==g.split('-')[0]

def score():
    assert sha(P/'CANDIDATES.json')==read(P/'CANDIDATES_FROZEN.json')['sha256']
    rr=read(P/'COHORT_LABELS.json');cc=read(P/'CANDIDATES.json');assert [r['index'] for r in rr]==[r['index'] for r in cc]
    c=[r['candidates'] for r in cc];y=np.array([r['truth'] for r in rr]);groups=['all','Normal-Resting','Abnormal-Resting','Task','Walking'];reals={};nulls={g:[] for g in groups};baselines={}
    for g in groups:
        ids=[i for i,r in enumerate(rr) if belongs(r,g)];rg=[rr[i] for i in ids];cg=[c[i] for i in ids]
        m,pe=stats(rg,cg,y[ids]);reals[g]=m
        if len(pe):ix=np.random.default_rng(0).integers(len(pe),size=(10000,len(pe)));m['oracle_CI95']=np.quantile(pe[ix].mean(1),[.025,.975]).tolist()
        vr=[r for r,ca in zip(rg,cg) if ca];people=sorted({r['person'] for r in vr});baselines[g]=[]
        for method in ['constant','calibrated']:
            ee=np.array([np.mean([abs(r[method]-r['truth']) for r in vr if r['person']==p]) for p in people])
            baselines[g].append(dict(method=method,videos=len(vr),people=len(people),MAE=float(ee.mean()) if len(ee) else None,CI95=np.quantile(ee[ix].mean(1),[.025,.975]).tolist() if len(ee) else None))
    rng=np.random.default_rng(0);moves=[];maps=[]
    for rep in range(100):
        yy,mapping,strata=permute_blocks(rr,y,rng);moves.append(np.mean([a!=b for a,b in mapping.items()]));maps.append(mapping)
        for g in groups:
            ids=[i for i,r in enumerate(rr) if belongs(r,g)];m,_=stats([rr[i] for i in ids],[c[i] for i in ids],yy[ids]);nulls[g].append(m)
    summaries={}
    for g in groups:
        summaries[g]={}
        for k in ['oracle_MAE','within5','within10']:
            v=[r[k] for r in nulls[g] if r[k] is not None]
            summaries[g][k]=dict(mean=float(np.mean(v)),p5_p95=np.quantile(v,[.05,.95]).tolist()) if v else None
        a=reals[g]
        if a['oracle_MAE'] is not None:
            summaries[g]['randomization_p_oracle']=(1+sum(r['oracle_MAE']<=a['oracle_MAE'] for r in nulls[g]))/101
            summaries[g]['randomization_p_within5']=(1+sum(r['within5']>=a['within5'] for r in nulls[g]))/101
    a=reals['all'];n=summaries['all'];gain=n['oracle_MAE']['mean']-a['oracle_MAE'] if a['oracle_MAE'] is not None else None;gain5=a['within5']-n['within5']['mean']
    passed=bool(gain is not None and gain>=4 and gain5>=.15)
    gate=dict(passed=passed,oracle_gain_bpm=gain,within5_gain_percentage_points=100*gain5,required_oracle_gain_bpm=4,required_within5_gain_percentage_points=15,thresholds_changed=False)
    save(P/'PERMUTATIONS.json',dict(seed=0,replicates=100,strata_person_counts=strata,movement_fraction_mean=float(np.mean(moves)),mappings=maps,metrics=nulls))
    save(P/'RESULTS.json',dict(actual=reals,permutation=summaries,baselines=baselines,gate=gate,strata_person_counts=strata,movement_fraction_mean=float(np.mean(moves)),candidate_counts={str(k):sum(len(x)==k for x in c) for k in range(4)},holdout_evaluations=0))
    save(P/'GATE.json',gate)
    lines=['# BTC: ba đỉnh phổ còn bao nhiêu khả năng chứa PR?','','Oracle dùng nhãn để chọn đỉnh gần nhất: đây là chẩn đoán, không phải MAE của mô hình có thể triển khai. Nhãn chỉ được đọc sau khi khóa tập ứng viên.','','|Nhóm|Video/người; có ứng viên|Oracle MAE|Xáo nhãn MAE trung bình [P5–P95]|Phủ ±5 thật / xáo|Phủ ±10 thật / xáo|','|---|---|---:|---|---|---|']
    for g in groups:
        a=reals[g];n=summaries[g]
        lines.append(f"|{g}|{a['videos']}/{a['people']}; {a['evaluable_videos']}|{a['oracle_MAE']}|{n['oracle_MAE']}|{a['within5']} / {n['within5']}|{a['within10']} / {n['within10']}|")
    lines+=['','Các tỷ lệ ở bảng trên là 0–1 (0,5 nghĩa là 50%). P5–P95 là phân vị của 100 lần xáo, không phải CI95 của một mô hình.','','|Nhóm|Phương pháp thật, trên cùng video có ứng viên|Video/người|MAE [CI95]|','|---|---|---:|---|']
    for g in groups:
        for r in baselines[g]:lines.append(f"|{g}|{r['method']}|{r['videos']}/{r['people']}|{r['MAE']} {r['CI95']}|")
    lines+=['',f"Cổng dư địa: **{'PASS' if passed else 'FAIL'}**. Oracle giảm {gain} bpm so với null (cần ≥4); phủ ±5 tăng {100*gain5} điểm phần trăm (cần ≥15).",'',f"Xáo nguyên khối theo số video/người: {strata}. Tỷ lệ người đổi donor trung bình {float(np.mean(moves)):.4%}. Khối đơn không thể đổi được giữ nguyên; null có giới hạn này.",'','Không dùng kết quả để sửa chọn đỉnh/ngưỡng. Không suy ra nguyên nhân là codec/nhãn hay mọi thông tin nhịp đã mất. Kết luận chỉ áp dụng ba đỉnh của bộ trích/ghép phổ cố định, native BTC dev. SNR/ROI của pipeline này không tương đương canonical oval đã sửa trên public.','',('Bước tiếp theo: preregister bộ đọc nhỏ trước khi fit, vẫn không mở holdout.' if passed else 'Không fit bộ đọc. Chờ disposition V5 rồi đóng băng theo các cổng đã định. Không tuyên bố đã loại trừ mọi bộ chọn phổ khác.'),'','V5 và sao lưu P100 tiếp tục độc lập. Không GPU mới.']
    (P/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    lease=(P/'run.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (P/'STARTED.json').exists(),'Existing attempt: inspect; no automatic restart'
    bindings=frozen();save(P/'PREREGISTRATION.json',dict(utc=now(),bindings=bindings))
    subprocess.run([sys.executable,str(P/'tests.py')],check=True)
    sys.path.insert(0,str(E/'v5_multiview_20261112'));import runtime
    budget=runtime.budget(fresh=True);os.environ['CUDA_VISIBLE_DEVICES']=''
    assert budget['scope_actual_bytes']+budget['remaining_reserved_bytes']+budget['reserve_bytes']+100_000_000<84_000_000_000
    save(P/'STARTED.json',dict(pid=os.getpid(),utc=now(),budget=budget,CPU_only=True,new_allocation_bytes=100_000_000))
    prepare();subprocess.run([sys.executable,str(P/'run.py'),'--extract'],check=True)
    assert frozen()==bindings;score()
    save(P/'COMPLETE.json',dict(utc=now(),gate=read(P/'GATE.json'),candidate_sha256=sha(P/'CANDIDATES.json'),holdout_evaluations=0))

if __name__=='__main__':
    try:
        if '--extract' in sys.argv:extraction()
        else:main()
    except Exception as e:save(P/'ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
