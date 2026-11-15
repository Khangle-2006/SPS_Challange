"""Strict five-fold, one-seed comparison; no old 30-run root assumptions."""
import os,sys,time,traceback,fcntl,argparse
from pathlib import Path
from datetime import datetime,timezone,timedelta
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
P=Path(__file__).resolve().parent;V=P.parent.parent/'round1_v4_20261105'
sys.path.insert(0,str(V/'aggregation_20261107'))
import aggregate as a
OUT=P/'public_padding';OUT.mkdir(exist_ok=True)
BRANCHES=('padding_fixed','v4_repaired','v3_joint','POS','CHROM')
def run(f):return (V/'v4_padding_pilot' if f==0 else P/'padding_fixed')/f'fold{f}/seed2027/joint'
def inputs(folds):
 frozen=a.read(a.FROZEN);new=[];receipts=[]
 for f in folds:
  d=run(f);rr=a.load_run(d,frozen)
  for r in rr:r['branch']='padding_fixed'
  new+=rr
  static=a.read(d/'STATIC.json');assert static['pass_static']
  receipts.append(dict(fold=f,checkpoint_sha256=a.sha(d/'final.pt'),static_sha256=a.sha(d/'STATIC.json')))
 rows=list(new);wanted={(r['condition'],r['fold'],r['seed'],a.key(r)):r for r in new}
 for f in folds:rows+=a.load_run(V/f'v4_repaired/fold{f}/seed2027/joint',frozen)
 for branch in ['v3_joint','POS','CHROM']:
  old=[r for r in frozen if r['branch']==branch and r['seed']==2027 and r['fold'] in folds];assert {(r['condition'],r['fold'],r['seed'],a.key(r)) for r in old}==set(wanted)
  rows+=old
 for r in rows:
  v=wanted[r['condition'],r['fold'],r['seed'],a.key(r)]
  assert (v['reference_eligible'],v['accepted'])==(r['reference_eligible'],r['accepted']),'No silent failure/support intersection'
  if r['reference_eligible']:assert r['reference_hr_bpm']==v['reference_hr_bpm']
 return rows,receipts
def aggregate():
 assert not (OUT/'FINAL.json').exists(),'Final exists; inspect rather than overwrite'
 rows,receipts=inputs(list(range(5)));ps=a.person_seed_rows(rows);pm=a.person_means(ps,required_seeds=(2027,));result=a.summary(pm,branches=BRANCHES)
 result['paired_padding_minus']=result.pop('paired_V4_minus')
 result.update(utc=datetime.now(timezone.utc).isoformat(),complete_folds=5,seeds=[2027],seed_scope='Five-fold one-seed comparison, not a three-seed replication',checkpoint_receipts=receipts,frozen_windows_sha256=a.sha(a.FROZEN),strict_reference_support_failure_parity=True,holdout_evaluations=0,automatic_BTC_promotion=False)
 delta=result['paired_padding_minus']['v3_joint'];result['public_gate_components']=dict(S_CI_improves=delta['S']['CI95'][1]<0,UBFC_clean_delta_within_half_bpm=delta['UBFC clean']['MAE']['value']<=.5,static_pass=True,scope='Public development, same seed2027; BTC decision gate remains separate')
 a.table_csv(OUT/'person_metrics.csv',ps);a.save(OUT/'FINAL.json',result)
 lines=['# Padding-fixed: đủ năm fold, một seed 2027','', 'Fold 0 pilot cũ được ghép với bốn fold mới. So cùng seed2027 của V3/V4 và cùng cửa sổ POS/CHROM; không so với trung bình ba seed như thể là cùng thí nghiệm. Mọi hash tham chiếu thô/canonical, eligibility và lỗi phải khớp; không âm thầm lấy giao cửa sổ.', '', '| Phương pháp | Điều kiện | MAE [CI95] | P95 [CI95] | Bias [CI95] | ≥179 [CI95] | Coverage [CI95] |','|---|---|---:|---:|---:|---:|---:|']
 fmt=lambda r:f"{r['value']:.4f} [{r['CI95'][0]:.4f}, {r['CI95'][1]:.4f}]"
 for b in BRANCHES:
  for c in a.CONDITIONS:lines.append('| '+' | '.join([b,c]+[fmt(result['branches'][b][c][m]) for m in a.METRICS])+' |')
 lines+=['', 'P95 là trung bình P95 sai số cửa sổ trong từng người, không phải P95 gộp. Tỷ lệ ≥179 và coverage dùng đơn vị 0–1. Bootstrap ghép cặp theo người 10.000 lần seed0. S = trung bình MAE người của UBFC sạch, UBFC C5 và MCD native. S nén chỉ là mô tả cho UBFC C5/MCD-C5.', '', '| Phương pháp | S [CI95] | S nén [CI95] |','|---|---:|---:|']
 for b in BRANCHES:lines.append(f"|{b}|{fmt(result['branches'][b]['S'])}|{fmt(result['branches'][b]['S_compressed'])}|")
 lines += ['', '| So với | Điều kiện | Δ MAE [CI95] |','|---|---|---:|']
 for b in BRANCHES[1:]:
  for c in a.CONDITIONS:lines.append(f"|{b}|{c}|{fmt(result['paired_padding_minus'][b][c]['MAE'])}|")
 lines+=['', 'Không tự đổi ứng viên BTC từ điểm S của dữ liệu công khai. Xem bảng BTC OOF và cổng native riêng; holdout vẫn đóng.']
 (OUT/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main(check=False):
 if check:
  rows,r=inputs([0]);a.save(P/'PUBLIC_AGGREGATION_CHECK.json',dict(passed=True,fold=0,rows=len(rows),receipts=r,strict_hash_support_parity=True));return
 lease=(OUT/'worker.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'LAUNCH.json').exists()
 a.save(OUT/'LAUNCH.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),code_sha256=a.sha(__file__),aggregator_source_sha256=a.sha(V/'aggregation_20261107/aggregate.py'),loader_source_sha256=a.sha(V/'controls_20261107/common.py')))
 end=datetime.now(timezone.utc)+timedelta(hours=12)
 while datetime.now(timezone.utc)<end:
  ready=[f for f in range(5) if (run(f)/'status.json').exists() and a.read(run(f)/'status.json')['state']=='complete']
  a.save(OUT/'STATUS.json',dict(utc=datetime.now(timezone.utc).isoformat(),ready_folds=ready,pid=os.getpid()))
  if len(ready)==5:aggregate();return
  if (P/'ERROR.json').exists():raise RuntimeError('Training error, preserve partial and inspect')
  time.sleep(30)
 raise RuntimeError('12hour bound reached; no automatic restart')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 try:main(args.check)
 except Exception as e:a.save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
