"""Read-only reuse of completed experiments with identical-cohort comparators."""
import csv,json,hashlib,warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
P=Path(__file__).resolve().parent;B=P.parent/'round1_bayes_20261108';S=B/'domain_sweep_20261109'
def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(P/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
def belongs(r,g):
 if g=='all':return True
 if g=='Resting':return r['task']=='Resting'
 if g in ['Task','Walking']:return r['task'].lower().startswith('task') if g=='Task' else r['task']=='Walking'
 return r['task']=='Resting' and r['group']==g.split('-')[0]
def metric(rows,methods):
 look={m:{r['recording']:r for r in rows if r['method']==m and r['prediction'] is not None} for m in methods};keys=sorted(set.intersection(*(set(a) for a in look.values())))
 if not keys:return []
 people=sorted({look[methods[0]][k]['person'] for k in keys});arr={};bias={};rho={}
 for m in methods:
  ee=[];bb=[];pp=[];tt=[]
  for person in people:
   rr=[look[m][k] for k in keys if look[m][k]['person']==person];e=np.array([float(r['prediction'])-float(r['truth']) for r in rr]);ee.append(float(abs(e).mean()));bb.append(float(e.mean()));pp.append(float(np.mean([r['prediction'] for r in rr])));tt.append(float(np.mean([float(r['truth']) for r in rr])))
  arr[m]=np.array(ee);bias[m]=float(np.mean(bb));r=float(spearmanr(pp,tt).statistic) if len(pp)>2 and np.std(pp)>0 and np.std(tt)>0 else float('nan');rho[m]=r if np.isfinite(r) else None
 ix=np.random.default_rng(0).integers(len(people),size=(10000,len(people)));ci=lambda a:np.quantile(a[ix].mean(1),[.025,.975]).tolist()
 return [dict(method=m,videos=len(keys),people=len(people),MAE=float(arr[m].mean()),CI95=ci(arr[m]),bias=bias[m],Spearman=rho[m],delta_constant=float((arr[m]-arr['constant']).mean()),delta_constant_CI=ci(arr[m]-arr['constant']),delta_calibrated=float((arr[m]-arr['calibrated']).mean()),delta_calibrated_CI=ci(arr[m]-arr['calibrated'])) for m in methods]
raw=csvread(S/'PBV_STACK_VIDEOS.csv');meta={r['recording']:r for r in raw if r['method']=='constant'}
cal={r['recording']:float(r['prediction']) for r in csvread(B/'VIDEO_PREDICTIONS.csv') if r['method']=='calibrated_median_POS_CHROM_V3' and r['condition']=='native'}
for r in raw:r['prediction']=float(r['prediction'])
rows=raw+[dict(meta[k],method='calibrated',prediction=v) for k,v in cal.items() if k in meta]
methods=['constant','calibrated','POS','CHROM','G','PBV_trainfold','stack5_nested'];groups=['all','Normal-Resting','Abnormal-Resting','Task','Walking']
pbv=[dict(group=g,**r) for g in groups for r in metric([r for r in rows if belongs(r,g)],methods)]
write('A5_PBV_G.json',pbv)
timing=json.loads((S/'TIMING_WINDOWS.json').read_text())['rows'];tim=[]
for g in ['all','Resting']+groups[1:]:
 for method in ['POS','CHROM','UBFC','V3','V4_old_padding']:
  records={r['recording'] for r in timing if r['method']==method and belongs(r,g)}
  chosen={(r['recording'],r['position']):r for r in timing if r['method']==method}
  keys={k for k in records if k in cal and all(chosen[k,pos]['prediction'] is not None for pos in ['first','middle','last'])}
  for pos in ['first','middle','last']:
   a=[r for r in timing if r['recording'] in keys and r['position']==pos and r['method'] in [method,'constant']]
   a += [dict(chosen[k,pos],method='calibrated',prediction=cal[k]) for k in sorted(keys)]
   tim += [dict(group=g,position=pos,target_method=method,**r) for r in metric(a,['constant','calibrated',method])]
write('A4_TIMING.json',tim)
# Original pairs retain the precise first/last paired bootstrap; no new timing model chosen.
pairs=csvread(S/'TIMING_PAIRED.csv');write('A4_PAIRED_EXISTING.json',pairs)
lines=['# CPU A1/A2/A4/A5/A6 — kết quả đã đối chiếu','', 'Dùng lại đầu ra đã lưu; không suy luận/train lại. CI95% bootstrap người10.000 lần seed0. Mọi bảng BTC so đúng giao video; số toàn cohort11,37 được tính lại trong từng nhóm/giao. Đơn vị bpm.','', '## A1: cổng vẫn không đạt','', 'POS multi-ROI mới: MCD13,7281; blind SNR median−3,8202dB. Audit xác định cả trace ROI và khối decoder góp phần. Bản sửa khôi phục canonical old oval/POS/decoder, 4304/4304 dự đoán public khớp chính xác. MCD4,67015 nhưng SNR−0,24956dB: vẫn FAIL điều kiện SNR, không diễn giải Bayes như bằng chứng mất tín hiệu. Không chọn tổ hợp4,03 hậu nghiệm.','', 'Bảng ghép cặp old/new/V4 và phân bố SNR gốc: ../round1_bayes_20261108/public_comparison/fixed_pos_20261109/REPORT.md, METRICS.json; audit: public_comparison/pos_repair_20261110/REPORT.md.','', '## A4: thời điểm nhãn','', 'Nguồn218video; POS/CHROM có162video111người đủ cả3cửa sổ. Không giả vờ đủ218. POS cuối−đầu−0,614 CI[−4,051;2,833]; Task cuối tệ hơn+0,793 CI[−3,037;4,743]. Không có bằng chứng chung để thêm ứng viên nghiêng cuối; không suy ra thời điểm đọc oximeter. Chi tiết MAE/CI/Spearman và hai mốc trong A4_TIMING.json.','', '## A5: PBV và G']
for g in groups:
 a=[r for r in pbv if r['group']==g];lines+=['',f'### {g}','', '| Phương pháp | Video/người | MAE [CI95%] | Δ so trung vị hiệu chỉnh [CI95%] |','|---|---:|---:|---:|']
 for r in a:
  lo,hi=r['CI95'];dl,dh=r['delta_calibrated_CI'];lines.append(f"| {r['method']} | {r['videos']}/{r['people']} | {r['MAE']:.3f} [{lo:.3f};{hi:.3f}] | {r['delta_calibrated']:+.3f} [{dl:+.3f};{dh:+.3f}] |")
lines+=['','constant = trung vị PR train-fold; calibrated = trung vị hiệu chỉnh (mốc toàn bộ11,37); stack5_nested là kết quả cũ, chưa phải stacking D với mô hình mới.','', '## A6: bitrate/luma','', 'Đã hoàn tất66hàng phân tầng có POS cũ, hằng số, hiệu chỉnh trên cùng video. Nguồn ../round1_bayes_20261108/revision_20261109/COMPRESSION_REPORT.md và COMPRESSION_STRATA.csv. Spearman cấp video native |lỗi|–bpp: POS−0,129; hiệu chỉnh+0,112. Liên hệ yếu, không khẳng định nguyên nhân. Bpp và luma là metadata quan sát, không phải lux hay chất lượng chuẩn hóa giữa codec.','', 'A2 trích nguyên văn và1b trong DECISIONS.md. A3/B/C/D chưa hoàn tất; không tuyên bố đã thử hết hoặc đạt cổng.']
(P/'CPU_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
write('CPU_REUSE_RECEIPT.json',dict(holdout_evaluations=0,inputs={str(p):sha(p) for p in [S/'PBV_STACK_VIDEOS.csv',B/'VIDEO_PREDICTIONS.csv',S/'TIMING_WINDOWS.json',S/'TIMING_PAIRED.csv',B/'revision_20261109/COMPRESSION_STRATA.csv']},new_inference=False,new_fit=False))
print('A4,A5 tables including both comparators complete; A1,A2,A6 evidence linked')
