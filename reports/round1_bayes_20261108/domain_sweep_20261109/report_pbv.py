import csv,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
rr=list(csv.DictReader((P/'PBV_STACK_VIDEOS.csv').open()))
for r in rr:r.update(prediction=float(r['prediction']),truth=float(r['truth']))
old=list(csv.DictReader((P.parent/'VIDEO_PREDICTIONS.csv').open()))
meta={r['recording']:r for r in rr if r['method']=='constant'}
for r in old:
 if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3' and r['recording'] in meta:
  rr.append(dict(meta[r['recording']],method='calibrated_existing',prediction=float(r['prediction'])))
methods=['constant','POS','CHROM','G','PBV_trainfold','stack5_nested','calibrated_existing'];lookup={m:{r['recording']:r for r in rr if r['method']==m} for m in methods}
common=set.intersection(*map(lambda d:set(d),lookup.values()));results=[]
def group(r,g):return g=='all' or (g=='Abnormal' and r['group']=='Abnormal') or (g=='Task' and r['task'].lower().startswith('task')) or (g=='Dark' and r['skin'].lower()=='dark') or (g=='Normal-Resting' and r['group']=='Normal' and r['task']=='Resting') or (g=='Abnormal-Resting' and r['group']=='Abnormal' and r['task']=='Resting')
for g in ['all','Abnormal','Task','Dark','Normal-Resting','Abnormal-Resting']:
 keys=sorted(k for k in common if group(meta[k],g));people=sorted({meta[k]['person'] for k in keys})
 for m in methods:
  err=[];bias=[];delta=[];delta_cal=[]
  for person in people:
   ks=[k for k in keys if meta[k]['person']==person]
   e=np.mean([abs(lookup[m][k]['prediction']-meta[k]['truth']) for k in ks]);b=np.mean([lookup[m][k]['prediction']-meta[k]['truth'] for k in ks]);c=np.mean([abs(lookup['constant'][k]['prediction']-meta[k]['truth']) for k in ks]);cal=np.mean([abs(lookup['calibrated_existing'][k]['prediction']-meta[k]['truth']) for k in ks]);err.append(e);bias.append(b);delta.append(e-c);delta_cal.append(e-cal)
  ix=np.random.default_rng(0).integers(len(people),size=(10000,len(people)))
  ci=lambda a:np.quantile(np.array(a)[ix].mean(1),[.025,.975]).tolist()
  results.append(dict(group=g,method=m,videos=len(keys),people=len(people),MAE=float(np.mean(err)),MAE_CI=ci(err),bias=float(np.mean(bias)),delta_constant=float(np.mean(delta)),delta_constant_CI=ci(delta),delta_calibrated=float(np.mean(delta_cal)),delta_calibrated_CI=ci(delta_cal)))
with (P/'PBV_STACK_METRICS.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
lines=['# PBV, G và stacking trên BTC native','', 'Đã hoàn tất CPU trên 218 video / 158 người dev, với năm fold theo người. Holdout chưa mở. PBV học chữ ký màu từ cửa sổ SNR cao của train-fold; stacking dùng đúng 5 hệ số tính cả intercept và cross-fit chữ ký PBV trong từng outer fold. Đây là biến thể ước lượng chữ ký PBV của đội, không phải kết quả sao chép nguyên bài PBV.','', 'MAE là sai số tuyệt đối (bpm): trung bình trong từng người, rồi trung bình giữa người. Δ âm là tốt hơn mốc hằng số. CI95% lấy bootstrap ghép cặp 10.000 lần theo người, seed 0, giữ các dự đoán OOF đã fit. CI chứa 0 nghĩa là chưa có bằng chứng chắc chắn về cải thiện trên so sánh này. Bias dương là dự đoán cao hơn nhãn.','', 'Các phương pháp cùng nhóm được so trên đúng cùng video; POS/CHROM/G/PBV dùng cùng các cửa sổ RGB hợp lệ của phép thử này. Mốc hiệu chỉnh cũ dùng dự đoán video đã lưu; không refit. Trường windows và bảng loại trừ được giữ để kiểm tra độ phủ. Không dùng kết quả này để đổi threshold SNR, chữ ký, clipping hoặc số tham số.']
for g in ['all','Abnormal','Task','Dark','Normal-Resting','Abnormal-Resting']:
 a=[r for r in results if r['group']==g];lines += ['',f"## {g}: {a[0]['videos']} video / {a[0]['people']} người",'', '| Phương pháp | MAE | Bias | Δ so hằng số [CI95%] |','|---|---:|---:|---|']
 for r in a:
  lo,hi=r['delta_constant_CI'];lines.append(f"| {r['method']} | {r['MAE']:.3f} | {r['bias']:+.3f} | {r['delta_constant']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
lines += ['', 'Tên gọi: G = chỉ kênh xanh lá; PBV_trainfold = chữ ký màu học từ train-fold; stack5_nested = intercept + POS + CHROM + PBV + SNR; calibrated_existing = hiệu chỉnh tuyến tính trước đó của median(POS, CHROM, V3). Báo cáo này chưa có checkpoint sweep khớp miền mới, cũng chưa có C5′ proxy toàn cohort. Không kết luận trần sai số BTC hoặc nguyên nhân nén/nhãn từ bảng này.']
(P/'PBV_STACK_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(P/'PBV_STACK_METRICS.json').write_text(json.dumps(dict(results=results,common_videos=len(common),prediction_sha256=hashlib.sha256((P/'PBV_STACK_VIDEOS.csv').read_bytes()).hexdigest(),holdout_evaluations=0),indent=2,allow_nan=False))
print(json.dumps([r for r in results if r['group']=='all'],indent=2))
