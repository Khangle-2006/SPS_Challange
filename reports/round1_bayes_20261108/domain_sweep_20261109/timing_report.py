"""Paired person-level analysis of fixed positions, preserving missing windows."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
P=Path(__file__).resolve().parent
a=json.loads((P/'TIMING_WINDOWS.json').read_text());rows=a['rows']
def selected(r,g):
 if g=='all':return True
 if g=='Task':return str(r['task']).lower().startswith('task')
 if g=='Walking':return str(r['task']).lower()=='walking'
 return r['group']==g.split('-')[0] and str(r['task']).lower()=='resting'
result=[];comparisons=[]
for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
 for method in sorted({r['method'] for r in rows}):
  records=sorted({r['recording'] for r in rows if selected(r,group)})
  lookup={(r['recording'],r['position']):r for r in rows if r['method']==method}
  good=[v for v in records if all(lookup[v,p]['prediction'] is not None for p in ['first','middle','last'])]
  people=sorted({lookup[v,'first']['person'] for v in good});errors={}
  for position in ['first','middle','last']:
   err=[];pred=[];truth=[];bias=[];const=[]
   for person in people:
    rr=[lookup[v,position] for v in good if lookup[v,position]['person']==person]
    err.append(np.mean([abs(r['prediction']-r['truth']) for r in rr]));pred.append(np.mean([r['prediction'] for r in rr]));truth.append(np.mean([r['truth'] for r in rr]));bias.append(np.mean([r['prediction']-r['truth'] for r in rr]))
    cv=[r for r in rows if r['method']=='constant' and r['position']==position and r['person']==person and r['recording'] in good]
    const.append(np.mean([abs(r['prediction']-r['truth']) for r in cv]))
   errors[position]=np.asarray(err)
   rho=float(spearmanr(pred,truth).statistic) if len(people)>2 and np.std(pred)>0 and np.std(truth)>0 else None
   result.append(dict(group=group,method=method,position=position,n_videos=len(good),n_people=len(people),candidate_videos=len(records),MAE=float(np.mean(err)) if err else None,bias=float(np.mean(bias)) if bias else None,person_Spearman=rho if rho is not None and np.isfinite(rho) else None,constant_same_videos_MAE=float(np.mean(const)) if const else None))
  if people:
   delta=errors['last']-errors['first'];rng=np.random.default_rng(0);boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
   comparisons.append(dict(group=group,method=method,n_videos=len(good),n_people=len(people),last_minus_first_MAE=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist()))
for name,rr in [('TIMING_METRICS.csv',result),('TIMING_PAIRED.csv',comparisons)]:
 with (P/name).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
text=['# Cửa sổ đầu, giữa và cuối — BTC dev','', 'Dùng lại dự đoán đã lưu; không chạy suy luận mới. Nhãn PR của mỗi video vẫn là một số duy nhất. Mỗi phương pháp chỉ so trên giao video có đủ cả ba vị trí, rồi lấy trung bình sai số trong từng người trước khi lấy trung bình giữa người. Hằng số được tính trên đúng giao video của từng hàng.','', 'Các vị trí được chọn từ danh sách cửa sổ 10 giây không chồng trước khi lọc lỗi: đầu tiên, chỉ số floor(n/2), cuối cùng. Video có dưới ba cửa sổ không được thay thế/đệm. Vị trí giữa có thể khác điểm giữa thời gian liên tục. Đây không phải kiểm nghiệm thời điểm thiết bị đo PR.','', '| Nhóm | Phương pháp | Video/người | MAE đầu | MAE giữa | MAE cuối | Hằng số cùng giao | Δ cuối−đầu [CI95%] |','|---|---|---:|---:|---:|---:|---:|---|']
for c in comparisons:
 r=[z for z in result if z['group']==c['group'] and z['method']==c['method']];m={z['position']:z for z in r};lo,hi=c['CI95']
 text.append(f"| {c['group']} | {c['method']} | {c['n_videos']}/{c['n_people']} | {m['first']['MAE']:.3f} | {m['middle']['MAE']:.3f} | {m['last']['MAE']:.3f} | {m['first']['constant_same_videos_MAE']:.3f} | {c['last_minus_first_MAE']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
text+=['','Đơn vị MAE và Δ là bpm. Δ âm nghĩa là cuối có sai số thấp hơn đầu. CI bootstrap ghép cặp 10.000 lần, seed 0, lấy mẫu người. Các nhóm/phương pháp là nhiều phép so sánh thăm dò, chưa hiệu chỉnh multiplicity; không dùng để chọn lại đoạn thời gian rồi báo chính tập này là kiểm nghiệm độc lập. Spearman cấp người, bias có dấu và mẫu số đầy đủ nằm trong TIMING_METRICS.csv.','', 'V4 ở đây là 15 checkpoint padding gốc đã chạy xong; không phải sweep mới. Không mở holdout.']
(P/'TIMING_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
(P/'TIMING_RECEIPT.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256((P/'TIMING_WINDOWS.json').read_bytes()).hexdigest(),rows=len(rows),videos=len(a['cohort']),metrics=result,paired=comparisons,holdout_evaluations=0),indent=2,allow_nan=False))
print(json.dumps([x for x in comparisons if x['group']=='all'],indent=2))
