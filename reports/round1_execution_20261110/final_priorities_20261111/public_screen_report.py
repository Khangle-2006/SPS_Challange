import os,json
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import padding_queue as q
P=Path(__file__).resolve().parent;O=P/'classical_screen'
def main():
 q.budget(fresh=True)
 assert not (O/'PUBLIC_SCREEN_REPORT.md').exists()
 rows=[r for p in (O/'predictions').glob('*/*/*.json') for r in q.p.read(p)['rows'] if r['dataset']!='BTC']
 stats=q.p.read(O/'PUBLIC_METRICS.json');pairs=[]
 methods=['POS','CHROM','G','OMIT','POS_schedule','CHROM_schedule','G_schedule']
 for ds,c in sorted({(r['dataset'],r['condition']) for r in rows}):
  rr=[r for r in rows if r['dataset']==ds and r['condition']==c];lookup={m:{(r['recording'],r['start']):r for r in rr if r['method']==m and r['prediction'] is not None and r['truth'] is not None} for m in methods};keys=sorted(set.intersection(*(set(v) for v in lookup.values())))
  people=sorted({lookup['POS'][k]['person'] for k in keys});vals={}
  for m in methods:
   vals[m]=np.array([np.mean([abs(lookup[m][k]['prediction']-lookup[m][k]['truth']) for k in keys if lookup[m][k]['person']==person]) for person in people]);expected=next(x for x in stats if (x['dataset'],x['condition'],x['method'])==(ds,c,m));assert abs(vals[m].mean()-expected['MAE'])<1e-10
  ix=np.random.default_rng(0).integers(len(people),size=(10000,len(people)))
  for m,b in [('POS_schedule','POS'),('CHROM_schedule','CHROM'),('G_schedule','G'),('OMIT','POS')]:
   d=vals[m]-vals[b];pairs.append(dict(dataset=ds,condition=c,method=m,comparator=b,people=len(people),windows=len(keys),delta=float(d.mean()),CI95=np.quantile(d[ix].mean(1),[.025,.975]).tolist()))
 q.save(O/'PUBLIC_PAIRED.json',dict(rows=pairs,bootstrap='10000 paired people seed0',predictions_unchanged=True,public_metric_sha256=q.p.sha(O/'PUBLIC_METRICS.json')))
 lines=['# UBFC/MCD C5′: OMIT và hồi quy lịch codec','', 'Dùng đúng giao cửa sổ của bảy phương pháp trong từng điều kiện. Sai số tuyệt đối được trung bình trong từng người trước, rồi trung bình đều các người. C5′ là proxy ffmpeg, không phải xác nhận tương đương MATLAB. Hồi quy lịch codec đã trượt probe giữ mạch thật 150 bpm: mọi cải thiện dưới đây chỉ là đối chứng, không đủ điều kiện triển khai.','', '| Dataset | Điều kiện | Phương pháp | Người/cửa sổ | MAE [CI95] | Bias |','|---|---|---|---:|---:|---:|']
 for r in stats:lines.append(f"|{r['dataset']}|{r['condition']}|{r['method']}|{r['people']}/{r['windows']}|{r['MAE']:.4f} {r['CI95']}|{r['bias']:+.4f}|")
 lines+=['','## Chênh lệch ghép cặp theo người','','Số âm nghĩa là sai số giảm so với phương pháp ghi ở cột đối chiếu. CI95% bootstrap 10.000 lần, seed0. Đây là so sánh thuật toán trên cùng bản nén, không suy ra riêng cơ chế B-frame.','','| Dataset | Điều kiện | Phương pháp | Đối chiếu | Δ MAE [CI95] |','|---|---|---|---|---:|']
 for r in pairs:lines.append(f"|{r['dataset']}|{r['condition']}|{r['method']}|{r['comparator']}|{r['delta']:+.4f} {r['CI95']}|")
 lines+=['','Chỉ dùng trace RGB trung bình oval36 đã có; chưa phải tất cả ROI. Mẫu số/loại lỗi ở COVERAGE.json. Baseline hằng số và trung vị hiệu chỉnh dành cho BTC nằm trong REPORT.md, không áp nhãn hằng BTC cho dữ liệu public.']
 (O/'PUBLIC_SCREEN_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 q.save(O/'PUBLIC_SCREEN_REPORT_RECEIPT.json',dict(utc=q.now(),code_sha256=q.p.sha(__file__),report_sha256=q.p.sha(O/'PUBLIC_SCREEN_REPORT.md'),paired_sha256=q.p.sha(O/'PUBLIC_PAIRED.json')))
 print('Public paired report complete',len(pairs))
if __name__=='__main__':main()
