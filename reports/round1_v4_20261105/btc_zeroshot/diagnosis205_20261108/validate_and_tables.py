from pathlib import Path
import json,csv,sys
import numpy as np
from collections import defaultdict
O=Path(__file__).resolve().parent;B=O.parent
sys.path.insert(0,str(B))
from summarize_dev import calibration
def rr(name):return list(csv.DictReader((O/name).open(encoding='utf8')))
# Meaningful numerical contracts: source exact cohort and leaked-person prevention.
c=json.loads((O/'COHORT.json').read_text());s=json.loads((O/'SUMMARY.json').read_text())
assert len(c['recordings'])==len(set(c['recordings']))==205
assert sum(r['windows'] for r in s['quartiles'] if r['method']=='POS')==543
assert sum(r['videos'] for r in s['video_quartiles'] if r['method']=='POS')==205
for m in rr('STRATA.csv'):
    if m['dimension']=='all':
        orig=next(x for x in c['metrics'] if x['method']==m['method'] and x['condition']==m['condition'])
        assert abs(float(m['MAE'])-orig['MAE'])<1e-8
top=rr('TOP20.csv');assert len(top)==20 and len({r['recording'] for r in top})==20
fits=rr('CALIBRATION_FITS.csv');meta={r['recording']:r for r in rr('VIDEO_METADATA.csv')}
import ast
for f in fits:
    train=set(ast.literal_eval(f['train_people']));valid={m['person'] for m in meta.values() if int(m['fold'])==int(f['fold'])};assert not train&valid
# Perturb held-fold labels: its fitted a,b and predictions must not change.
rows=[dict(recording=str(i),person=str(i),fold=i%5,prediction=float(i+2),truth=float(2*i+1)) for i in range(20)]
a,fa=calibration(rows);b,fb=calibration([dict(r,truth=r['truth']+1000 if r['fold']==0 else r['truth']) for r in rows])
assert next(f for f in fa if f['fold']==0)==next(f for f in fb if f['fold']==0)
assert np.allclose([r['prediction'] for r in a if r['fold']==0],[r['prediction'] for r in b if r['fold']==0])
(O/'VALIDATION.json').write_text(json.dumps(dict(fixed_cohort=True,baseline_exact_MAE_parity=True,quartile_partition=True,top20_unique=True,person_disjoint_calibration=True,heldfold_label_perturbation_test=True,remote_frozen_POS_parity_windows=543),indent=2))
text=['# Bảng phân tầng BTC — cùng205video,145người','','MAE/bias/P95 đơn vịbpm. P95 ở đây: tính P95 lỗi video trong mỗi người rồi lấy trung bình người; không phải phân vị95 của mọi cửa sổ. nngười trong các tầng không cộng thành145 vì một người có thể có nhiều điều kiện. Native/C5 dùng cùng video, số cửa sổ hợp lệ có thể khác.','', '|Chiều|Nhóm|Phương pháp|Điều kiện|Video/người|MAE|Bias|CI95 MAE|','|---|---|---|---|---:|---:|---:|---|']
for r in rr('STRATA.csv'):
    ci=ast.literal_eval(r['CI95_MAE']);text.append(f"|{r['dimension']}|{r['group']}|{r['method']}|{r['condition']}|{r['videos']}/{r['people']}|{float(r['MAE']):.2f}|{float(r['bias']):+.2f}|[{ci[0]:.2f},{ci[1]:.2f}]|")
(O/'STRATA.md').write_text('\n'.join(text),encoding='utf8')
text=['# 20video có median SNR cao nhất','','SNR không dùng PR để chọn; POS là trung vị cửa sổ hợp lệ trong video. Delta có dấu=POS−PR. Plot đi kèm dùng cửa sổ SNR giữa, không chọn cửa sổ gần nhãn nhất.','', '|Video|Hoạt động trong bảng|PR|POS video|Delta|Median SNR|Phổ|','|---|---|---:|---:|---:|---:|---|']
for r in top:
    rec=r['recording'];text.append(f"|{rec}|{meta[rec]['task']}|{float(r['truth']):.0f}|{float(r['video_POS']):.2f}|{float(r['video_error']):+.2f}|{float(r['video_snr_median']):.3f}|[PNG]({rec}_PSD.png)|")
(O/'TOP20.md').write_text('\n'.join(text),encoding='utf8')
print('VALIDATION PASS')
