"""R1 complete tables, paired old/new public comparison and inventory; never tune."""
from expert_spectrum import *
import csv
mode=selected_mode();assert (OUT/'PUBLIC_FREEZE_COMPLETE.json').exists() and (OUT/'BTC_SCORE_COMPLETE.json').exists();assert not (OUT/'STOP.json').exists()
btc=list(csv.DictReader((OUT/'btc_v4_conditions.csv').open()));pub=list(csv.DictReader((OUT/'z13_public_rerun.csv').open()));oldpub={(r['condition'],r['dataset']):r for r in csv.DictReader((W/'out/z13_public_p4_prefilter.csv').open())}
with np.load(W/'out/z14_p4m_frozen.npz') as z:old={k:z[k].copy() for k in z.files}
with np.load(OUT/'z14_p4m_frozen_rerun.npz') as z:new={k:z[k].copy() for k in z.files}
with np.load(OUT/'PUBLIC_PREDICTIONS_PRIVATE.npz') as z:pr={k:z[k].copy() for k in z.files}
def person_errors(p,y,per):
 ids=sorted(set(per));return ids,np.array([np.abs(p[per==q]-y[per==q]).mean() for q in ids])
def boot(e):
 ix=np.random.default_rng(0).integers(0,len(e),(10000,len(e)));return np.quantile(e[ix].mean(1),[.025,.975]).tolist()
def ci_str(a):return f'[{a[0]:+.3f}; {a[1]:+.3f}]'
def keys(z,mask):return [(str(d),str(r),round(float(s),3)) for d,r,s in zip(z['ds'][mask],z['rec'][mask],z['start'][mask])]
comparisons=[]
for row in pub:
 c,D=row['condition'],row['dataset'];nm=(new['cond']==c)&(new['ds']==D);om=(old['cond']==c)&(old['ds']==D)
 nk,ok=keys(new,nm),keys(old,om);assert len(set(nk))==len(nk) and set(nk)==set(ok),(c,D,'public old/new windows differ')
 lookup={k:i for i,k in zip(np.flatnonzero(om),ok)};ni=np.flatnonzero(nm);oi=np.array([lookup[k] for k in nk]);assert np.array_equal(new['truth'][ni],old['truth'][oi]);assert np.array_equal(new['person'][ni],old['person'][oi])
 native_set=set(zip(pr['native__rec'],np.round(pr['native__start'],3)));matched=np.array([(r,round(float(s),3)) in native_set for r,s in zip(new['rec'][ni],new['start'][ni])]);ni=ni[matched];oi=oi[matched];assert len(ni)==int(row['windows'])
 ids,en=person_errors(new['pred'][ni],new['truth'][ni],new['person'][ni]);_,eo=person_errors(old['pred'][oi],old['truth'][oi],old['person'][oi]);delta=en-eo
 # Full z14 output can include windows excluded by z13's native intersection. Match that set explicitly.
 pd=pr[c+'__ds'];pm=pd==D;native_keys=set(zip(pr['native__rec'],np.round(pr['native__start'],3)))
 pm &= np.array([(r,round(float(s),3)) in native_keys for r,s in zip(pr[c+'__rec'],pr[c+'__start'])])
 per=pr[c+'__person'][pm];truth=pr[c+'__truth'][pm];_,p4=person_errors(pr[c+'__P4'][pm],truth,per);_,p4m=person_errors(pr[c+'__P4m'][pm],truth,per);_,const=person_errors(pr[c+'__constant_P4'][pm],truth,per)
 cov=float(row['coverage']);d4=boot(p4-const);g4=('a' if cov>=.9 else '-')+('b' if d4[0]<=0 else '-')+('c' if d4[1]<0 else '-')
 comparisons.append(dict(condition=c,dataset=D,windows_common=len(ni),people=len(ids),new_p4m_person_mae=float(en.mean()),old_p4m_person_mae=float(eo.mean()),paired_delta=float(delta.mean()),paired_ci95=boot(delta),P4_ci95=boot(p4),P4_gates=g4,P4_const_delta=float((p4-const).mean()),P4_const_ci95=d4))
save(OUT/'PUBLIC_PAIRED_COMPARISONS.json',comparisons)
decision=json.loads((OUT/'MODE.json').read_text());lines=['# R1 — chạy lại sau sửa trạng thái open-rppg','',f'**MODE: {mode}**. Quyết định được lấy nguyên từ z18: `{decision["decision"]}`; không chọn lại dựa trên lượt R1.', '',
'Ba mô hình giữ nguyên trọng số. Trạng thái khởi tạo được sao chép ngay khi tạo Model, khôi phục trước mỗi video/bản ghi/điều kiện. Selfwarm (nếu được chọn) chỉ dùng lần 2 của chính video đó. Không huấn luyện CNN, không mở holdout, không thêm lưới tham số.', '',
'## Kiểm trước chạy lớn','', '| Mô hình | Reset: 2 video | Selfwarm: 2 video | A → B → A (hai MODE) | Thời gian (s) |','|---|---|---|---|---:|']
for n in MODELS:
 g=json.loads((OUT/'gates'/(n+'_PASS.json')).read_text());lines.append(f'| {n} | lệch 0,0 | lệch 0,0 | lệch 0,0 | {g["seconds"]:.1f} |')
lines+=['','## BTC dev','', 'Mốc cũ 8,350 bị loại; reset độc lập 9,130. Cột cũ chỉ để kiểm toán, không phải baseline hợp lệ để quảng bá hiệu năng. File z18 ghi 8,390 ở hàng historical vì mã z18 dùng phổ reset thay 17 hàng heavy; đây không phải chính lượt lịch sử 8,350.','',
'| Điều kiện | Video | V4 mới [CI95] | V4 cũ | Dịch MAE | Hằng số | Hiệu chỉnh native cũ | V3 cùng điều kiện | Δ V4−V3 [CI95] | Δ V4−hằng số [CI95] | Cổng |',
'|---|---:|---|---:|---:|---:|---:|---:|---|---|---|']
for r in btc:
 f=lambda k:float(r[k]);lines.append(f'| {r["condition"]} | {r["videos"]} | {f("v4_MAE"):.3f} [{f("lo"):.3f}; {f("hi"):.3f}] | {f("old_v4_MAE"):.3f} | {f("shift_vs_old"):+.3f} | {f("constant_MAE"):.3f} | {f("calibrated_native_historical_MAE"):.3f} | {f("reader_v3_MAE"):.3f} | {f("d_v3"):+.3f} [{f("d3_lo"):+.3f}; {f("d3_hi"):+.3f}] | {f("d_const"):+.3f} [{f("dc_lo"):+.3f}; {f("dc_hi"):+.3f}] | {r["gates"]} |')
lines+=['','Giữ logic y11: với mỗi điều kiện, fit trên các video native thuộc tập so ghép cặp và fold train, rồi áp phổ điều kiện mà không fit nhãn điều kiện. Mốc hiệu chỉnh là cấu hình native lịch sử áp lại trên đúng video, không phải fit riêng từng điều kiện. Heavy chỉ là 17 video native đã có: tái sử dụng, không suy luận lại.','',
'## Công khai — P4m và P4','', '| Điều kiện | Bộ | Cửa sổ | P4 mới [CI95] / cũ | P4m mới [CI95] / cũ | Δ P4m mới−cũ [CI95] | Hằng số | Cổng P4 / P4m |','|---|---|---:|---|---|---|---:|---|']
for r,d in zip(pub,comparisons):
 o=oldpub[(r['condition'],r['dataset'])];f=lambda k:float(r[k]);lines.append(f'| {r["condition"]} | {r["dataset"]} | {r["windows"]} | {f("P4"):.3f} {ci_str(d["P4_ci95"])} / {float(o["P4"]):.3f} | {f("P4m"):.3f} [{f("p4m_lo"):.3f}; {f("p4m_hi"):.3f}] / {float(o["P4m"]):.3f} | {d["paired_delta"]:+.3f} {ci_str(d["paired_ci95"])} | {f("constant"):.3f} | {d["P4_gates"]} / {r["gates"]} |')
lines+=['','Tất cả so sánh old/new dùng đúng giao cửa sổ và ghép cặp theo người; 10.000 bootstrap, seed 0. Cổng a: coverage ≥0,9; b: CI không chứng minh tệ hơn hằng số; c: CI chứng minh tốt hơn hằng số. Không kết luận cải thiện khi CI chứa 0.','', '## Tham số P4m','', '| Fold | Trọng số mới (CNN, POS, CHROM, experts) | Trọng số cũ |','|---|---|---|']
for tag in ['fold0','fold1','fold2','fold3','fold4','deploy']:
 lines.append(f'| {tag} | {tuple(new["w_"+tag])} | {tuple(old["w_"+tag])} |')
lines+=['','Tham số readout đầy đủ (gamma, sigma, tau, estimator, ref, hằng số train):','', '```text',(OUT/'P4M_DEFINITION_rerun.txt').read_text().strip(),'```','',
'## Giới hạn và thời gian','', '**Việc chọn ba mô hình trong nhóm 17 đã làm trên phổ kiểu cũ; không làm lại lựa chọn/nested selection trong R1.** Mọi kết quả vẫn là dev sau nhiều quyết định nghiên cứu, không phải kiểm định holdout độc lập. Không lấy tham số deploy để báo OOF.','']
for stage in ['gates','inference','score','freeze_public']:
 p=OUT/(stage+'_COMPLETE.json')
 if p.exists():d=json.loads(p.read_text());lines.append(f'- {stage}: {d["seconds"]:.1f} giây.')
lines+=['','CLI/repo: chưa cập nhật ở thời điểm tạo báo cáo này; bước 5 chỉ thực hiện sau khi rà lại các cổng. PR không tự merge.']
(OUT/'RERUN_SUMMARY.md').write_text('\n'.join(lines)+'\n')
save(OUT/'REPORT_COMPLETE.json',dict(mode=mode,btc_rows=len(btc),public_rows=len(pub),holdout_reads=0))
files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('FILE_SHA256.json',) and not p.name.endswith('.lock')}
save(OUT/'FILE_SHA256.json',files)
