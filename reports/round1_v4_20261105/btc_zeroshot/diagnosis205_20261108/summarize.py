"""Local, person-first descriptive analysis; frozen 205-video cohort."""
from pathlib import Path
import json,csv,sys
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import spearmanr
O=Path(__file__).resolve().parent;B=O.parent
sys.path.insert(0,str(B))
from summarize_dev import boot,calibration
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(name,rows):
    if not rows:return
    with (O/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,x):(O/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def num(x):return 'không xác định' if x is None else f'{x:.3f}'
def rho(rows):
    per=defaultdict(list)
    for r in rows:per[r['person']].append((r['prediction'],r['truth']))
    a=np.array([np.mean(v,axis=0) for v in per.values()])
    if len(a)<3 or any(np.ptp(a[:,i])==0 for i in [0,1]):return None
    return float(spearmanr(a[:,0],a[:,1]).statistic)
def metric(rows):
    z=boot(rows);return dict(videos=len(rows),people=z['people'],MAE=z['MAE'],bias=z['bias'],P95=z['P95'],CI95_MAE=z['CI95'][0],spearman=rho(rows))
def strata(m):
    pr=float(m['truth']);fps=float(m['fps']);task=m['task'].lower()
    return dict(all='all',group=m['group'],activity='Resting' if task=='resting' else 'Walking' if task=='walking' else 'Task' if task.startswith('task') else 'Other',resolution=m['resolution'],codec=m['codec'],HDR='HDR' if m['hdr']=='True' else 'SDR',rotation='rotated' if float(m['rotation'])%360 else 'none',fps='<25' if fps<25 else '25–<35' if fps<35 else '35–<100' if fps<100 else '>=100',lighting=m['lighting'],skin=m['skin'],PR='<60' if pr<60 else '60–<80' if pr<80 else '80–<100' if pr<100 else '100–<120' if pr<120 else '>=120')
def main():
    cohort=json.loads((O/'COHORT.json').read_text());keys=set(cohort['recordings'])
    meta={m['recording']:m for m in readcsv(O/'VIDEO_METADATA.csv')};assert set(meta)==keys
    source=readcsv(B/'VIDEO_PREDICTIONS_status_20261108.csv');methods=['constant','POS','CHROM','ubfc_only_median15','v3_joint_median15']
    records={}
    for method in methods:
        for c in ['native','c5']:
            rr=[dict(recording=r['recording'],person=r['person'],fold=int(r['fold']),truth=float(r['truth']),prediction=float(r['prediction'])) for r in source if r['scheme']=='10s' and r['condition']==c and r['method']==method and r['recording'] in keys]
            assert len(rr)==205 and {r['recording'] for r in rr}==keys
            records[method,c]={r['recording']:r for r in rr}
    groups=defaultdict(list)
    for r,m in meta.items():
        for dim,val in strata(m).items():groups[dim,val].append(r)
    strat=[]
    for (dim,val),kk in groups.items():
        for method in methods:
            for c in ['native','c5']:strat.append(dict(dimension=dim,group=val,method=method,condition=c,**metric([records[method,c][r] for r in kk])))
    write('STRATA.csv',strat)
    wins=readcsv(O/'SNR_WINDOWS.csv')
    for w in wins:
        for k in ['start','truth','prediction','snr_ratio','snr_db','peak_hz']:w[k]=float(w[k])
    edges=np.quantile([w['snr_ratio'] for w in wins],[.25,.5,.75]);qt=[]
    for qi in range(4):
        ww=[w for w in wins if np.searchsorted(edges,w['snr_ratio'],side='left')==qi];byrec=defaultdict(list)
        for w in ww:byrec[w['recording']].append(w)
        for method in ['POS','constant']:
            rr=[dict(records[method,'native'][r],prediction=float(np.median([w['prediction'] for w in ws])) if method=='POS' else records[method,'native'][r]['prediction']) for r,ws in byrec.items()]
            qt.append(dict(quartile=qi+1,method=method,windows=len(ww),snr_min=min(w['snr_ratio'] for w in ww),snr_max=max(w['snr_ratio'] for w in ww),**metric(rr)))
    write('SNR_QUARTILES_WINDOW.csv',qt)
    medsnr={r:float(np.median([w['snr_ratio'] for w in wins if w['recording']==r])) for r in keys};vedges=np.quantile(list(medsnr.values()),[.25,.5,.75]);vq=[]
    for qi in range(4):
        kk=[r for r in keys if np.searchsorted(vedges,medsnr[r],side='left')==qi]
        for method in ['POS','constant']:vq.append(dict(quartile=qi+1,method=method,**metric([records[method,'native'][r] for r in kk])))
    write('SNR_QUARTILES_VIDEO.csv',vq)
    snrgroups=[]
    for (dim,val),kk in groups.items():
        ww=[w for w in wins if w['recording'] in kk];snrgroups.append(dict(dimension=dim,group=val,videos=len(kk),people=len({meta[r]['person'] for r in kk}),windows=len(ww),median_video_snr=float(np.median([medsnr[r] for r in kk])),fraction_windows_Q1=float(np.mean([w['snr_ratio']<=edges[0] for w in ww]))))
    write('SNR_BY_GROUP.csv',snrgroups)
    repeated=[];pairs=[];prpersons=defaultdict(list)
    for rec,m in meta.items():
        if m['task'].lower()=='resting':prpersons[m['person']].append(rec)
    for person,kk in prpersons.items():
        if len(kk)<2:continue
        a=[float(meta[r]['truth']) for r in kk];b=[records['POS','native'][r]['prediction'] for r in kk]
        repeated.append(dict(person=person,videos=len(kk),recordings=';'.join(kk),PR_values=a,POS_values=b,PR_sd=float(np.std(a)),POS_sd=float(np.std(b)),PR_range=float(np.ptp(a)),POS_range=float(np.ptp(b))))
        for i in range(len(kk)):
            for j in range(i+1,len(kk)):pairs.append(dict(person=person,video1=kk[i],video2=kk[j],PR_delta=a[j]-a[i],POS_delta=b[j]-b[i]))
    write('REPEATED_RESTING.csv',repeated);write('REPEATED_RESTING_PAIRS.csv',pairs)
    fits=[];calmetrics=[];calrows=[]
    for cond in ['native','c5']:
        ensemble=[dict(records['POS',cond][r],prediction=float(np.median([records[m,cond][r]['prediction'] for m in ['POS','CHROM','v3_joint_median15']]))) for r in sorted(keys)]
        for label,rr in [('POS',list(records['POS',cond].values())),('median_POS_CHROM_V3',ensemble)]:
            corrected,ff=calibration(rr);fits.extend(dict(method=label,condition=cond,**f) for f in ff)
            for kind,rows in [('raw',rr),('calibrated',corrected)]:
                base=list(records['constant',cond].values());delta=boot(rows,base)
                calmetrics.append(dict(method=label,kind=kind,condition=cond,**metric(rows),delta_constant=delta['MAE'],delta_CI95=delta['CI95'][0]))
                calrows.extend(dict(method=label,kind=kind,condition=cond,**r) for r in rows)
    write('CALIBRATION_METRICS.csv',calmetrics);write('CALIBRATION_FITS.csv',fits);write('CALIBRATION_VIDEO.csv',calrows)
    cw=readcsv(O/'CLASSICAL_WINDOWS.csv');clusters=[]
    for method in ['POS','CHROM']:
        for cond in ['native','c5']:
            rr=[r for r in cw if r['method']==method and r['condition']==cond];good=[r for r in rr if r['prediction']]
            near=sum(abs(float(r['prediction'])-150)<=2 for r in good)
            clusters.append(dict(method=method,condition=cond,windows_valid=len(good),windows_total=len(rr),near150_count=near,near150_fraction=near/len(good),videos=len({r['recording'] for r in good}),people=len({r['person'] for r in good})))
    write('C5_150_CLUSTER.csv',clusters)
    if (O/'QC_VIDEO.csv').exists():
        qr=readcsv(O/'QC_VIDEO.csv');qgroups=[]
        for (dim,val),kk in groups.items():
            for cond in ['native','c5']:
                rr=[r for r in qr if r['condition']==cond and r['recording'] in kk]
                if not rr:continue
                qgroups.append(dict(dimension=dim,group=val,condition=cond,videos=len(rr),people=len({r['person'] for r in rr}),frame_weighted_face_fraction=sum(float(r['valid_frames']) for r in rr)/sum(float(r['frames']) for r in rr),mean_video_face_fraction=float(np.mean([float(r['face_fraction']) for r in rr])),**{k:float(np.mean([float(r[k]) for r in rr])) for k in ['box_width_mean','box_height_mean','box_area_mean','luma_mean','saturated_fraction','dark_fraction']}))
        write('QC_GROUPS.csv',qgroups)
    summary=dict(n_videos=205,n_people=145,n_windows=len(wins),quartile_edges=edges.tolist(),quartiles=qt,video_quartiles=vq,resting_repeated_people=len(repeated),resting_repeated_videos=sum(r['videos'] for r in repeated),calibration=calmetrics,clusters=clusters,holdout_evaluations=0,fine_tune='DEFERRED_BY_USER_PENDING_CAUSE_DIAGNOSIS',V4='pending_complete_predictions_on_fixed205',padding='one checkpoint only; extended205 inference pending')
    save('SUMMARY.json',summary)
    txt=['# BTC: chẩn đoán trên cùng 205 video / 145 người','','Báo cáo thăm dò dev, không dùng holdout, không chọn ứng viên và không fine-tune. Số thấp hơn là tốt hơn đối với MAE. Bias dương nghĩa dự đoán cao hơn PR. Mỗi người có trọng số bằng nhau.','',
         '## 1. Đỉnh phổ mạnh có đi kèm dự đoán tốt hơn?','','SNR ở đây đo mức tập trung công suất quanh đỉnh POS, không biết đỉnh đó là mạch hay nhiễu. Q1 thấp nhất, Q4 cao nhất; chia theo cửa sổ, lấy trung vị những cửa sổ thuộc nhóm trong mỗi video rồi trung bình sai số theo người. Cùng người/video có thể xuất hiện ở nhiều nhóm. Spearman gần1 nghĩa xếp hạng dự đoán đi cùng PR, gần0 là ít quan hệ đơn điệu.','',
         '|Nhóm|Phương pháp|Cửa sổ|Video/người|MAE bpm|CI95 MAE|Spearman người|','|---|---|---:|---:|---:|---|---:|']
    for r in qt:txt.append(f"|Q{r['quartile']}|{r['method']}|{r['windows']}|{r['videos']}/{r['people']}|{num(r['MAE'])}|{r['CI95_MAE']}|{num(r['spearman'])}|")
    txt+=['','Không tự gán nguyên nhân nhãn từ SNR cao. Chuyển động/ánh sáng/codec cũng tạo đỉnh hẹp; giới hạn dải PR làm Spearman giảm. SNR_QUARTILES_VIDEO.csv là kiểm tra độ nhạy theo median SNR cả video.','',
          '## 2. Resting so với Task/Walking','','|Nhóm|Phương pháp|Điều kiện|Video/người|MAE|Bias|','|---|---|---|---:|---:|---:|']
    for r in strat:
        if r['dimension']=='activity':txt.append(f"|{r['group']}|{r['method']}|{r['condition']}|{r['videos']}/{r['people']}|{num(r['MAE'])}|{num(r['bias'])}|")
    txt+=['','Tất cả phân tầng (nhóm, hoạt động, độ phân giải, codec, HDR, xoay, FPS, ánh sáng, da, PR) có trong STRATA.csv; cùng mẫu cho mọi phương pháp và constant trong từng ô. FPS là cadence PTS, không xác nhận FPS vật lý cảm biến.','',
          '## 3. Kiểm nhãn trực tiếp','',f'Có {len(repeated)} người có nhiều video được bảng BTC ghi Resting trong205video. Không tự coi Task là Resting. Các số PR/POS, độ lệch chuẩn, range và chênh từng cặp ở REPEATED_RESTING.csv và REPEATED_RESTING_PAIRS.csv (chỉ tạo nếu có dòng). Không có cặp thì kiểm tra này không có dữ liệu, không được nói đã chứng minh nhãn ổn/hỏng.','',
          'TOP20.csv và các *_PSD.png:20video median SNR cao nhất, không chọn theo sai số. Mỗi plot dùng cửa sổ SNR giữa của video; bảng ghi cả dự đoán median của video. Đường PR chỉ là scalar tham chiếu chưa rõ timing; không phải nhãn đồng bộ từng cửa sổ. Xem DECISIONS.md cho nguồn chính thức và giới hạn.','',
          '## 4. Hiệu chỉnh dự phòng','','Fit a·HR+b trên4fold, chấm fold5 theo người; xoay đủ5fold. Không ép a<1, không dùng holdout. Đó là 5fold group-OOF, không phải train lại từng leave-one-person-out. Median ba phương pháp lấy ở cấp video.','',
          '|Đầu vào|Loại|Điều kiện|MAE|Delta so constant|CI95 delta|','|---|---|---|---:|---:|---|']
    for r in calmetrics:txt.append(f"|{r['method']}|{r['kind']}|{r['condition']}|{num(r['MAE'])}|{num(r['delta_constant'])}|{r['delta_CI95']}|")
    txt+=['','a,b từngfold và người train ở CALIBRATION_FITS.csv. Delta âm là tốt hơn constant; CI chứa0 nghĩa chưa phân biệt được. Hiệu chỉnh least-squares nhắm MSE, không phải tối ưu MAE trực tiếp; cải thiện không chứng minh lỗi nhãn.','',
          '## 5. Cụm150bpm','','|Phương pháp|Điều kiện|Trong148–152bpm / cửa sổ hợp lệ|Tỷ lệ|','|---|---|---:|---:|']
    for r in clusters:txt.append(f"|{r['method']}|{r['condition']}|{r['near150_count']}/{r['windows_valid']}|{100*r['near150_fraction']:.2f}%|")
    txt+=['','## 6. QC và những phần chưa xong','','QC_VIDEO.csv / QC_GROUPS.csv: tỉ lệ crop hợp lệ, khung bao oval trên ảnh gốc đã xoay, luma và clipping proxy sau decode/tone-map. Nền mask đen bị loại bằng điều kiện có channel>0; đây không phải skin mask chính xác. Montage chỉ những video xoay/HDR thuộc205dev; không mở28file toàn cohort nếu có holdout.','',
          'V4 đầy đủ15checkpoint trên205video chưa xong; không trộn median12 với median15. Padding-fixed chỉ có1checkpoint, báo riêng khi suy luận205hoàn tất. GOP/điều kiện tổng hợp vẫn là chẩn đoán riêng, chưa gán150bpm cho GOP. No fine-tune / holdout0.']
    (O/'REPORT.md').write_text('\n'.join(txt)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
