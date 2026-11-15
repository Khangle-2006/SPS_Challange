"""Incremental person-paired reporting; no fitting, inference or holdout."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P.parent;N=E/'final_priorities_20261111'
sys.path.insert(0,str(E));from common import read,save,sha,now
from report_metrics import metric,belongs
METHOD='v5_multiview_seed2027'
def main():
    windows=[];completed=[]
    for f in range(5):
        d=P/f'runs/fold{f}/seed2027/joint'
        if not (d/'NATIVE.json').exists():continue
        completed.append(f)
        for name in ['NATIVE.json','C5PRIME.json']:
            if (d/name).exists():windows.extend(r for r in read(d/name) if r.get('dataset','BTC')=='BTC')
    video=[]
    for rid,c in sorted({(r['recording'],r['condition']) for r in windows}):
        rr=[r for r in windows if (r['recording'],r['condition'])==(rid,c) and r['prediction'] is not None]
        if rr:video.append(dict(rr[0],prediction=float(np.median([r['prediction'] for r in rr]))))
    baseline=[r for r in read(E/'C5_BTC_VIDEO.json') if r['method'] in ['constant','calibrated']]+read(N/'padding_evaluation/BTC_VIDEO.json')
    methods=['constant','calibrated','padding_fixed_seed2027',METHOD];table=[]
    for c in sorted({r['condition'] for r in video}):
        for f in completed+['pooled_complete' if len(completed)==5 else 'pooled_partial']:
            rr=[r for r in baseline+video if r['condition']==c and (isinstance(f,str) or r['fold']==f)]
            for g in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
                selected=[r for r in rr if belongs(r,g)];mt=metric(selected,methods)
                keys=set.intersection(*({r['recording'] for r in selected if r['method']==m} for m in methods))
                for a in mt:
                    vv=[r['prediction'] for r in selected if r['method']==a['method'] and r['recording'] in keys];a['video_prediction_SD']=float(np.std(vv,ddof=1)) if len(vv)>1 else None
                    table.append(dict(condition=c,fold=f,group=g,**a))
    coverage=[]
    for c in sorted({r['condition'] for r in windows}):
        rr=[r for r in windows if r['condition']==c];valid=[r for r in rr if r['prediction'] is not None]
        coverage.append(dict(condition=c,all_windows=len(rr),valid_windows=len(valid),all_videos=len({r['recording'] for r in rr}),valid_videos=len({r['recording'] for r in valid}),fraction_ge175=sum(r['prediction']>=175 for r in valid)/len(valid) if valid else None))
    save(P/'BTC_METRICS.json',dict(rows=table,native_folds=completed,coverage=coverage,holdout_evaluations=0));save(P/'BTC_VIDEO.json',video)
    lines=['# V5 — kết quả theo fold','','Chỉ kết quả đủ năm fold mới là OOF đầy đủ. Cùng video, tính MAE từng người rồi lấy trung bình; bootstrap ghép cặp 10.000 lần, seed0. Không mở holdout.','','|Fold|Điều kiện|Nhóm|Phương pháp|Video/người|MAE [CI95]|Bias|Spearman|SD video|Δ hiệu chỉnh [CI95]|','|---|---|---|---|---:|---|---:|---:|---:|---|']
    for r in table:lines.append(f"|{r['fold']}|{r['condition']}|{r['group']}|{r['method']}|{r['videos']}/{r['people']}|{r['MAE']:.4f} {r['CI95']}|{r['bias']:+.4f}|{r['Spearman']}|{r['video_prediction_SD']}|{r['delta_calibrated']:+.4f} {r['delta_calibrated_CI']}|")
    lines+=['','Độ phủ và tỷ lệ ≥175 ở BTC_METRICS.json. P95_person_MAE trong JSON là phân vị 95 của MAE từng người, không phải P95 của tất cả cửa sổ. C5′ là mô phỏng ffmpeg, chưa xác minh MATLAB. Đây là thử nghiệm cuối; không mở thêm sweep sau V5.']
    (P/'BTC_PROGRESS.md').write_text('\n'.join(lines))
    if (P/'TRAIN_COMPLETE.json').exists():
        assert len(completed)==5
        g=next(r for r in table if r['method']==METHOD and r['condition']=='native' and r['fold']=='pooled_complete' and r['group']=='all')
        static=all(read(P/f'runs/fold{f}/seed2027/joint/STATIC.json')['pass_static'] for f in range(5))
        passed=bool(g['MAE']<=10 and g['delta_calibrated']<=-1 and g['delta_calibrated_CI'][1]<0 and static)
        save(P/'BTC_GATE.json',dict(passed=passed,metrics=g,static_pass=static,thresholds_unchanged=True,no_training_after_V5=True))
        subprocess.run([sys.executable,str(P/'public_summary.py')],check=True,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
        chosen='V5 multiview' if passed else 'saved calibrated median'
        (P/'FREEZE.md').write_text(f'# Cấu hình đóng băng sau V5\n\nBTC: {chosen}. Cổng BTC V5: {passed}. Public: padding-fixed. V4 cũ chỉ là ablation, trượt kiểm soát ảnh tĩnh. Không huấn luyện thêm sau V5. Holdout 72 video/41 người vẫn đóng, chỉ đánh giá cấu hình đã chọn một lần ngày26–27/11. C5-prime chưa xác minh MATLAB. Xem BTC_GATE.json, PUBLIC_FINAL.json và BTC_PROGRESS.md; tiếp tục phần báo cáo/đóng gói C–F, không tự nộp bài.\n')
if __name__=='__main__':main()
