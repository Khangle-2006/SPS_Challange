"""Independent numeric, missingness and paired detector-ROI effect audit."""
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_spectral_ensemble import independent_psd,audit_report
HERE=Path(__file__).resolve().parent;OUT=HERE/'reports/end_to_end_detection_20261019';BASE=OUT/'results/end_to_end_detection_20261019'
SEEDS=[2027,2028,2029];METHODS=[str(s) for s in SEEDS]+['hr_median','psd_mean'];CONDS=['h264_250k','h264_500k','h264_1000k']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def predictions(waves):
    valid=np.isfinite(waves).all(axis=-1)&(np.std(waves,axis=-1)>=1e-10)
    freq=np.fft.rfftfreq(4096,1/30);f=freq[(freq>=.75)&(freq<=3)]*60
    p=np.full((*waves.shape[:2],len(f)),np.nan)
    if valid.any():ff,pp=independent_psd(waves[valid]);np.testing.assert_array_equal(f,ff);p[valid]=pp
    preds={str(s):[float(f[np.argmax(p[j,i])]) if valid[j,i] else None for i in range(42)] for j,s in enumerate(SEEDS)}
    preds['hr_median']=[];preds['psd_mean']=[]
    for i in range(42):
        preds['hr_median'].append(float(np.median([preds[str(s)][i] for s in SEEDS])) if valid[:,i].all() else None)
        preds['psd_mean'].append(float(f[np.argmax(p[:,i].mean(axis=0))]) if valid[:,i].all() else None)
    return preds,p,valid
def main():
    done=read(BASE/'COMPLETE.json');assert done['complete'] and done['timed_video_config_runs']==63 and not done['test_evaluated']
    result=read(BASE/'results.json');protocol=read(BASE/'protocol.json');assert result['protocol_sha256']==sha(BASE/'protocol.json')
    for n,h in protocol['code_sha256'].items():assert sha(OUT/'code/deep_hr'/n)==h
    assert read(BASE/'frontend_tests.json')['passed']
    replay=read(BASE/'clean_control_replay.json');assert len(replay)==7 and all(r['all_boxes_exact'] for r in replay)
    clean=read(BASE/'input_source/clean_control.json');assert len(clean)==7 and all(r['boxes_exact'] and r['merged_RGB_exact'] for r in clean)
    controls=read(BASE/'native_pixel_controls.json');assert len(controls)==21 and all(r['same_host_native_pixels_exact'] for r in controls)
    inv=read(BASE/'native_reference/inventory.json')
    for n,h in inv.items():assert sha(BASE/'native_reference'/n)==h
    oldbase=HERE/'reports/spectral_ensemble_20261019/results/spectral_ensemble_20261019'
    old=read(oldbase/'results.json');oldprotocol=read(oldbase/'protocol.json')
    assert protocol['old_ensemble_protocol_sha256']==sha(oldbase/'protocol.json') and protocol['checkpoints']==oldprotocol['checkpoints']
    assert protocol['input_protocol_sha256']==sha(BASE/'input_source/protocol.json')
    assert protocol['code_sha256']['video_roi_frontend.py']==read(BASE/'input_source/protocol.json')['code_sha256']['video_roi_frontend.py']
    for family in ['mse','spectral']:
        previous=HERE/'reports'/('h264_experiment_20261015/results/h264_ablation_v1' if family=='mse' else 'spectral_experiment_20261015/results/spectral_ablation_v1')
        for seed in SEEDS:assert sha(previous/f'augmentation_{seed}/best.pt')==protocol['checkpoints'][family][str(seed)]['sha256']
    coverage=read(BASE/'coverage.json');front=read(BASE/'frontend_reports.json');groups=result['groups'];maxdelta=0.;flat=[];summary={};window_changes={}
    for family,data in groups.items():
        summary[family]={};window_changes[family]={}
        for c in CONDS:
            z=np.load(BASE/f'waveforms/{family}_{c}.npz');summary[family][c]={};window_changes[family][c]={}
            for branch,prefix in [('native_ROI','native_'),('detected_ROI','')]:
                waves=z[prefix+'waves'];assert waves.shape==(3,42,300);preds,psd,valid=predictions(waves)
                stored=z[prefix+'normalized_psd'];allgood=valid.all(axis=0)
                if allgood.any():
                    delta=float(np.max(np.abs(psd.transpose(1,0,2)[allgood]-stored[allgood])));assert delta<1e-7;maxdelta=max(maxdelta,delta)
                summary[family][c][branch]={}
                for method in METHODS:
                    r=data[c][branch][method];audit_report(r);assert r['n']==42 and all(x['split']=='val' for x in r['records'])
                    for i,row in enumerate(r['records']):assert row['pred_hr']==preds[method][i]
                    summary[family][c][branch][method]={k:v for k,v in r.items() if k!='records'}
                    flat.append({'family':family,'condition':c,'ROI':branch,'method':method,**summary[family][c][branch][method]})
                    if branch=='native_ROI':
                        previous=old['groups'][family]['validation'][c][method]
                        for row,prev in zip(r['records'],previous['records']):assert all(row[k]==prev[k] for k in ['subject','window','start_s','gt_hr','pred_hr','abs_error'])
            for method in METHODS:
                a=data[c]['native_ROI'][method]['records'];b=data[c]['detected_ROI'][method]['records'];changes=[]
                for left,right in zip(a,b):
                    for k in ['subject','window','gt_hr','start_s']:assert left[k]==right[k]
                    changes.append({'subject':left['subject'],'window':left['window'],'start_s':left['start_s'],'gt_hr':left['gt_hr'],
                        'native_hr':left['pred_hr'],'detected_hr':right['pred_hr'],'native_error':left['abs_error'],'detected_error':right['abs_error'],
                        'error_change':right['abs_error']-left['abs_error'],'invalid_detected':right.get('invalid',False)})
                window_changes[family][c][method]=sorted(changes,key=lambda r:r['error_change'],reverse=True)
    covsummary={}
    for c,rows in coverage.items():
        assert len(rows)==42;total_frames=0;valid_frames=0;attempts=0;success=0;boxdiff=0.;iousum=0.
        for rr in front[c]:
            s=rr['subject'];a=np.load(BASE/'native_reference'/f'{s}_meta.npz');b=np.load(OUT/f'datasets/UBFC-rPPG/detected_on_h264_20261019/{c}/{s}_meta.npz')
            for k in ['t','ppg','original_t']:np.testing.assert_array_equal(a[k],b[k])
            u,inverse=np.unique(a['original_t'],return_inverse=True);valid_u=np.ones(len(u),bool);np.logical_and.at(valid_u,inverse,b['original_valid'])
            np.testing.assert_array_equal(u,b['t']);np.testing.assert_array_equal(valid_u,b['valid'])
            assert len(b['boxes'])==rr['expected_frames'];assert int(b['original_valid'].sum())==rr['usable_ROI_frames']
            for row in [x for x in rows if x['subject']==s]:
                q=row['start_s']+np.arange(301)/30;j=np.clip(np.searchsorted(u,q,side='right'),1,len(u)-1);w=((q-u[j-1])/(u[j]-u[j-1])).astype(np.float32)
                good=((w>=1)|b['valid'][j-1])&((w<=0)|b['valid'][j]);assert int((~good).sum())==row['invalid_resampled_samples']
                if not good.all():assert not row['usable']
                if not row['usable']:
                    for family in groups:
                        for method in METHODS:
                            rec=next(x for x in groups[family][c]['detected_ROI'][method]['records'] if x['subject']==s and x['window']==row['window'])
                            assert rec['invalid'] and rec['abs_error']==135. and rec['pred_hr'] is None
            n=rr['usable_ROI_frames'];total_frames+=rr['expected_frames'];valid_frames+=n;attempts+=rr['detection_attempts'];success+=rr['fresh_detection_successes']
            if n:boxdiff+=rr['ROI_agreement_diagnostic']['different_boxes_fraction']*n;iousum+=rr['ROI_agreement_diagnostic']['mean_IoU_to_native']*n
        covsummary[c]={'total_frames':total_frames,'usable_ROI_frames':valid_frames,'unusable_ROI_frames':total_frames-valid_frames,
            'detection_attempts':attempts,'successful_detection_attempts':success,'success_fraction_of_attempts':success/attempts if attempts else None,
            'usable_windows':sum(r['usable'] for r in rows),'unusable_windows':sum(not r['usable'] for r in rows),'expected_windows':42,
            'different_boxes_fraction_of_usable_frames':boxdiff/valid_frames if valid_frames else None,'mean_IoU_to_native_over_usable_frames':iousum/valid_frames if valid_frames else None}
    timing=read(BASE/'timing.json');assert len(timing)==63 and all(r['all_predictions_match_cache_pass'] for r in timing)
    tsummary={}
    for c in CONDS:
        tsummary[c]={}
        for family,method in [('mse','2029'),('mse','psd_mean'),('spectral','psd_mean')]:
            records=[r for r in timing if r['condition']==c and r['family']==family and r['method']==method];assert len(records)==7
            assert len({r['subject'] for r in records})==7;assert sum(r['windows'] for r in records)==42
            elapsed=np.array([r['wall_s'] for r in records]);assert (elapsed>0).all()
            tsummary[c][family+'_'+method]={'videos':7,'wall_s_per_video_median':float(np.median(elapsed)),
                'wall_s_per_video_p95':float(np.percentile(elapsed,95)),'total_wall_s':float(elapsed.sum()),
                'total_recorded_duration_s':sum(r['recorded_duration_s'] for r in records),
                'compute_wall_over_recorded_duration':float(elapsed.sum()/sum(r['recorded_duration_s'] for r in records)),
                'frontend_fraction_of_total_wall':float(sum(r['frontend_wall_s'] for r in records)/elapsed.sum()),
                'invalid_windows':sum(r['invalid_windows'] for r in records)}
    concentration={}
    for family in ['mse','spectral']:
        concentration[family]={}
        for c in CONDS:
            rows=window_changes[family][c]['psd_mean'];net=sum(r['error_change'] for r in rows);worst=rows[0]
            concentration[family][c]={'improved_windows':sum(r['error_change']<0 for r in rows),'worsened_windows':sum(r['error_change']>0 for r in rows),
               'tied_windows':sum(r['error_change']==0 for r in rows),'net_abs_error_increase':net,
               'largest_regression_subject':worst['subject'],'largest_regression_window':worst['window'],
               'largest_regression_share_of_net_increase':worst['error_change']/net if net>0 else None}
    verified={'metrics':summary,'coverage':covsummary,'timing':tsummary,'regression_concentration':concentration,'all_numeric_and_missingness_audits_passed':True,'max_independent_PSD_difference':maxdelta,
       'test_evaluated':False,'timing_scope':'Direct same-P100 full-record offline wall clock, warm file cache and resident models. No capture/network/encoding/weights loading; not streaming first-output latency.',
       'ROI_agreement_scope':'Agreement with clean-derived boxes is diagnostic, not anatomical detection accuracy.'}
    write('verified_comparison.json',verified);write('window_changes.json',window_changes)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['DETECTOR TRỰC TIẾP TRÊN VIDEO H264 — 2026-10-19',
       'Đã hoàn tất 21 video/7 người,42 cửa sổ mỗi bitrate;6 checkpoint,5 predictor mỗi nhóm. Không huấn luyện/chọn lại tham số; không đánh giá test.',
       'Đối chứng sạch7/7: ROI và RGB tái tạo chính xác trên A100, mọi bbox replay chính xác trên P100.',
       'Đối chứng pixel21/21 video: native ROI crop lại trên P100 khớp hoàn toàn cache cũ. Mọi HR native ROI batch1 tái lập kết quả thí nghiệm ensemble trước.',
       f'Mọi dự đoán được tính lại độc lập từ waveform; chênh PSD lớn nhất {maxdelta:.3g}.',
       'ROI mất: frame chưa có mặt hoặc>4 scheduled misses không hợp lệ. Không bù bằng ROI sạch, không nội suy qua frame lỗi. Cửa sổ dùng hỗ trợ nội suy lỗi được giữ đủ mẫu và tính135bpm.',
       'Metric là subject-macro MAE trên toàn42 cửa sổ/rate. Tỷ lệ detector tìm được mặt và IoU với ROI native không phải độ chính xác theo ground truth giải phẫu.',
       'Mỗi dòng: group/method/condition | MAE native->detected | P95 native->detected | >20 native->detected | invalid detected']
    for family in ['mse','spectral']:
        for method in METHODS:
            for c in CONDS:
                a=summary[family][c]['native_ROI'][method];b=summary[family][c]['detected_ROI'][method]
                lines.append(f"{family}/{method}/{c}: {a['subject_macro_MAE']:.6f}->{b['subject_macro_MAE']:.6f} | {a['P95']:.6f}->{b['P95']:.6f} | {a['errors_over20']}->{b['errors_over20']} | invalid={b['invalid']}")
    lines+=['COVERAGE (KHÔNG LOẠI BỎ CỬA SỔ LỖI)',json.dumps(covsummary,ensure_ascii=False,indent=2),
        'THỜI GIAN ĐO TRỰC TIẾP (GIÂY/TOÀN VIDEO, OFFLINE)',json.dumps(tsummary,ensure_ascii=False,indent=2),
        verified['timing_scope'],'Đây là validation đã dùng phát triển. Thí nghiệm định vị tác động của ROI trên đúng dữ liệu này, không phải bằng chứng tổng quát hóa hay bảo đảm xếp hạng challenge.',
        'MỨC SUY GIẢM TẬP TRUNG Ở ĐÂU (PHÂN TÍCH HẬU NGHIỆM)',json.dumps(concentration,ensure_ascii=False,indent=2),
        'Phần lớn mức suy giảm MSE ở các bitrate và spectral ở250k tập trung vào subject32/window2. Không suy rộng thành suy giảm đồng đều mọi cửa sổ.',
        'Trung vị HR của nhóm spectral ở250k trên ROI thực tế tốt hơn PSD-mean. Không chọn decoder theo bitrate sau khi xem kết quả; cần giữ nguyên các ứng viên để kiểm chứng độc lập.',
        'CÁC CỬA SỔ PSD ENSEMBLE XẤU ĐI NHIỀU NHẤT']
    for family in ['mse','spectral']:
        for c in CONDS:lines+=[family+' '+c,json.dumps(window_changes[family][c]['psd_mean'][:3],ensure_ascii=False,indent=2)]
    (OUT/'REPORT_VI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained',sharey=True);x=np.arange(3)
    for ax,family in zip(axes,['mse','spectral']):
        for method,color,label in [('2029','#4477aa','Seed2029'),('psd_mean','#228833','PSD ensemble')]:
            for branch,style,short in [('native_ROI','--','native ROI'),('detected_ROI','-','detect on H264')]:
                ax.plot(x,[summary[family][c][branch][method]['subject_macro_MAE'] for c in CONDS],marker='o',ls=style,color=color,label=label+' / '+short)
        ax.set_xticks(x,['250kbps','500kbps','1000kbps']);ax.set(title=family+' + H264',ylabel='Validation subject-macro MAE (bpm)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(OUT/'detector_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({'coverage':covsummary,'PSD_MAE':{f:{c:[summary[f][c][b]['psd_mean']['subject_macro_MAE'] for b in ['native_ROI','detected_ROI']] for c in CONDS} for f in summary},
       'PSD_P95_over20':{f:{c:{b:[summary[f][c][b]['psd_mean']['P95'],summary[f][c][b]['psd_mean']['errors_over20']] for b in ['native_ROI','detected_ROI']} for c in CONDS} for f in summary},'timing':tsummary},indent=2))
if __name__=='__main__':main()
