"""Independent OOF audit and five-fold report; descriptive development comparison."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_spectral_ensemble import independent_psd,audit_report
HERE=Path(__file__).resolve().parent;OUT=HERE/'reports/cv_5fold_20261020';BASE=OUT/'results/cv_5fold_20261020'
FAMILIES=['mse','spectral'];CONDITIONS=['clean','h264_500k','h264_250k','h264_1000k']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    done=read(BASE/'COMPLETE.json');assert done['runs']==10 and done['epochs_each']==30 and done['folds']==5 and done['oof_subjects']==35 and done['oof_windows_per_condition']==203 and done['paired_verified'] and done['final_epoch_only'] and not done['test_opened']
    protocol=read(BASE/'protocol.json');assignment=read(OUT/'inputs/folds.json');primary=read(OUT/'inputs/primary_manifest.json');secondary=read(OUT/'inputs/secondary_manifest.json');ev=read(BASE/'evaluation_manifest.json')
    assert sha(OUT/'inputs/folds.json')==protocol['folds_sha256'] and sha(OUT/'inputs/primary_manifest.json')==protocol['primary_manifest_sha256']==ev['primary_manifest_sha256']
    assert sha(OUT/'inputs/secondary_manifest.json')==ev['secondary_manifest_sha256'] and sha(BASE/'protocol.json')==ev['protocol_sha256']
    assert secondary['frontend_code_sha256']==sha(OUT/'code/deep_hr/video_roi_frontend.py')
    for manifest in [protocol,ev]:
        for name,h in manifest['code_sha256'].items():assert sha(OUT/'code/deep_hr'/name)==h
    assert sha(OUT/'code/deep_hr/spectral_objective.py')==sha(HERE/'reports/spectral_experiment_20261015/code/deep_hr/spectral_objective.py')
    pool=assignment['pool_subjects'];test=set(assignment['original_test_subjects']);assert len(set(pool))==35 and len(test)==7 and not set(pool)&test
    perm=np.random.default_rng(2027).permutation(pool).tolist()
    assert [s for fold in assignment['folds'] for s in fold['val_subjects']]==perm
    rows=assignment['rows'];keys={(r['subject'],r['window']) for r in rows};assert len(rows)==len(keys)==203 and {r['subject'] for r in rows}==set(pool)
    results=read(BASE/'fold_results.json');oof=read(BASE/'oof_results.json');comparison=read(BASE/'comparison.json');frozen=read(BASE/'frozen_checkpoints.json');recheck=read(BASE/'checkpoint_recheck.json')
    assert len(frozen)==len(recheck)==10
    for n,item in frozen.items():assert item['epoch']==30 and recheck[n]['verified'] and item['sha256']==recheck[n]['sha256']
    flat=[];durations={};maxdelta=0.;allpred={fam:{c:[] for c in CONDITIONS} for fam in FAMILIES};foldscores=[]
    for fold in range(5):
        spec=assignment['folds'][fold];train=set(spec['train_subjects']);val=set(spec['val_subjects']);assert len(train)==28 and len(val)==7 and train|val==set(pool) and not train&val
        held=[{**r,'original_split':r['split'],'split':'cv_val','fold':fold} for r in rows if r['subject'] in val]
        training=[{**r,'original_split':r['split'],'split':'cv_train','fold':fold} for r in rows if r['subject'] in train]
        trainhash=hashlib.sha256(json.dumps(training,sort_keys=True).encode()).hexdigest();pair=[]
        for family in FAMILIES:
            root=BASE/f'fold{fold}'/family;st=read(root/'status.json');prov=read(root/'provenance.json');h=read(root/'history.json')
            assert st['state']=='complete' and st['completed_epochs']==st['final_epoch']==30 and st['reload_verified'] and len(h)==30
            assert st['checkpoint_sha256']==frozen[f'fold{fold}_{family}']['sha256']
            assert prov['protocol_sha256']==sha(BASE/'protocol.json') and prov['train_rows_sha256']==trainhash and prov['config']['seed']==2027 and prov['config']['train_windows']==len(training)
            assert all('score' not in item and 'conditions' not in item for item in h)
            durations[f'fold{fold}_{family}']=sum(item['elapsed_s'] for item in h);pair.append((prov,h))
            for c in CONDITIONS:
                report=results[str(fold)][family][c];audit_report(report);assert report['n']==len(held)
                for x,y in zip(report['records'],held):assert all(x[k]==y[k] for k in y)
                w=np.load(BASE/f'evaluation/fold{fold}_{family}_{c}.npz')['waves'];assert w.shape==(len(held),300)
                finite=np.isfinite(w).all(axis=1)&(np.std(w,axis=1)>=1e-10);pred=[None]*len(held)
                if finite.any():
                    f,p=independent_psd(w[finite]);ii=np.where(finite)[0]
                    for j,i in enumerate(ii):pred[i]=float(f[np.argmax(p[j])])
                for i,row in enumerate(report['records']):
                    m=np.load(OUT/f'inputs/meta/{c}'/(row['subject']+'_meta.npz'));t=m['t'];q=row['start_s']+np.arange(301)/30
                    j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1);ww=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
                    valid=m['valid'] if 'valid' in m else np.ones(len(t),bool)
                    supports=((ww>=1)|valid[j-1])&((ww<=0)|valid[j]);usable=q[0]>=t[0] and q[-1]<=t[-1] and supports.all()
                    if not usable:assert row['pred_hr'] is None and row['invalid'] and row['abs_error']==135.
                    assert row['pred_hr']==pred[i]
                allpred[family][c].extend(report['records'])
                flat.append({'fold':fold,'family':family,'condition':c,**{k:v for k,v in report.items() if k!='records'}})
        (a,ha),(b,hb)=pair
        assert a['initial_model_sha256']==b['initial_model_sha256'] and a['torch']==b['torch'] and a['gpu']==b['gpu']
        assert {k:v for k,v in a['config'].items() if k not in ['loss','spectral_weight']}=={k:v for k,v in b['config'].items() if k not in ['loss','spectral_weight']}
        assert all(x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips'] for x,y in zip(ha,hb))
        scores={fam:float(np.mean([results[str(fold)][fam][c]['subject_macro_MAE'] for c in ['clean','h264_500k']])) for fam in FAMILIES}
        foldscores.append({'fold':fold,**scores,'change_pct':100*(scores['spectral']/scores['mse']-1),'train_windows':len(training),'val_windows':len(held),'heldout_subjects':spec['val_subjects']})
    # Audit every subject's immutable metadata against the exact cache manifests.
    for c in CONDITIONS:
        manifest=primary['conditions']['clean' if c=='clean' else 'detected_500k'] if c in ['clean','h264_500k'] else secondary['conditions'][c]
        for s in pool:
            p=OUT/f'inputs/meta/{c}'/(s+'_meta.npz');assert sha(p)==manifest[s]['_meta.npz']['sha256']
            a=np.load(OUT/'inputs/meta/clean'/(s+'_meta.npz'));b=np.load(p)
            for key in ['t','ppg','original_t']:np.testing.assert_array_equal(a[key],b[key])
    summary={}
    for fam in FAMILIES:
        summary[fam]={}
        for c in CONDITIONS:
            rr=sorted(allpred[fam][c],key=lambda r:(int(r['subject'][7:]),r['window']));r=oof[fam][c];audit_report(r)
            assert len(rr)==203 and {(x['subject'],x['window']) for x in rr}==keys and {x['subject'] for x in rr}==set(pool) and not test&{x['subject'] for x in rr}
            assert rr==r['records']
            values=[results[str(k)][fam][c]['subject_macro_MAE'] for k in range(5)]
            assert abs(np.mean(values)-r['subject_macro_MAE'])<1e-9
            summary[fam][c]={**{k:v for k,v in r.items() if k!='records'},'fold_MAE_sd':float(np.std(values,ddof=1))}
            assert abs(summary[fam][c]['fold_MAE_sd']-comparison['fold_summary'][fam][c]['MAE_sd'])<1e-9
    score=lambda d:(d['clean']['subject_macro_MAE']+d['h264_500k']['subject_macro_MAE'])/2
    a,b=summary['mse'],summary['spectral'];improved=sum(f['spectral']<f['mse'] for f in foldscores)
    checks={'primary_gain_at_least_10pct':score(b)<=.9*score(a),'clean_increase_at_most_025':b['clean']['subject_macro_MAE']<=a['clean']['subject_macro_MAE']+.25,
        'improved_at_least_3_folds':improved>=3,'h264500_P95_increase_at_most_1':b['h264_500k']['P95']<=a['h264_500k']['P95']+1,
        'h264500_over20_not_increased':b['h264_500k']['errors_over20']<=a['h264_500k']['errors_over20']}
    assert checks==comparison['checks'] and all(checks.values())==comparison['keep_gate_passed']==done['keep_gate_passed'] and improved==comparison['improved_folds']
    subject_scores={}
    for s in pool:
        subject_scores[s]={fam:float(np.mean([np.mean([r['abs_error'] for r in oof[fam][c]['records'] if r['subject']==s]) for c in ['clean','h264_500k']])) for fam in FAMILIES}
        subject_scores[s]['delta']=subject_scores[s]['spectral']-subject_scores[s]['mse']
    assert abs(np.mean([v['delta'] for v in subject_scores.values()])-(score(b)-score(a)))<1e-9
    window_changes={}
    for c in CONDITIONS:
        changes=[]
        for x,y in zip(oof['mse'][c]['records'],oof['spectral'][c]['records']):
            assert (x['subject'],x['window'])==(y['subject'],y['window'])
            changes.append({'subject':x['subject'],'window':x['window'],'fold':x['fold']+1,'gt_hr':x['gt_hr'],'mse_hr':x['pred_hr'],'spectral_hr':y['pred_hr'],'mse_error':x['abs_error'],'spectral_error':y['abs_error'],'error_delta':y['abs_error']-x['abs_error']})
        window_changes[c]=sorted(changes,key=lambda r:r['error_delta'],reverse=True)
    save('window_changes_audited.json',window_changes)
    save('verified_comparison.json',{'all10_runs_complete':True,'paired_verified':True,'all_HR_independently_reproduced':True,'source_metadata_and_ROI_missingness_verified':True,
        'summary':summary,'fold_scores':foldscores,'subject_scores':subject_scores,'primary_score_mse':score(a),'primary_score_spectral':score(b),'primary_relative_change_pct':100*(score(b)/score(a)-1),'improved_folds':improved,
        'checks':checks,'keep_gate_passed':all(checks.values()),'train_epoch_total_seconds':durations,'test_opened':False})
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['SO SÁNH MSE VÀ MSE +0,1 SPECTRAL KL QUA5 FOLD THEO SUBJECT','',
        'Hoàn tất10 lượt mới, mỗi lượt30 epoch. Seed2027 cố định; mỗi fold28 subject train và7 subject giữ lại.',
        'Chỉ dùng35 subject train/validation cũ. Bảy subject test được giữ nguyên, không đánh giá trong thí nghiệm này.',
        'Nhóm fold được đóng băng bằng hoán vị không dùng nhãn. Mỗi subject chỉ được chấm bởi mô hình của fold không train trên subject đó.',
        'Không đánh giá heldout trong30 epoch, không chọn best epoch, không early stopping: chỉ checkpoint cuối epoch30.',
        'EfficientPhys36, batch2, AdamW wd0, OneCycleLR max0.009, 50% clean/50% H264500, ROI train từ ảnh gốc; hai nhánh chỉ khác loss.',
        'Đã kiểm tra khởi tạo, config ngoài loss, train rows, shuffle và lượt augmentation khớp từng fold;10 checkpoint khớp hash và reload.',
        'Toàn bộ HR/chỉ số được tái tính độc lập bằng detrend dense SPD. Nhãn, timestamp và mask ROI trong các điều kiện khớp metadata nguồn.',
        '', 'MAE subject-macro OOF trên35 người, kèm SD mẫu của5 fold (bpm):','Điều kiện | MSE | MSE+KL | thay đổi']
    for c in CONDITIONS:lines.append(f"{c} | {a[c]['subject_macro_MAE']:.6f} ± {a[c]['fold_MAE_sd']:.6f} | {b[c]['subject_macro_MAE']:.6f} ± {b[c]['fold_MAE_sd']:.6f} | {100*(b[c]['subject_macro_MAE']/a[c]['subject_macro_MAE']-1):+.2f}%")
    lines+=['','Điểm chính clean/500 từng fold: fold | MSE | MSE+KL | thay đổi | số cửa sổ heldout']
    for r in foldscores:lines.append(f"{r['fold']+1} | {r['mse']:.6f} | {r['spectral']:.6f} | {r['change_pct']:+.2f}% | {r['val_windows']}")
    lines+=['',f'Điểm chính OOF: {score(a):.6f} → {score(b):.6f}; thay đổi {100*(score(b)/score(a)-1):+.2f}%. KL cải thiện{improved}/5 fold.',f'Cổng giữ cấu hình định trước: {all(checks.values())}.']
    lines += [f'{k}: {v}' for k,v in checks.items()]
    lines+=['','P95 / lỗi >20 bpm / invalid, tính trên203 dự đoán OOF mỗi điều kiện:']
    for c in CONDITIONS:lines.append(f"{c}: P95 {a[c]['P95']:.6f} → {b[c]['P95']:.6f}; >20 {a[c]['errors_over20']} → {b[c]['errors_over20']}; invalid {a[c]['invalid']} → {b[c]['invalid']}")
    lines+=['','Thay đổi điểm chính theo subject (mô tả hậu nghiệm, không sửa quy tắc inference):']
    for s,r in sorted(subject_scores.items(),key=lambda x:x[1]['delta'],reverse=True):lines.append(f"{s}: {r['mse']:.6f} → {r['spectral']:.6f} ({r['delta']:+.6f})")
    lines+=['','Các cửa sổ tăng lỗi lớn nhất ở500k (hậu nghiệm, không chỉnh decoder):']
    for r in window_changes['h264_500k'][:5]:lines.append(f"Fold{r['fold']} {r['subject']}/window{r['window']}, GT={r['gt_hr']:.6f}: MSE HR={r['mse_hr']}, KL HR={r['spectral_hr']}; lỗi {r['mse_error']:.6f} → {r['spectral_error']:.6f} ({r['error_delta']:+.6f})")
    lines+=['','Giới hạn: đây là CV development trên35 subject đã từng tham gia phát triển, không phải validation ngoài độc lập.',
        'Một seed không đo được độ ổn định theo seed. SD giữa fold phản ánh khác nhóm subject, không phải CI; các tập train giữa fold chồng lấp.',
        '203 cửa sổ thuộc35 người, không phải203 quan sát độc lập. 250/1000 là điều kiện phụ, không dùng để chọn epoch hoặc decoder.',
        'Không ensemble các model giữa fold: bốn trong năm model đã train trên subject được giữ lại của một fold khác.',
        'Không so trực tiếp OOF này với điểm best-checkpoint của split cũ để tuyên bố cải thiện: khác tập heldout và chính sách chọn checkpoint.',
        'Kết luận theo cổng định trước: '+('giữ KL làm ứng viên để kiểm tra tiếp bằng nhiều seed/dữ liệu ngoài, chưa thay thế bằng tuyên bố tổng quát hóa.' if all(checks.values()) else 'chưa thay baseline MSE bằng KL; không dò trọng số hoặc chọn epoch trên OOF sau khi thấy kết quả.')]
    (OUT/'REPORT_VI.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');x=np.arange(4)
    for shift,fam,col,label in [(-.17,'mse','#3b6fb6','MSE'),(.17,'spectral','#d77b32','MSE +0.1 spectral KL')]:
        d=summary[fam];axes[0].bar(x+shift,[d[c]['subject_macro_MAE'] for c in CONDITIONS],width=.32,yerr=[d[c]['fold_MAE_sd'] for c in CONDITIONS],capsize=4,color=col,label=label)
    axes[0].set(xticks=x,xticklabels=['Clean','H264 500k','H264 250k','H264 1000k'],ylabel='OOF subject-macro MAE (bpm)',title='35 subjects, 5 folds; bars show fold SD');axes[0].legend(fontsize=8)
    for r in foldscores:axes[1].plot([0,1],[r['mse'],r['spectral']],'o-',label=f"Fold{r['fold']+1}")
    axes[1].set(xticks=[0,1],xticklabels=['MSE','MSE +spectral KL'],ylabel='Clean/H264500 mean MAE (bpm)',title='Final epoch30, no heldout checkpoint selection');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    vals=sorted(subject_scores.items(),key=lambda r:r[1]['delta']);fig,ax=plt.subplots(figsize=(11,8),layout='constrained')
    y=np.arange(len(vals));ax.barh(y,[r['delta'] for s,r in vals],color=['#d77b32' if r['delta']>0 else '#3b6fb6' for s,r in vals]);ax.axvline(0,color='black',lw=1)
    ax.set(yticks=y,yticklabels=[s for s,r in vals],xlabel='KL minus MSE clean/H264500 MAE (bpm)',title='Paired OOF changes by subject; negative is better');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'subject_changes.png',dpi=160);plt.close(fig)
    print('\n'.join(lines[:28]))
if __name__=='__main__':
    if '--e0' in sys.argv:
        # New report is isolated from immutable historical hashes/protocols.
        from replay_band_e0 import main as replay
        replay()
    else:
        main()
