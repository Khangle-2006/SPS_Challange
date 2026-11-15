"""Evaluate only epoch30 and only the subject's heldout fold; no cross-fold fusion."""
import json,os
import numpy as np
import torch
from cv_common import ROOT,WORK,OUT,CONDITIONS,FAMILIES,SEED,read,write,sha,CVClips,rows_for,roots
from EfficientPhys_fixed import EfficientPhys
from spectral_ensemble import normalized_psd
from evaluate_spectral_ensemble import metrics
def verify_sources(manifest):
    for condition,subjects in manifest['conditions'].items():
        for s,files in subjects.items():
            for suffix,entry in files.items():
                from pathlib import Path
                p=Path(entry['path']);assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
def main():
    if (OUT/'COMPLETE.json').exists():raise RuntimeError('CV already evaluated; refusing overwrite')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    protocol=read(OUT/'protocol.json');assignment=read(WORK/'folds.json');secondary=read(OUT/'SECONDARY_COMPLETE.json')
    assert sha(WORK/'folds.json')==protocol['folds_sha256'] and sha(WORK/'primary_manifest.json')==protocol['primary_manifest_sha256']
    assert sha(WORK/'secondary_manifest.json')==secondary['secondary_manifest_sha256']
    verify_sources(read(WORK/'primary_manifest.json'));verify_sources(read(WORK/'secondary_manifest.json'))
    frozen={};pairing={}
    for fold in range(5):
        pair=[]
        for family in FAMILIES:
            p=OUT/f'fold{fold}'/family;st=read(p/'status.json');prov=read(p/'provenance.json');h=read(p/'history.json')
            assert st['state']=='complete' and st['completed_epochs']==st['final_epoch']==30 and st['reload_verified'] and len(h)==30
            assert sha(p/'final.pt')==st['checkpoint_sha256'] and prov['protocol_sha256']==sha(OUT/'protocol.json')
            assert all('score' not in item and 'conditions' not in item for item in h)
            frozen[f'fold{fold}_{family}']={'path':str(p/'final.pt'),'sha256':st['checkpoint_sha256'],'epoch':30};pair.append((prov,h))
        (a,ha),(b,hb)=pair
        assert a['initial_model_sha256']==b['initial_model_sha256'] and a['train_rows_sha256']==b['train_rows_sha256'] and a['torch']==b['torch'] and a['gpu']==b['gpu']
        assert {k:v for k,v in a['config'].items() if k not in ['loss','spectral_weight']}=={k:v for k,v in b['config'].items() if k not in ['loss','spectral_weight']}
        assert all(x['sample_order_sha256']==y['sample_order_sha256'] and x['compressed_train_clips']==y['compressed_train_clips'] for x,y in zip(ha,hb))
        pairing[str(fold)]={'initialization_config_order_mix_verified':True,'train_subjects':assignment['folds'][fold]['train_subjects'],'heldout_subjects':assignment['folds'][fold]['val_subjects']}
    write(OUT/'frozen_checkpoints.json',frozen);write(OUT/'pairing.json',pairing)
    results={};oof={fam:{c:[] for c in CONDITIONS} for fam in FAMILIES};dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
    for fold in range(5):
        rows=rows_for(fold,'val');assert len({r['subject'] for r in rows})==7 and not set(assignment['folds'][fold]['train_subjects'])&{r['subject'] for r in rows}
        results[str(fold)]={}
        for family in FAMILIES:
            item=frozen[f'fold{fold}_{family}'];ck=torch.load(item['path'],map_location='cuda',weights_only=False);assert ck['epoch']==29
            model=EfficientPhys(frame_depth=10,img_size=36).cuda().eval();model.load_state_dict(ck['model']);results[str(fold)][family]={}
            for c,root in roots().items():
                ds=CVClips(root,rows);waves=np.full((len(rows),300),np.nan,np.float32);records=[]
                with torch.no_grad():
                    for i,row in enumerate(rows):
                        hr=None;reason=None
                        if i in ds.invalid:reason='unusable_ROI_interpolation_support'
                        else:
                            try:
                                x=ds[i][0][None].cuda();pred=model(x);wave=torch.cat([torch.zeros_like(pred[:,:1]),torch.cumsum(pred[:,:-1],dim=1)],dim=1)[0].cpu().numpy();waves[i]=wave
                                f,p=normalized_psd(wave);hr=float(f[np.argmax(p)])
                            except ValueError as error:reason=str(error)
                        records.append({**row,'pred_hr':hr,'abs_error':abs(hr-row['gt_hr']) if hr is not None else 135.,'invalid':hr is None,'invalid_reason':reason})
                report=metrics(records);results[str(fold)][family][c]=report;oof[family][c].extend(records)
                np.savez(dest/f'fold{fold}_{family}_{c}.npz',waves=waves)
            del model,ck;torch.cuda.empty_cache();print('EVALUATED',fold,family,flush=True)
    pooled={};fold_summary={}
    expected={(r['subject'],r['window']) for r in assignment['rows']}
    for family in FAMILIES:
        pooled[family]={}
        for c in CONDITIONS:
            records=sorted(oof[family][c],key=lambda r:(int(r['subject'][7:]),r['window']))
            assert len(records)==len(expected)==203 and {(r['subject'],r['window']) for r in records}==expected
            assert {r['subject'] for r in records}==set(assignment['pool_subjects']) and not set(assignment['original_test_subjects'])&{r['subject'] for r in records}
            pooled[family][c]=metrics(records)
        fold_summary[family]={c:{'MAE_mean':float(np.mean([results[str(k)][family][c]['subject_macro_MAE'] for k in range(5)])),
            'MAE_sd':float(np.std([results[str(k)][family][c]['subject_macro_MAE'] for k in range(5)],ddof=1))} for c in CONDITIONS}
        for c in CONDITIONS:assert abs(fold_summary[family][c]['MAE_mean']-pooled[family][c]['subject_macro_MAE'])<1e-9
    score=lambda d:(d['clean']['subject_macro_MAE']+d['h264_500k']['subject_macro_MAE'])/2
    a,b=pooled['mse'],pooled['spectral'];improved=sum(score(results[str(k)]['spectral'])<score(results[str(k)]['mse']) for k in range(5))
    checks={'primary_gain_at_least_10pct':score(b)<=.9*score(a),'clean_increase_at_most_025':b['clean']['subject_macro_MAE']<=a['clean']['subject_macro_MAE']+.25,
        'improved_at_least_3_folds':improved>=3,'h264500_P95_increase_at_most_1':b['h264_500k']['P95']<=a['h264_500k']['P95']+1,
        'h264500_over20_not_increased':b['h264_500k']['errors_over20']<=a['h264_500k']['errors_over20']}
    write(OUT/'fold_results.json',results);write(OUT/'oof_results.json',pooled)
    write(OUT/'comparison.json',{'fold_summary':fold_summary,'primary_score_mse':score(a),'primary_score_spectral':score(b),'primary_relative_change_pct':100*(score(b)/score(a)-1),'improved_folds':improved,'checks':checks,'keep_gate_passed':all(checks.values()),'test_opened':False})
    sources=['evaluate_cv_loss.py','cv_common.py','prepare_cv_secondary.py','video_roi_frontend.py','spectral_ensemble.py']
    write(OUT/'evaluation_manifest.json',{'protocol_sha256':sha(OUT/'protocol.json'),'primary_manifest_sha256':sha(WORK/'primary_manifest.json'),'secondary_manifest_sha256':sha(WORK/'secondary_manifest.json'),'code_sha256':{n:sha(ROOT/'code/deep_hr'/n) for n in sources},'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'test_opened':False})
    write(OUT/'COMPLETE.json',{'runs':10,'epochs_each':30,'folds':5,'oof_subjects':35,'oof_windows_per_condition':203,'paired_verified':True,'final_epoch_only':True,'keep_gate_passed':all(checks.values()),'test_opened':False})
    print(json.dumps(read(OUT/'comparison.json'),indent=2),flush=True)
if __name__=='__main__':main()
