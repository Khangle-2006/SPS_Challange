"""Precommitted 3-seed PSD ensembles vs individual seeds and HR median; no test."""
import json,os,platform,subprocess,time
from pathlib import Path
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from train_h264_ablation import ROOT,CLEAN,COMP_TRAIN,COMP_VAL,SEEDS,Clips,verify,same_pair,EfficientPhys
from train_efficientphys import sha
from spectral_ensemble import summarize_three,normalized_psd
from verify_spectral_ensemble import run as verify_math
OUT=ROOT/'results/spectral_ensemble_20261019'
CONDS=['clean','h264_500k','h264_250k','h264_1000k'];METHODS=[str(s) for s in SEEDS]+['hr_median','psd_mean']
def read(p):return json.loads(p.read_text())
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2))
def metrics(rows):
    e=np.array([r['abs_error'] for r in rows])
    return {'subject_macro_MAE':float(np.mean([np.mean([r['abs_error'] for r in rows if r['subject']==s]) for s in sorted({r['subject'] for r in rows})])),
      'window_MAE':float(e.mean()),'window_RMSE':float(np.sqrt((e**2).mean())),'P95':float(np.percentile(e,95,method='linear')),
      'errors_over10':int((e>10).sum()),'errors_over20':int((e>20).sum()),'max_error':float(e.max()),
      'invalid':sum(r.get('invalid',False) for r in rows),'n':len(rows),'records':rows}
def make_reports(waves,rows):
    records={m:[] for m in METHODS};powers=[]
    for i,row in enumerate(rows):
        try:
            r=summarize_three(waves[:,i,:]);preds=dict(zip([str(s) for s in SEEDS],r['single_hr']))
            preds.update(hr_median=r['median_hr'],psd_mean=r['psd_mean_hr']);powers.append(r['psd'])
        except ValueError:
            # Preserve each valid single; both ensembles require all three.
            preds={}
            for j,s in enumerate(SEEDS):
                try:f,p=normalized_psd(waves[j,i]);preds[str(s)]=float(f[np.argmax(p)])
                except ValueError:preds[str(s)]=None
            bins=np.fft.rfftfreq(4096,1/30.);nb=int(((bins>=.75)&(bins<=3)).sum())
            preds.update(hr_median=None,psd_mean=None);powers.append(np.full((3,nb),np.nan))
        for m,hr in preds.items():
            rr={**row,'pred_hr':hr,'abs_error':abs(hr-row['gt_hr']) if hr is not None else 135.}
            if hr is None:rr['invalid']=True
            records[m].append(rr)
    return {m:metrics(r) for m,r in records.items()},np.stack(powers)
def forward_wave(model,x):
    pred=model(x)
    return torch.cat([torch.zeros_like(pred[:,:1]),torch.cumsum(pred[:,:-1],dim=1)],dim=1)
def benchmark(models,ds):
    report={};order=[str(s) for s in SEEDS]+['hr_median','psd_mean']
    # Rotated method order across clips reduces fixed-order thermal/cache bias.
    ms={m:[] for m in order};gpu_ms={m:[] for m in order}
    with torch.no_grad():
        for m in order:
            selected=[models[int(m)]] if m.isdigit() else list(models.values())
            for i in range(5):
                x=ds[i][0][None].cuda()
                for model in selected:forward_wave(model,x)
                torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for i in range(len(ds)):
            rotated=order[i%5:]+order[:i%5]
            for m in rotated:
                selected=[models[int(m)]] if m.isdigit() else list(models.values())
                torch.cuda.synchronize();start=time.perf_counter();x=ds[i][0][None].cuda();torch.cuda.synchronize()
                gstart=time.perf_counter();ww=[forward_wave(model,x)[0].cpu().numpy() for model in selected];torch.cuda.synchronize();gend=time.perf_counter()
                if m.isdigit():f,p=normalized_psd(ww[0]);hr=float(f[np.argmax(p)])
                elif m=='hr_median':
                    hh=[]
                    for w in ww:f,p=normalized_psd(w);hh.append(float(f[np.argmax(p)]))
                    hr=float(np.median(hh))
                else:hr=summarize_three(np.stack(ww))['psd_mean_hr']
                assert np.isfinite(hr);ms[m].append((time.perf_counter()-start)*1000);gpu_ms[m].append((gend-gstart)*1000)
    for m in order:
        report[m]={'n_clips':len(ms[m]),'end_to_end_ms_median':float(np.median(ms[m])),'end_to_end_ms_p95':float(np.percentile(ms[m],95)),
           'forward_and_wave_copy_ms_median':float(np.median(gpu_ms[m])),'samples_ms':ms[m],
           'effective_input_frames_per_second':float(301/(np.median(ms[m])/1000))}
    return {'methods':report,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
       'scope':'Batch1, resident models, warm cached 36px RGB; timed clip timestamp interpolation/normalization, H2D, sequential forward/cumulative reconstruction, D2H and HR postprocessing. Excludes raw video decode, face detection, checkpoint loading and waiting to acquire the 10-second clip. P100, not target-device deployment latency.'}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():raise RuntimeError('Experiment already completed; refusing overwrite')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    frozen={}
    for family,folder in [('mse','h264_ablation_v1'),('spectral','spectral_ablation_v1')]:
        frozen[family]={}
        for seed in SEEDS:
            p=ROOT/'results'/folder/f'augmentation_{seed}'/'best.pt';st=read(p.parent/'status.json')
            assert st['state']=='complete' and st['reload_verified'] and sha(p)==st['checkpoint_sha256']
            frozen[family][str(seed)]={'path':str(p),'sha256':st['checkpoint_sha256'],'best_epoch':st['best_epoch']}
    protocol={'groups':['mse','spectral'],'seeds':SEEDS,'checkpoints':frozen,'ROI':'original fixed native ROI, no EMA',
       'psd_fusion':'Reconstruct 300 samples from forward differences; smoothness detrend lambda100; zero-phase first-order Butterworth .75-3Hz; periodic Hann periodogram nfft4096; normalize EACH seed power to sum1 over .75-3Hz; arithmetic mean with weights1/3; argmax, earliest frequency on ties. No peak interpolation.',
       'median_baseline':'Median of exactly three scalar HR predictions from the unchanged single estimator.',
       'invalid_policy':'Single invalid penalty135 bpm; either ensemble invalid if any of three members invalid; never drop members/windows.',
       'conditions':CONDS,'selection_or_tuning':'None: no new checkpoint, weighting, ROI, FPS, smoothing or parameter tuning.',
       'primary':'Equal mean clean and H264500 validation subject-macro MAE; all 7 subjects/42 windows. Other rates are already-seen secondary development conditions.',
       'retain_gate':'PSD mean must improve primary score >=10% vs BOTH mean single-seed score and HR-median score; clean MAE increase <=.25 vs both; H264500 P95 increase <=1 vs both; H264500 >20 rate not increase vs either. No claim of independent generalization.',
       'train':'Supplementary only: reuse previously audited 6-model waveforms for all161 train windows, validate checkpoint hashes and all single HRs again.',
       'timing':'Batch1 on all42 clean val clips, 5 warmups/method, rotated method order, sequential three models, resident weights. Report median/P95; no real-time deployment claim.',
       'test_evaluated':False,'code_sha256':{n:sha(Path(__file__).parent/n) for n in ['evaluate_spectral_ensemble.py','spectral_ensemble.py','verify_spectral_ensemble.py','train_efficientphys.py','EfficientPhys_fixed.py','hr_baseline.py']}}
    pp=OUT/'protocol.json'
    if pp.exists():assert read(pp)==protocol
    else:write('protocol.json',protocol)
    test=verify_math();write('math_verification.json',test);print('MATH VERIFICATION PASSED',flush=True)
    roots={'clean':CLEAN,'h264_500k':COMP_VAL,'h264_250k':ROOT/'datasets/UBFC-rPPG/efficientphys36_val_h264_250k_v1','h264_1000k':ROOT/'datasets/UBFC-rPPG/efficientphys36_val_h264_1000k_v1'}
    datasets={}
    for c,p in roots.items():verify(p);datasets[c]=Clips(p,'val');assert len(datasets[c])==42
    for ds in datasets.values():same_pair(datasets['clean'],ds)
    cached=ROOT/'results/train_compression_diagnosis_20261019';train_result=read(cached/'results.json');train_protocol=read(cached/'protocol.json');tw=np.load(cached/'model_waveforms.npz')
    assert train_result['complete'] and train_result['all_saved_HR_predictions_reproduced']
    assert sha(cached/'protocol.json')==train_result['protocol_sha256']
    train_rows=[{k:r[k] for k in ['subject','split','window','start_s','gt_hr','pos_hr','max_gap_s']} for r in train_result['records']]
    assert len(train_rows)==161 and all(r['split']=='train' for r in train_rows)
    write('source_verification.json',{'train_waveform_sha256':sha(cached/'model_waveforms.npz'),'train_diagnostic_protocol_sha256':sha(cached/'protocol.json'),
       'cache_inventory_sha256':{c:sha(p/'inventory.json') for c,p in roots.items()},'protocol_sha256':sha(pp)})
    allresults={};timings={};dest=OUT/'waveforms';dest.mkdir(exist_ok=True)
    for family in ['mse','spectral']:
        models={};valwaves={c:np.empty((3,42,300),np.float32) for c in CONDS};prior={}
        for j,seed in enumerate(SEEDS):
            p=Path(frozen[family][str(seed)]['path']);assert sha(p)==frozen[family][str(seed)]['sha256']
            model=EfficientPhys(frame_depth=10,img_size=36).cuda().eval();model.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);models[seed]=model
            prior[seed]=read(ROOT/'results/spectral_ablation_v1/frozen_evaluation'/f'{family}_augmentation_{seed}.json')
            with torch.no_grad():
                for c,ds in datasets.items():
                    for x,y,ids in DataLoader(ds,batch_size=2,num_workers=2):valwaves[c][j,ids.numpy()]=forward_wave(model,x.cuda()).cpu().numpy()
            print('COLLECTED',family,seed,flush=True)
        group={'validation':{},'train':{}}
        for partition,conditions in [('validation',CONDS),('train',['clean','h264_500k'])]:
            for c in conditions:
                if partition=='validation':waves=valwaves[c];rows=datasets[c].rows
                else:
                    waves=np.stack([[tw[f'{i}_{family}_{s}_{c}'] for i in range(161)] for s in SEEDS]);rows=train_rows
                    for s in SEEDS:assert train_protocol['checkpoints'][f'{family}_{s}']['sha256']==frozen[family][str(s)]['sha256']
                reports,powers=make_reports(waves,rows)
                for s in SEEDS:
                    old=prior[s]['conditions' if partition=='validation' else 'train_conditions'][c]
                    prev={(r['subject'],r['window']):r for r in old['records']}
                    for r in reports[str(s)]['records']:
                        oldrow=prev[(r['subject'],r['window'])]
                        assert r['pred_hr']==oldrow['pred_hr'] and r['gt_hr']==oldrow['gt_hr'] and r['start_s']==oldrow['start_s']
                    assert abs(reports[str(s)]['subject_macro_MAE']-old['subject_macro_MAE'])<1e-8
                group[partition][c]=reports
                np.savez_compressed(dest/f'{family}_{partition}_{c}.npz',waves=waves,normalized_psd=powers)
                print('EVALUATED',family,partition,c,flush=True)
        allresults[family]=group;write(f'{family}_results.json',group)
        timings[family]=benchmark(models,datasets['clean']);write('timing.json',timings);del models;torch.cuda.empty_cache()
    write('results.json',{'groups':allresults,'protocol_sha256':sha(pp),'all_single_predictions_reproduced':True,'test_evaluated':False})
    write('environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,
      'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(),
      'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu,driver_version','--format=csv'],text=True),
      'cpu_threads':torch.get_num_threads()})
    write('COMPLETE.json',{'complete':True,'groups':2,'frozen_models':6,'val_windows':42,'val_conditions':4,'train_windows':161,'train_conditions':2,'all_single_predictions_reproduced':True,'test_evaluated':False})
    print('ALL ENSEMBLE EVALUATIONS COMPLETE',flush=True)
if __name__=='__main__':main()
