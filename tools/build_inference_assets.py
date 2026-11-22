"""Maintainer-only export of fixed weights/reader parameters. No fitting or dataset reads."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

for k, v in {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}.items():
    os.environ[k] = v

sys.dont_write_bytecode = True

EXPECTED = [
    'e0315095fda595be37e4dc3ce3b693016111cfe7da81ef924a839c5ddf32ab0c',
    'f7ada0c3249624bbe664700f76e12e137a9d21afbd7aac9e0ed5ec3683baa3d4',
    '8a21a1f53d4552a02d2deeb1ce4a9ce8eaea98b84c180b63059d4ed9eab64d70',
    '8277e569511d11f1181e09354ba820af16a03a65d74b8e8985cbd5569c8ac141',
    '1f38cb4e16c0bc41602689b614191dc8756ebf1d4d28c4ae74568ce6ea66d88a',
]
P4_SHA = '1acb07a3a43c84a8d2eeacf1d96e08cb806b5ae97914e7493df6a4e821f58b70'
REPO = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, out = a.project_root.resolve(), a.output.resolve()
    if out.exists(): raise ValueError('Refusing to overwrite an existing export')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    frozen = json.loads((root/'reports/round1_execution_20261110/inference_repair_20261113/FROZEN.json').read_text())
    integration = root/'reports/submission_integration_20261122'
    btc_source = integration/'step34/BTC_REFERENCE_FOLD_MODELS.json'
    btc = json.loads(btc_source.read_text())
    assert set(btc) == set(map(str, range(5)))
    for m in btc.values():
        assert set(m) == {'prior', 'beta', 'off'}
        assert len(m['prior']) == 277 and set(m['beta']) == {'0','1'} and set(m['off']) == {'0','1','2','3'}
        assert np.isfinite(m['prior']).all() and min(m['prior']) >= 0
    p4_source = root/'reports/claude_research_20261119_constraints/out/z14_p4m_frozen.npz'
    assert sha(p4_source) == P4_SHA
    for f in range(5):
        src = Path(frozen['padding_checkpoints'][str(f)]['path'])
        assert sha(src) == EXPECTED[f] == frozen['padding_checkpoints'][str(f)]['sha256']
    # Only the whitelisted reader parameter arrays are accessed; never load truth/IDs/predictions.
    p4 = {}
    with np.load(p4_source, allow_pickle=False) as z:
        for tag in [f'fold{f}' for f in range(5)] + ['deploy']:
            for key, shape in [(f'w_{tag}', (4,)), (f'ro_{tag}_UBFC2', (6,)), (f'ro_{tag}_MCD', (6,))]:
                v = z[key]
                assert v.shape == shape and np.isfinite(v).all()
                p4[key] = v.tolist()
    out.mkdir(parents=True)
    bundle = out/'facevitals-inference-assets-20261123'
    (bundle/'code').mkdir(parents=True)
    (bundle/'checkpoints').mkdir()
    (bundle/'readers').mkdir()
    provenance = []
    sources = {
        'EfficientPhys_fixed.py': REPO/'ops/t8_polarity_v3_20261102/EfficientPhys_fixed.py',
        'replicate_tsm.py': REPO/'reports/round1_v4_20261105/padding_pilot_20261107/replicate_tsm.py',
        **{n: integration/'pipeline'/n for n in ['reader_v4.py', 'reader_models_fixed.py', 'btc_spectral_math.py', 'public_fixed_math.py']},
    }
    for name, src in sources.items():
        dst = bundle/'code'/name
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        provenance.append({'component': 'code/'+name, 'source_sha256': sha(src)})
    sys.path.insert(0, str(bundle/'code'))
    from EfficientPhys_fixed import EfficientPhys
    from replicate_tsm import use_replicate_padding
    checks = []
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((1,301,3,36,36), generator=gen)
    for f in range(5):
        src = Path(frozen['padding_checkpoints'][str(f)]['path'])
        original = torch.load(src, map_location='cpu', weights_only=True)
        state = original['model']
        assert original['epoch'] == 30
        assert all(isinstance(k, str) and isinstance(v, torch.Tensor) for k,v in state.items())
        target = bundle/'checkpoints'/f'padding_fixed_fold{f}_seed2027.pt'
        # Keep the original 'model' envelope for existing loaders; remove all training config.
        torch.save({'model': {k:v.detach().cpu().clone() for k,v in state.items()}, 'epoch':30}, target)
        exported = torch.load(target, map_location='cpu', weights_only=True)
        assert set(exported) == {'model','epoch'} and set(exported['model']) == set(state)
        assert all(torch.equal(v, exported['model'][k]) for k,v in state.items())
        model = use_replicate_padding(EfficientPhys(frame_depth=10, img_size=36)).eval()
        model.load_state_dict(state, strict=True)
        with torch.inference_mode(): old = model(x).clone()
        model.load_state_dict(exported['model'], strict=True)
        with torch.inference_mode(): new = model(x); static = model(torch.zeros_like(x))
        assert torch.equal(old,new) and torch.isfinite(new).all() and new.shape == (1,300)
        static_std = float(static.double().std(unbiased=False))
        assert torch.isfinite(static).all() and static_std < 1e-10
        checks.append({'fold':f,'state_tensors':len(state),'state_equal':True,'synthetic_waveform_max_abs_delta':0.0,'static_waveform_std':static_std})
        provenance.append({'component':str(target.relative_to(bundle)).replace('\\','/'), 'source_sha256':sha(src),
                           'transformation':'Keep model tensors and epoch only; training config removed; tensor equality verified.'})
        print(f'fold{f}: weights-only export and synthetic parity PASS',flush=True)
    save(bundle/'readers/btc_reader_v4_folds.json', btc)
    save(bundle/'readers/public_p4m_parameters.json', p4)
    # Check exact JSON round trip before publication.
    assert json.loads((bundle/'readers/btc_reader_v4_folds.json').read_text()) == btc
    with np.load(p4_source, allow_pickle=False) as z:
        for k,v in json.loads((bundle/'readers/public_p4m_parameters.json').read_text()).items():
            assert np.array_equal(v,z[k])
    # Synthetic readout checks: source vs exported parameters, all folds and flag cells.
    import reader_v4 as vr
    import public_fixed_math as pm
    rng = np.random.default_rng(0)
    zz = rng.normal(size=(8,277)); cl = np.tile([0,0,1,1],2); task = np.tile([False,True],4)
    def convert(m):return dict(prior=np.asarray(m['prior']), beta={int(k):v for k,v in m['beta'].items()},off={int(k):v for k,v in m['off'].items()})
    for f in btc:
        assert np.array_equal(vr.predG(convert(btc[f]),zz,cl,cl,task),vr.predG(convert(json.loads((bundle/'readers/btc_reader_v4_folds.json').read_text())[f]),zz,cl,cl,task))
    e = rng.normal(size=(5,len(pm.BPM)))
    for tag in [f'fold{f}' for f in range(5)]+['deploy']:
        for ds in ['UBFC2','MCD']:
            gamma,sigma,tau,est,ref,_ = p4[f'ro_{tag}_{ds}']
            pred = pm.point(pm.marg(pm.qscale(e,gamma,ref),sigma,tau,np.zeros(len(pm.BPM))), 'median' if est == 1 else 'mean')
            assert len(pred)==5 and np.isfinite(pred).all()
    provenance.extend([{'component':'readers/btc_reader_v4_folds.json','source_sha256':sha(btc_source),'transformation':'Only prior/beta/offset parameters of existing five train-fold fits; no refit.'},
                       {'component':'readers/public_p4m_parameters.json','source_sha256':P4_SHA,'transformation':'Only 18 w_/ro_ parameter arrays; excludes all labels, IDs and predictions.'}])
    save(bundle/'CONFIG.json', {'schema_version':1,'status':'research_inference_assets_not_end_to_end_release',
        'cnn':{'architecture':'EfficientPhys_fixed','frame_depth':10,'img_size':36,'temporal_padding':'replicate','seed':2027,'epoch':30,'input_shape':[1,301,3,36,36],'input_normalization':'one global mean/std over raw clip; reject raw.std < 1e-6','output':'300 temporal differences; cumsum before readout'},
        'btc':{'model':'reader_v4','folds':[0,1,2,3,4],'all_dev_deploy_model_available':False,'grid_bpm':{'start':42,'stop':180,'step':0.5},'class':'original source stream bit_rate < 4000000 bps; do not use derivative bitrate/dimensions','task_flag':'activity starts with Task','offset_cell':'2 * source_class + int(task_flag)','evidence':'(Z_CHROM + Z_POS + Z_FacePhys + Z_ME_chunk + Z_ME_flow) / 5','zero_evidence':'zero Z gives prior median plus fitted class/activity offset; caller must mark abstain'},
        'public':{'model':'P4m','tags':[f'fold{f}' for f in range(5)]+['deploy'],'channel_order':['CNN','POS','CHROM','mean_experts'],'ro_order':['gamma','sigma','tau','median_is_1_else_mean','quality_reference','constant_baseline'],'grid':'60*rfftfreq(4096,1/30), frequency 0.7..3 Hz inclusive','prior_log_potential':0,'cnn_prefilter':'centred valid-weighted moving average k=ceil(fps/10 - 1e-3), before interpolation; see code/public_fixed_math.py:ma3','readout':'offline forward-backward over windows of ONE recording, never concatenate different people'},
        'experts':{'package':'open-rppg==0.1.1','models':['FacePhys.rlap','ME-chunk.pure','ME-flow.pure'],'included':False},
        'limitations':['No raw-video frontend or participant data bundled.','No BTC all-dev deployment fit; do not invent a model by averaging fold parameters.','New-frontend integration parity gates were not met; this export does not change that disposition.','OOF evaluation needs the private exact person/fold mapping; do not use deploy parameters for OOF scoring.']})
    shutil.copyfile(REPO/'ops/deep_hr/TOOLBOX_LICENSE.txt',bundle/'TOOLBOX_LICENSE.txt')
    shutil.copyfile(REPO/'docs/INFERENCE_ASSETS.md',bundle/'README.md')
    save(bundle/'VALIDATION.json',{'cpu_only':True,'participant_data_read':False,'checkpoint_checks':checks,'btc_parameter_roundtrip_exact':True,'p4m_parameter_roundtrip_exact':True,'btc_synthetic_flag_cells_checked':40,'p4m_synthetic_sequences_checked':12,'not_a_real_video_accuracy_test':True})
    files = [{'path':p.relative_to(bundle).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(bundle.rglob('*')) if p.is_file()]
    save(bundle/'MANIFEST.json',{'schema_version':1,'release':'inference-assets-20261123','files':files,'provenance':provenance})
    archive = out/'facevitals-inference-assets-20261123.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in files:
            assert hashlib.sha256(z.read(bundle.name+'/'+row['path'])).hexdigest()==row['sha256']
    digest = sha(archive)
    (out/'SHA256SUMS.txt').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
    save(out/'RELEASE.json',{'tag':'inference-assets-20261123','asset':archive.name,'bytes':archive.stat().st_size,'sha256':digest,'files':len(files)+1})
    print(json.dumps({'archive':str(archive),'sha256':digest,'bytes':archive.stat().st_size,'files':len(files)+1}))

if __name__=='__main__':main()
