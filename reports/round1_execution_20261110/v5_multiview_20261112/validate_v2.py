"""Real cache/target/fold validation before any V5 GPU job."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
import sys,traceback
import numpy as np
from runtime import *
from provider import SideProvider,DiskProvider
os.environ['CUDA_VISIBLE_DEVICES']=''
def main(fold):
    receipt_path=P/f'VALIDATION_FOLD{fold}.json';assert not receipt_path.exists();ready=read(P/f'FOLD{fold}_READY.json');plan=read(P/'TRAIN_PLAN_EARLY.json');index=read(old.p.INDEX);tests=[]
    assert ready['source_jobs_sha256']==sha(P/'AUG_JOBS_FROZEN.json') and ready['all_side_cache_hashes_verified']
    sys.path.insert(0,'/data_pool/khangle/facevitals/code');from mcd_errata_audit import timed
    side=read(P/'SIDE_INDEX.json')
    for r in side:
        off=read(P/'offsets_v2'/f"{r['recording_id']}.json");assert off['eligible'] and off['fold']==r['fold']
        raw=off['raw_row'];_,mt=timed(R/'datasets/MCD-rPPG'/raw['meta']);pv,pt=timed(R/'datasets/MCD-rPPG'/raw['ppg']);u,inv,cnt=np.unique((pt-mt[0])/1e6+r['offset_seconds'],return_inverse=True,return_counts=True);expected=np.zeros(len(u));np.add.at(expected,inv,-pv);expected/=cnt
        with np.load(P/'side_native'/r['recording_id']/'native/crop/meta.npz') as z:
            np.testing.assert_allclose(z['ppg_t'],u,rtol=0,atol=1e-10);np.testing.assert_allclose(z['ppg'],expected,rtol=0,atol=1e-12)
        tests.append(dict(recording=r['recording_id'],raw_sign_time_mean_exact=True))
    for run in [ready['run']]:
        f=run['fold'];spec=index['folds']['folds'][f];provider=SideProvider(run['btc_bpp_quartiles']);public=DiskProvider(index,run['btc_bpp_quartiles'],300_000_000)
        assert set(run['btc_train_people']).isdisjoint(run['btc_val_people'])
        for ds,prefix in [('MCD','mcd'),('UBFC2','ubfc')]:
            byid={r['recording_id']:r['subject_id'] for r in index['records'] if r['dataset']==ds}
            assert all(byid[j['recording']] in spec[prefix+'_train_subjects'] for j in run['public_jobs'][ds])
        assert all(j['person'] in spec['mcd_train_subjects'] and j['person'] not in spec['mcd_val_subjects'] for j in run['side_jobs'])
        # Frozen representative all codecs, each view, and min/max resampling.
        selections=[]
        for camera in ['IriunWebcam','USBVideo']:
            for recipe in ['hevc','h264','c5_proxy']:
                js=[j for j in run['side_jobs'] if j['camera']==camera and j['recipe']==recipe]
                if js:selections.append(js[0])
        for j in selections:
            for rate in j['rates']:
                x,y,rec=provider.clip('MCD_SIDE',j['recording'],j['start'],j['recipe'],j['lowres'],f,rate)
                _,m=provider.metadata('MCD_SIDE',j['recording']);q=j['start']+np.arange(301)/30*rate;target=np.diff(np.interp(q,m['ppg_t'],m['ppg']).astype(np.float32));target/=target.std();np.testing.assert_array_equal(y,target);assert np.isfinite(x).all()
        # Read one reused public sample in each domain, without any encoding.
        for ds in ['UBFC2','MCD']:
            j=run['public_jobs'][ds][0];public.clip(ds,j['recording'],j['start'],j['recipe'],j['lowres'],f,j['rates'][0])
        tests.append(dict(fold=f,people_disjoint=True,real_side_codec_samples=len(selections),min_max_time_target_parity=True,public_reuse_passed=True))
    # Batch-8 finite CPU forward/backward with the real new view targets.
    import torch
    from EfficientPhys_fixed import EfficientPhys
    from replicate_tsm import use_replicate_padding
    from repair_loss import phase_loss
    from test_static_tsm import static_result
    torch.set_num_threads(2);torch.manual_seed(2027);model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));xs=[];ys=[]
    run=ready['run'];provider=SideProvider(run['btc_bpp_quartiles'])
    for j in run['side_jobs'][:8]:
        x,y,_=provider.clip('MCD_SIDE',j['recording'],j['start'],j['recipe'],j['lowres'],fold,1. if j['rates'][1]>=1 else j['rates'][0]);xs.append(x);ys.append(y)
    assert len(xs)==8;loss=phase_loss(model(torch.from_numpy(np.stack(xs))),torch.from_numpy(np.stack(ys))).mean();assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    static=static_result(model);assert static['hr'] is None
    save(receipt_path,dict(passed=True,tests=tests,finite_cpu_batch8_backward_loss=float(loss.detach()),static=static,plan_sha256=sha(P/'TRAIN_PLAN_EARLY.json'),ready_sha256=sha(P/f'FOLD{fold}_READY.json'),code_sha256=sha(__file__),holdout_evaluations=0,validation_is_not_training=True))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);args=ap.parse_args()
    try:main(args.fold)
    except Exception as e:save(P/'VALIDATION_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
