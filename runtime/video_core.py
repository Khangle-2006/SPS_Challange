"""Apply fixed reader parameters; no labels, no fitting, no private manifests."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from common import save, load_model, sha, ROOT
from btc_spectral_math import avg_spectra, GRID
from reader_v4 import MODELS,evidence,predG


def main(out):
    req=json.loads((out/'request.json').read_text());front=json.loads((out/'frontend.json').read_text())
    experts=json.loads((out/'experts.json').read_text());m,msha=load_model(req['fold'])
    low=np.array([int(front['capture_class_low'])]);task=np.array([req['activity']=='Task'])
    why=experts.get('reason');a=None;Z=np.zeros((1,len(GRID)))
    if not experts.get('abstain',False):
        with np.load(out/'trace.npz') as z: a=avg_spectra(z['t'],z['valid'],z['rgb_mean'])
        if a is None: why='no_valid_classical_spectrum'
        else:
            with np.load(out/'experts.npz') as z: S={k:z[k][None] for k in MODELS}
            Z=evidence(a[0][None],a[1][None],S)
            if not np.isfinite(Z).all(): why='invalid_expert_spectrum';Z[:]=0
    else: why=why or 'no_valid_expert_spectrum'
    pred=float(predG(m,Z,low,low,task)[0]);abstain=why is not None
    # The prior fallback is explicitly NOT reported as a measured pulse.
    result=dict(method='BTC reader_v4',pr_bpm=None if abstain else pred,
        fallback_prior_bpm=pred if abstain else None,abstain=abstain,reason=why,
        activity=req['activity'],capture_class='low' if low[0] else 'high',
        frontend=req['frontend'],model_use='deployment' if req['fold'] is None else f'dev_oof_fold{req["fold"]}',
        reader_sha256=msha,expert_weight_sha256=experts['weight_sha256'],
        beta=float(m['beta'][int(low[0])]),
        uses_spectral_evidence=bool(not abstain and m['beta'][int(low[0])]>0),
        classical_windows=0 if a is None else a[2],expert_windows=experts.get('windows',{}),
        valid_frame_fraction=front['valid_frames']/front['frames'],frames=front['frames'],
        video_sha256=front['video_sha256'],
        runtime_sha256={p.name:sha(p) for p in sorted((ROOT/'runtime').glob('*.py'))})
    save(out/'result.json',result)
    with (out/'result.csv').open('w',newline='') as f:
        fields=['method','pr_bpm','abstain','fallback_prior_bpm','reason','activity','capture_class','frontend','model_use']
        w=csv.DictWriter(f,fields);w.writeheader();w.writerow({k:result[k] for k in fields})
    print(json.dumps(result,indent=2))


if __name__=='__main__': main(Path(sys.argv[1]))
