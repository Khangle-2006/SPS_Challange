"""Frozen three experts, sequential CPU inference with the original spectrum operator."""
import json
import copy
import sys
from pathlib import Path
import numpy as np
from common import save, verify_weights
from btc_spectral_math import supported, window_spectrum, GRID
from reader_v4 import MODELS


def spectrum(model, crops, t, valid):
    # open-rppg 0.1.1 clears frame buffers, but keeps recurrent state between
    # process_faces_tensor calls. Never let another video's state enter this one.
    if not hasattr(model, '_facevitals_initial_state'):
        model._facevitals_initial_state=copy.deepcopy(model.state)
    model.state=copy.deepcopy(model._facevitals_initial_state)
    fps=(len(t)-1)/(t[-1]-t[0])
    model.process_faces_tensor(np.ascontiguousarray(crops),fps=float(fps))
    bvp,ts=model.bvp(raw=True)
    if len(bvp)<300: return None,0
    bvp=np.asarray(bvp,float);ts=np.asarray(ts,float);ts-=ts[0];powers=[]
    for s in np.arange(0,max(0.,t[-1]-10+1e-8),5.):
        if not supported(t,valid,s,s+10) or s+10>ts[-1]+.2: continue
        w=np.interp(s+np.arange(300)/30.,ts,bvp)
        if np.isfinite(w).all() and w.std()>1e-12: powers.append(window_spectrum(w))
    return (np.mean(powers,0),len(powers)) if powers else (None,0)


def main(out):
    hashes=verify_weights()
    front=json.loads((out/'frontend.json').read_text())
    with np.load(out/'trace.npz') as z: t=z['t'];valid=z['valid']
    any_window=any(supported(t,valid,s,s+10) for s in np.arange(0,max(0.,t[-1]-10+1e-8),5.))
    if front['static_input'] or not any_window:
        save(out/'experts.json',{'abstain':True,'reason':'static_input' if front['static_input'] else 'no_supported_window','weight_sha256':hashes})
        return
    import rppg
    crops=np.load(out/'crop72.npy',mmap_mode='r');spectra={};counts={}
    for name in MODELS:
        print('model',name,flush=True)
        model=rppg.Model(name);s,n=spectrum(model,crops,t,valid);counts[name]=n
        spectra[name]=s if s is not None else np.full(len(GRID),np.nan)
        del model
    np.savez_compressed(out/'experts.npz',**spectra)
    save(out/'experts.json',dict(windows=counts,weight_sha256=hashes,abstain=not all(counts.values())))


if __name__=='__main__': main(Path(sys.argv[1]))
