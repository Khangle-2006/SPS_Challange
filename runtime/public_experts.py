"""Three frozen experts, reset and self-warm on this video only; preserve raw waveforms."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
from common import save,verify_weights
from reader_v4 import MODELS


def main(out):
    hashes=verify_weights();front=json.loads((out/'frontend.json').read_text())
    if front['static_input']:
        save(out/'experts.json',dict(abstain=True,reason='static_input',weight_sha256=hashes));return
    import rppg
    with np.load(out/'trace.npz') as z:t=z['t']
    crop=np.load(out/'crop72.npy',mmap_mode='r');fps=(len(t)-1)/(t[-1]-t[0]);waves={}
    for name in MODELS:
        print(name,flush=True);model=rppg.Model(name);initial=copy.deepcopy(model.state)
        model.state=copy.deepcopy(initial)
        for _ in range(2):model.process_faces_tensor(np.ascontiguousarray(crop),fps=float(fps))
        bvp,ts=model.bvp(raw=True);ts=np.asarray(ts,float);ts=ts-ts[0]+t[0]
        waves[name]=np.asarray(bvp,float);waves[name+'_t']=ts;del model
    np.savez_compressed(out/'expert_waves.npz',**waves)
    save(out/'experts.json',dict(abstain=False,weight_sha256=hashes,mode='reset then two same-video passes'))


if __name__=='__main__':main(Path(sys.argv[1]))
