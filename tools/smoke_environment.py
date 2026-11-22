"""Synthetic-only environment checks; never loads participant data or private checkpoints."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

for k,v in {'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','KERAS_BACKEND':'jax',
            'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1',
            '__EGL_VENDOR_LIBRARY_DIRS':'/nonexistent'}.items():os.environ[k]=v
ROOT=Path(__file__).resolve().parents[1]

def run(profile):
    import numpy as np
    out={'profile':profile,'python':sys.version.split()[0],'prefix':sys.prefix,
         'base_prefix':sys.base_prefix,'synthetic_only':True}
    assert sys.prefix != sys.base_prefix, 'Use a fresh venv'
    if profile=='core':
        path=ROOT/'reports/claude_research_20261118/code/lib_btc.py'
        spec=importlib.util.spec_from_file_location('smoke_lib_btc',path)
        lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
        t=np.arange(901)/30.;s=np.sin(2*np.pi*1.5*t)
        rgb=np.stack([100+.1*s,120+s,80+.2*s],axis=1)
        spectra=lib.avg_spectra(t,np.ones(len(t),bool),rgb)
        assert spectra is not None
        peaks=[float(lib.GRID[x.argmax()]) for x in spectra[:2]]
        assert max(abs(p-90) for p in peaks)<=1,peaks
        with np.errstate(divide='ignore',invalid='ignore'):
            static=lib.avg_spectra(t,np.ones(len(t),bool),np.tile([100.,120.,80.],(len(t),1)))
        assert static is None
        out.update(pos_chrom_peaks=peaks,static_abstains=True)
    elif profile=='frontend':
        import av,cv2,mediapipe as mp
        cv2.setNumThreads(1)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1) as model:
            result=model.process(np.zeros((128,128,3),np.uint8))
            assert not result.multi_face_landmarks
        out.update(blank_image_faces=0,av_version=av.__version__,opencv=cv2.__version__,mediapipe=mp.__version__)
    elif profile=='torch-cpu':
        import torch
        torch.set_num_threads(1)
        assert not torch.cuda.is_available()
        assert torch.version.cuda is None,'Install the CPU wheel'
        x=torch.arange(12,dtype=torch.float32).reshape(3,4)
        assert float((x@x.T).sum())==1134.
        out.update(torch=torch.__version__,cuda_runtime=None)
    elif profile=='experts':
        import jax,rppg
        assert all(d.platform=='cpu' for d in jax.devices())
        # Exact official weight files, separate from the team's private CNN checkpoints.
        weights=Path(rppg.__file__).parent/'weights'
        expected={'FacePhys.rlap.weights.h5':'6cd99510bcc5771f9cebcdb0f47a5cb3692d19f1e3d6e2444af5e8fac0ee3c07',
                  'ME.pure.weights.h5':'2135995cb4161d1d4c7c3aa45d333dc6b1f5818b0652d11da9fc0fb0b6e11bf0'}
        hashes={name:hashlib.sha256((weights/name).read_bytes()).hexdigest() for name in expected}
        assert hashes==expected,'Weight mismatch: do not silently substitute a release'
        out['weights_sha256']=hashes
        results={}
        for name in ('FacePhys.rlap','ME-chunk.pure','ME-flow.pure'):
            model=rppg.Model(name)
            # Synthetic varying colors, no face images; exercises tensor inference, not physiological accuracy.
            t=np.arange(360)/30
            colors=np.stack([100+2*np.sin(2*np.pi*1.5*t),120+4*np.sin(2*np.pi*1.5*t),80+np.cos(2*np.pi*1.5*t)],1)
            crop=np.broadcast_to(colors[:,None,None,:],(len(t),72,72,3)).copy().astype(np.uint8)
            model.process_faces_tensor(crop,fps=30.)
            bvp,ts=model.bvp(raw=True)
            assert len(bvp)>0 and len(bvp)==len(ts) and np.isfinite(bvp).all()
            results[name]={'samples':len(bvp),'finite':True}
        out.update(models=results,jax=metadata.version('jax'),keras=metadata.version('keras'))
    out['passed']=True
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile',choices=['core','frontend','torch-cpu','experts'])
    p.add_argument('--receipt',type=Path)
    a=p.parse_args();r=run(a.profile)
    if a.receipt:
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        a.receipt.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r))
