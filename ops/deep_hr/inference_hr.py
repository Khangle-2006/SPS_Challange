"""HR inference for an already aligned 10-second 36px RGB clip, without labels."""
import numpy as np
import torch
from input_quality import check_temporal_input
from hr_decoder import estimate_hr, BAND_HZ

def predict_hr(model,frames,device='cuda',band=BAND_HZ):
    """Caller supplies 301 RGB frames on the exact 30Hz grid used in training.

    This is not a video decoder or face detector. A successful numerical output
    is not a calibrated signal-quality or clinical-validity certificate.
    """
    x=np.asarray(frames,dtype=np.float32)
    if x.shape!=(301,36,36,3):
        raise ValueError('Expected 301 aligned 36x36 RGB frames at 30Hz')
    quality=check_temporal_input(x)
    if not quality['usable']:return {'hr_bpm':None,'quality':quality}
    std=float(x.std())
    if std<1e-6:return {'hr_bpm':None,'quality':{'usable':False,'reason':'INSUFFICIENT_INPUT_VARIANCE'}}
    x=(x-float(x.mean()))/std
    tensor=torch.from_numpy(x.transpose(0,3,1,2).copy())[None].to(device)
    model.eval()
    with torch.no_grad():
        pred=model(tensor)
        if not torch.isfinite(pred).all():
            return {'hr_bpm':None,'quality':{'usable':False,'reason':'NONFINITE_MODEL_OUTPUT'}}
        wave=torch.cat([torch.zeros_like(pred[:,:1]),torch.cumsum(pred[:,:-1],dim=1)],dim=1)[0].cpu().numpy()
    try:hr=estimate_hr(wave,30.,band=band)
    except ValueError:return {'hr_bpm':None,'quality':{'usable':False,'reason':'INVALID_PREDICTED_SIGNAL'}}
    return {'hr_bpm':hr,'quality':quality,'band_hz':list(band)}
