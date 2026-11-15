"""Minimal input check: exact static clips cannot support a temporal pulse measurement."""
import numpy as np

def check_temporal_input(frames):
    x=np.asarray(frames)
    if x.ndim!=4 or len(x)<2 or not np.isfinite(x).all():
        return {'usable':False,'reason':'INVALID_VIDEO_ARRAY'}
    if not np.any(x[1:]!=x[:-1]):
        return {'usable':False,'reason':'NO_TEMPORAL_VARIATION'}
    return {'usable':True,'reason':'TEMPORAL_VARIATION_PRESENT_NOT_A_PHYSIOLOGICAL_VALIDITY_GUARANTEE'}

def guarded_prediction(frames,predict_fn):
    quality=check_temporal_input(frames)
    if not quality['usable']:
        return {'hr_bpm':None,'quality':quality}
    return {'hr_bpm':predict_fn(frames),'quality':quality}
