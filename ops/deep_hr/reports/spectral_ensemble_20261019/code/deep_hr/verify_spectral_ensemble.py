"""Estimator equivalence and fusion invariants before evaluating real ensembles."""
import json
import numpy as np
from spectral_ensemble import normalized_psd,summarize_three
from hr_baseline import estimate_hr
def run():
    t=np.arange(300)/30
    for bpm in [48,72,120,168]:
        w=np.sin(2*np.pi*bpm/60*t);f,p=normalized_psd(w)
        assert float(f[np.argmax(p)])==estimate_hr(w,30)
        assert abs(p.sum()-1)<1e-12
        base=summarize_three(np.stack([w,-w,7*w]));assert abs(base['psd_mean_hr']-bpm)<.5
        np.testing.assert_allclose(base['psd'],np.tile(p,(3,1)),rtol=1e-7,atol=1e-10)
        other=summarize_three(np.stack([7*w,w,-w]))
        assert base['psd_mean_hr']==other['psd_mean_hr']
    for bad in [np.zeros(300),np.full(300,np.nan),np.ones(30)]:
        try:normalized_psd(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid waveform accepted')
    try:summarize_three(np.stack([w,w,np.zeros(300)]))
    except ValueError:pass
    else:raise AssertionError('An invalid member must not be silently dropped')
    return {'passed':True,'single_estimator_equivalence':True,'unit_band_power':True,'sign_and_scale_invariance':True,'seed_order_invariance':True,'invalid_member_rejected':True}
if __name__=='__main__':print(json.dumps(run()),flush=True)
