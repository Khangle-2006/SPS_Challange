import numpy as np
from metrics import person_metrics,aggregate,CONDITIONS
from spectral import snr
w=[]
for branch,error in [('ubfc_only',1.),('v3_joint',2.),('hybrid_v3_joint',.5)]:
    for c in CONDITIONS:
        ds='MCD' if c.startswith('MCD') else 'UBFC2';n=60 if ds=='MCD' else 35
        for k in range(n):
            for seed in (2027,2028,2029):
                for j in range(1 if k else 4):
                    w.append(dict(subject_id=ds+':'+str(k),dataset=ds,condition=c,branch=branch,fold=k%5,seed=seed,recording_id=str(k),window_start_s=j*10.,prediction_hr_bpm=70+error,reference_hr_bpm=70.,reference_eligible=True,accepted=True))
s=aggregate(person_metrics(w))
assert s['branches']['ubfc_only']['S']==dict(value=1.,CI95=[1.,1.])
assert s['paired']['v3_joint minus ubfc_only']['UBFC C5']['CI95']==[1.,1.]
assert s['gates']['hybrid_v3_joint']['pass_all']
assert s['recommended_branch']=='hybrid_v3_joint'
t=np.arange(300)/30
assert snr(np.zeros(300)) is None
assert snr(np.sin(2*np.pi*1.2*t))>snr(np.random.default_rng(0).normal(size=300))
assert np.isclose(snr(np.sin(2*np.pi*1.2*t)),snr(-3*np.sin(2*np.pi*1.2*t)))
print('PASS unequal person window counts, seed averaging, paired bootstrap, S/gate selection, SNR invalidity and sign/scale invariance')
