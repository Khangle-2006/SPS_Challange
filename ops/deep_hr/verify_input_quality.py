import json
import numpy as np
from stress_ubfc import StressClips,ROOT
from input_quality import check_temporal_input,guarded_prediction
results={}
for split in ('train','val'):
    for condition in ('clean','frozen_frame'):
        ds=StressClips(split,condition);rejected=0
        for i in range(len(ds)):
            frames=ds[i][0].numpy()
            q=check_temporal_input(frames);rejected+=not q['usable']
            assert q['usable']==(condition=='clean')
        results[split+'_'+condition]={'n':len(ds),'rejected':rejected}
def must_not_call(x):raise AssertionError('Predictor must not be called for static input')
assert guarded_prediction(np.ones((301,3,36,36)),must_not_call)['hr_bpm'] is None
(ROOT/'results/stress_ubfc_v1/input_quality_verification.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
