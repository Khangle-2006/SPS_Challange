"""Fixed five-fold development protocol; all samples exclude the original test subjects."""
import json
from pathlib import Path
import numpy as np
from train_efficientphys import Clips as OriginalClips,sha
ROOT=Path('/data_pool/khangle/facevitals');DATA=ROOT/'datasets/UBFC-rPPG'
CLEAN=DATA/'efficientphys36_trainval_v1'
WORK=DATA/'cv_5fold_20261020';OUT=ROOT/'results/cv_5fold_20261020'
SEED=2027;FAMILIES=['mse','spectral'];CONDITIONS=['clean','h264_500k','h264_250k','h264_1000k']
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(obj,indent=2))
class CVClips(OriginalClips):
    def __init__(self,root,rows):
        self.root=root;self.rows=rows;self.cache={};self.invalid=set()
        excluded=set(read(WORK/'folds.json')['original_test_subjects']);assert not excluded&{r['subject'] for r in rows}
        for i,r in enumerate(rows):
            m=np.load(root/(r['subject']+'_meta.npz'))
            if 'valid' not in m:continue
            q=r['start_s']+np.arange(301)/30;t=m['t']
            if q[0]<t[0] or q[-1]>t[-1]:self.invalid.add(i);continue
            j=np.clip(np.searchsorted(t,q,side='right'),1,len(t)-1);w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
            if not np.all(((w>=1)|m['valid'][j-1])&((w<=0)|m['valid'][j])):self.invalid.add(i)
    def __getitem__(self,i):
        if i in self.invalid:raise ValueError('Unusable input must be explicitly scored as invalid at evaluation')
        return super().__getitem__(i)
def rows_for(fold,role):
    p=read(WORK/'folds.json');allowed=set(p['folds'][fold][role+'_subjects'])
    return [{**r,'original_split':r['split'],'split':'cv_'+role,'fold':fold} for r in p['rows'] if r['subject'] in allowed]
def roots():return {'clean':WORK/'cache/clean',**{f'h264_{r}k':WORK/f'cache/detected_{r}k' for r in [500,250,1000]}}
