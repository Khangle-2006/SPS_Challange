import numpy as np,json
from core import GRID,top3,permute_blocks,stats
from pathlib import Path
s=sum(a*np.exp(-.5*((GRID-h)/.4)**2) for a,h in [(5,72),(4,74),(3,96),(2,150)])
c=top3(s);assert c==[72.,96.,150.] and min(abs(a-b) for i,a in enumerate(c) for b in c[i+1:])>=6
assert top3(np.zeros(len(GRID)))==[] and top3(np.ones(len(GRID)))==[]
rows=[dict(person=p) for p in ['a','a','b','b','c','d']];y=np.array([60,70,80,90,100,110.])
yy,m,st=permute_blocks(rows,y,np.random.default_rng(0));assert sorted(yy)==sorted(y)
for p,d in m.items():assert [yy[i] for i,r in enumerate(rows) if r['person']==p]==[y[i] for i,r in enumerate(rows) if r['person']==d]
out,pe=stats(rows,[[float(v)] for v in y],y);assert out['oracle_MAE']==0 and out['within5']==1
out,pe=stats(rows,[[] for v in y],y);assert out['oracle_MAE'] is None and out['within5']==0
Path(__file__).with_name('TESTS.json').write_text(json.dumps(dict(passed=True,checks=['peak_spacing_and_ranking','zero_flat_abstention','whole_person_label_blocks','label_multiset_preserved','oracle_exact_match','missing_candidates_explicit'])))
print('6 CPU tests PASS')
