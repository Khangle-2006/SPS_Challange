"""Imbalanced-person, missing-output and reference-mismatch regressions."""
from report_metrics import metric
rows=[]
for rid,person,error in [('a1','A',0),('a2','A',0),('a3','A',0),('b1','B',20)]:
 for method,offset in [('constant',5),('calibrated',2),('new',0)]:
  rows.append(dict(recording=rid,person=person,truth=80,prediction=80+error+offset,method=method))
out={r['method']:r for r in metric(rows,['constant','calibrated','new'])}
assert out['new']['MAE']==10 # Window/video-weighted mean would be 5.
assert out['new']['delta_calibrated']==-2 and out['new']['delta_calibrated_CI']==[-2,-2]
assert out['new']['people']==2 and out['new']['videos']==4
rows.append(dict(recording='missing',person='C',truth=80,prediction=120,method='constant'))
assert metric(rows,['constant','calibrated','new'])==list(out.values())
rows[0]['truth']=81
try:metric(rows,['constant','calibrated','new'])
except AssertionError:pass
else:raise AssertionError('Reference mismatch was silently paired')
print('PASS: person weighting, paired bootstrap, common cohort, fail-closed reference')
