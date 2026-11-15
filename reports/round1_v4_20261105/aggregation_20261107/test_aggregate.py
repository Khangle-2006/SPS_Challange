import unittest
import numpy as np
from aggregate import person_seed_rows,person_means,summary,METRICS,CONDITIONS
class AggregationTests(unittest.TestCase):
    def test_equal_person_weight_not_window_weight(self):
        rows=[]
        for seed in (2027,2028,2029):
            for sid,n,err in [('A',10,2),('B',1,12)]:
                for i in range(n):rows.append(dict(subject_id=sid,condition='MCD native',branch='v4_repaired',fold=0,seed=seed,accepted=True,reference_eligible=True,prediction_hr_bpm=70+err,reference_hr_bpm=70))
        p=person_means(person_seed_rows(rows));self.assertEqual(np.mean([v['MAE'] for v in p.values()]),7)
    def test_missing_seed_rejected(self):
        with self.assertRaises(AssertionError):person_means([dict(person='A',condition='X',branch='b',fold=0,seed=2027,**{m:1 for m in METRICS})])
    def test_identical_paired_bootstrap_zero(self):
        p={}
        for c in CONDITIONS:
            for s in ('A','B'):
                for b in ('v4_repaired','v3_joint'):
                    p[s,c,b]={m:(1 if s=='A' else 3) for m in METRICS}
        out=summary(p,('v4_repaired','v3_joint'),False)
        self.assertEqual(out['paired_V4_minus']['v3_joint']['S']['CI95'],[0.,0.])
        self.assertEqual(out['branches']['v4_repaired']['S']['value'],2.)
    def test_p95_is_mean_of_seed_metrics_not_prediction_ensemble(self):
        rr=[dict(person='A',condition='X',branch='b',fold=0,seed=s,**{m:v for m in METRICS}) for s,v in [(2027,1),(2028,3),(2029,8)]]
        self.assertEqual(person_means(rr)['A','X','b']['P95_abs_error'],4)
if __name__=='__main__':unittest.main()
