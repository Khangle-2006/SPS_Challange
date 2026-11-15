import unittest
from summarize_dev import person_metrics,boot,calibration
class Metrics(unittest.TestCase):
    def test_equal_person_weight(self):
        r=[dict(person='a',recording=str(i),prediction=71,truth=70,fold=0) for i in range(10)]+[dict(person='b',recording='b',prediction=80,truth=70,fold=1)]
        self.assertEqual(boot(r)['MAE'],5.5)
        self.assertEqual(boot(r,r)['CI95'],[[0.,0.],[0.,0.],[0.,0.]])
    def test_pairing_requires_same_videos(self):
        a=[dict(person='a',recording='1',prediction=75,truth=70),dict(person='a',recording='2',prediction=100,truth=70)]
        b=[dict(person='a',recording='1',prediction=74,truth=70)]
        self.assertEqual(boot(a,b)['MAE'],1)
    def test_calibration_never_fits_validation(self):
        rows=[dict(person=str(i),recording=str(i),prediction=50+i,truth=2*(50+i)+3,fold=i%5) for i in range(15)]
        pred,fits=calibration(rows)
        for r in pred:self.assertAlmostEqual(r['prediction'],r['truth'],places=9)
        for f in fits:self.assertFalse({r['person'] for r in rows if r['fold']==f['fold']}&set(f['train_people']))
if __name__=='__main__':unittest.main()
