import unittest
from evaluate import summarize
class Metrics(unittest.TestCase):
 def test_paired_person_weighting(self):
  a=[dict(recording=str(i),person='a',truth=80,prediction=90) for i in range(4)]+[dict(recording='b',person='b',truth=80,prediction=80)]
  b=[dict(r,prediction=85) for r in a]
  self.assertEqual(summarize(a)['MAE'],5)
  self.assertEqual(summarize(a,b)['MAE'],0)
if __name__=='__main__':unittest.main()
