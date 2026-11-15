import unittest,numpy as np
from pipeline import *
class Contract(unittest.TestCase):
 def test_static_abstains(self):
  for fs in [10,19.99,29.97,240]:
   x=np.tile([170,120,90],(int(30*fs),1))
   for m in ['POS','CHROM','G']:self.assertEqual(peak_info(*psd(pulse(x,fs,m),fs)),[])
 def test_known_pulse(self):
  for fs in [10,19.99,29.97,240]:
   t=np.arange(int(30*fs))/fs;s=np.sin(2*np.pi*1.2*t)
   rgb=np.array([170,120,90])[None,:]*(1+s[:,None]*np.array([.0025,.01,.001])[None,:])
   for m in ['POS','CHROM','G']:self.assertLess(abs(peak_info(*psd(pulse(rgb,fs,m),fs))[0]['hr']-72),1.)
 def test_native_clock(self):
  t=np.arange(600)/19.99;x=np.tile([170.,120.,90.],(600,1));q,_,fs=native_grid(t,x)
  self.assertAlmostEqual(fs,19.99);self.assertLessEqual(len(q),600)
  x[100:120]=np.nan
  with self.assertRaises(ValueError):native_grid(t,x)
 def test_person_balance(self):
  x=[dict(person='a')]*8+[dict(person='b')]*2;w=person_weights(x)
  self.assertAlmostEqual(w[:8].sum(),.5);self.assertAlmostEqual(w[8:].sum(),.5)
 def test_prior_and_no_candidates(self):
  r=[dict(person=str(i),truth=80+i,candidates=[]) for i in range(10)]
  model=fit(r);p=predict([],model)
  self.assertFalse(p['pulse_observed']);self.assertGreaterEqual(p['hr'],80);self.assertLessEqual(p['hr'],89)
  self.assertAlmostEqual(p['posterior'].sum(),1.);self.assertLess(p['lower90'],p['upper90'])
 def test_heldout_labels_absent(self):
  p=dict(hr=72,hz=1.2,weight=1,power=1)
  c=dict(roi='forehead',method='POS',features=[10,1,.9,-1],peaks=[p])
  train=[dict(person=str(i),truth=72+i,candidates=[c]) for i in range(8)]
  m=fit(train);self.assertEqual(m['training_people'],list(map(str,range(8))))
  pred=predict([c],m);self.assertTrue(np.isfinite(pred['posterior']).all());self.assertLessEqual(m['b'],0)
if __name__=='__main__':unittest.main()
