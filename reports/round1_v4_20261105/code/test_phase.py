import sys,unittest
from pathlib import Path
import numpy as np
import torch
from phase_loss import phase_loss,correlations,grouped_draws
class LossTests(unittest.TestCase):
 def test_identical_affine_and_sign(self):
  torch.manual_seed(1);d=torch.randn(3,300,dtype=torch.float64)
  self.assertLess(float(phase_loss(d,d).max()),1e-12)
  self.assertLess(float(phase_loss(3*d,d).max()),1e-12)
  self.assertGreater(float(phase_loss(-d,d).min()),.5)
 def test_nonwrapping_lag(self):
  torch.manual_seed(2);y=torch.randn(2,300,dtype=torch.float64);p=torch.cat([y[:,4:],torch.ones(2,4)*1e5],-1)
  r=correlations(p,y);self.assertTrue(torch.allclose(r[:,10],torch.ones(2,dtype=torch.float64)))
  self.assertTrue(torch.equal(r.argmax(-1),torch.tensor([10,10])))
 def test_finite_collapsed_and_random_gradient(self):
  for zero in (False,True):
   p=(torch.zeros(2,300) if zero else torch.randn(2,300)).requires_grad_();y=torch.randn(2,300)
   phase_loss(p,y).mean().backward();self.assertTrue(torch.isfinite(p.grad).all())
   self.assertGreater(float(p.grad.abs().sum()),0)
 def test_gradcheck(self):
  torch.manual_seed(4);p=torch.randn(1,30,dtype=torch.double,requires_grad=True);y=torch.randn(1,30,dtype=torch.double)
  self.assertTrue(torch.autograd.gradcheck(lambda p:phase_loss(p,y),p,atol=1e-4))
 def test_grouped_examples_tail(self):
  def draw(pool,seed,fold,epoch,step,arm):return [('UBFC2',step,seed,epoch),('MCD',step,seed,epoch)]
  for batch in (4,8):
   groups=list(grouped_draws(None,2027,0,0,11,batch,draw))
   self.assertEqual([x for _,g in groups for x in g],[x for s in range(11) for x in draw(None,2027,0,0,s,'joint')])
   self.assertEqual(len(groups[-1][1]),2*(11%(batch//2) or batch//2))
if __name__=='__main__':
 torch.set_num_threads(2);unittest.main()
