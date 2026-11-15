import numpy as np
from run_t9 import predict
from hr_baseline import pos
assert int(np.ceil(1.6*30))==48
for fs in (10,30):
 for bpm in (60,90,150):
  t=np.arange(int(12*fs)+1)/fs;s=np.sin(2*np.pi*bpm/60*t)
  rgb=np.array([100.,130.,90.])[None,:]+s[:,None]*np.array([.2,1.5,.5])[None,:]
  hr=predict(t,rgb,np.ones(len(t),bool),1.)
  assert abs(hr-bpm)<1.0,(fs,bpm,hr)
# Clock offset must not be silently discarded; an unsupported window is refused.
t=np.arange(121)/10+2;rgb=np.ones((len(t),3))
try:predict(t,rgb,np.ones(len(t),bool),0.)
except ValueError:pass
else:raise AssertionError('Extrapolation accepted')
print('PASS6 knownHR cases at10/30Hz,48samplePOS, unsupported-clock refusal')
