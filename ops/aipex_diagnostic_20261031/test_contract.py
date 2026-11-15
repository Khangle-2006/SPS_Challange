import math
import numpy as np
from diagnostic import angle, bin_ita
from hr_baseline import estimate_hr

assert abs(angle(70,20)-45)<1e-10
assert abs(angle(30,20)+45)<1e-10
assert angle(50,0) is None
assert [bin_ita(x) for x in (-30,10,28,41,55,56,None)] == ['<=-30','(-30,10]','(10,28]','(28,41]','(41,55]','>55','unknown']
for bpm in (60,90,150):
    t=np.arange(150)/15
    assert abs(estimate_hr(np.sin(2*np.pi*bpm/60*t),15)-bpm)<1
print('PASS ITA formula, missingness, bin boundaries, 15 Hz decoder known HR')
