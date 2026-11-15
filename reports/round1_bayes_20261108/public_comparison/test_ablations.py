import unittest,json
import numpy as np
from pipeline_frozen import trace_candidates
from features import derive
class DerivedAblations(unittest.TestCase):
    def test_exact_direct_pipeline_parity(self):
        t=np.arange(301)/30;rng=np.random.default_rng(14)
        rgb=100+rng.normal(0,.15,(301,4,3));rgb[:,:,1]+=2*np.sin(2*np.pi*1.2*t[:,None])
        original=trace_candidates(t,rgb,[1.2])
        for abl in ['none','no_background','no_codec','no_viterbi','constant_epsilon','no_prior','K1']:
            self.assertEqual(json.dumps(derive(original,abl),sort_keys=True),json.dumps(trace_candidates(t,rgb,[1.2],abl),sort_keys=True),abl)
if __name__=='__main__':unittest.main()
