import unittest
import numpy as np
from decode_hr_v2 import decode, decode_sequence, ablation_config, fold_frequency
from hr_decoder import estimate_hr


class DecoderTests(unittest.TestCase):
    def wave(self, fs, hz=1.5, seed=17):
        t=np.arange(int(10*fs))/fs
        return (np.sin(2*np.pi*hz*t)+.3*np.sin(4*np.pi*hz*t)+
                .1*np.random.default_rng(seed).normal(size=len(t)))[None]

    def test_sine_10_and_30(self):
        for fs in (10,30):
            r=decode(self.wave(fs),fs)
            self.assertLess(abs(r['hr_bpm']-90),2)
            self.assertFalse(r['abstain'])

    def test_fold(self):
        np.testing.assert_allclose(fold_frequency([5,5.5,6,19],10),[5,4.5,4,1])
        self.assertAlmostEqual(float(fold_frequency(6,30)),6)

    def test_legacy_and_single_region_parity(self):
        w=self.wave(30)
        for stage,band in [('legacy',(.75,3)),('a',(.7,3)),('b',(.7,3))]:
            r=decode(w,30,config=ablation_config(stage))
            self.assertEqual(r['hr_bpm'],estimate_hr(w[0],30,band))

    def test_stable_hmm(self):
        windows=np.array([self.wave(30,seed=i) for i in range(12)])
        r=decode_sequence(windows,30)
        self.assertLessEqual(np.abs(np.diff([x['hr_bpm'] for x in r])).max(),10)

    def test_causal_prefix_independence(self):
        windows=np.array([self.wave(30,hz=x) for x in [1.5,1.5,1.55,2,2]])
        a=decode_sequence(windows,30)
        b=decode_sequence(windows[:3],30)
        np.testing.assert_array_equal([x['hr_bpm'] for x in a[:3]],[x['hr_bpm'] for x in b])
        for mode in ['offline','fixedlag1']:
            r=decode_sequence(windows,30,mode=mode)
            self.assertEqual(len(r),5)
            self.assertTrue(all(np.isfinite(x['hr_bpm']) for x in r))
        self.assertEqual(decode_sequence(windows,30,mode='fixedlag1')[-1]['lookahead_windows'],0)

    def test_invalid_regions(self):
        w=np.concatenate([self.wave(30),self.wave(30)])
        mask=np.ones_like(w,bool);mask[0,10]=False
        self.assertEqual(decode(w,30,mask)['valid_regions'],[1])
        r=decode(w,30,np.zeros_like(mask))
        self.assertTrue(r['abstain']);self.assertTrue(np.isnan(r['hr_bpm']))

    def test_multi_region_label_free_weights(self):
        w=np.concatenate([self.wave(30),np.random.default_rng(33).normal(size=(1,300))])
        r=decode(w,30)
        self.assertGreater(r['region_weights'][0],r['region_weights'][1])
        self.assertLess(abs(r['hr_bpm']-90),2)


if __name__=='__main__':unittest.main()
