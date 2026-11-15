import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from simulate import select_frames,noise_float,c1_pixels,simulate,qp_summary,PROTOCOL
from run_t5 import supported_interval,merge_duplicates


class SimulationTests(unittest.TestCase):
    def test_timestamp_selection_jitter_duplicates(self):
        t=np.arange(301)/29.97;t[22]=t[21];t[55]+=.009
        i=select_frames(t,10)
        self.assertTrue(np.all(np.diff(i)>0));self.assertEqual(i[0],0)
        self.assertTrue(np.all(np.abs(t[i]-np.arange(len(i))/10)<.04))
        self.assertGreater(len(select_frames(t,30)),290)

    def test_noise_physics_and_determinism(self):
        x=np.full((1000,1000,1),128,np.uint8);gain=.25;mu=128/255*gain
        y=noise_float(x,gain,np.random.default_rng(7));expected=PROTOCOL['c1']['a']*mu+PROTOCOL['c1']['b']
        self.assertAlmostEqual(y.mean(),mu,delta=.0001)
        self.assertAlmostEqual(y.var(),expected,delta=expected*.015)
        a=c1_pixels(x[:10],gain,np.random.default_rng(7));b=c1_pixels(x[:10],gain,np.random.default_rng(7))
        np.testing.assert_array_equal(a,b);self.assertEqual(a.dtype,np.uint8)

    def test_gap_and_roi_invalid(self):
        t=np.arange(101)/10;valid=np.ones(101,bool)
        self.assertTrue(supported_interval(t,valid,0,10)[0]);valid[5]=False
        self.assertFalse(supported_interval(t,valid,0,10)[0])
        self.assertTrue(supported_interval(t,valid,1,10)[0])
        self.assertFalse(supported_interval(np.array([0.,.3,1.]),np.ones(3,bool),0,1)[0])

    def test_duplicate_validity_conservative(self):
        t,x,v=merge_duplicates(np.array([0.,0.,1.]),np.array([1.,3.,5.]),np.array([True,False,True]))
        np.testing.assert_array_equal(x,[2,5]);self.assertFalse(v[0])

    def test_qp_actual_summary(self):
        q=qp_summary('[libx264] frame I:2 Avg QP:10.00 size:200\n[libx264] frame P:8 Avg QP:20.00 size:100')
        self.assertEqual(q['mean_qp'],18);self.assertIsNone(qp_summary('no stats')['mean_qp'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg not installed')
    def test_actual_encode_mapping_and_budget(self):
        rng=np.random.default_rng(42);frames=[rng.integers(0,256,(48,64,3),dtype=np.uint8) for _ in range(61)]
        t=np.arange(61)/30
        with tempfile.TemporaryDirectory() as d:
            p,m=simulate(frames,t,Path(d)/'c5','c5',fps=10,budget_root=d,budget_bytes=10000000)
            self.assertEqual(m['actual']['frames'],21);self.assertEqual(m['selected_frames'],21)
            z=np.load(Path(d)/'c5/frame_mapping.npz');np.testing.assert_array_equal(z['timestamps_s'],t[z['source_frame_indices']])
            self.assertTrue(m['bitrate_pass']);self.assertIsNotNone(m['qp']['mean_qp'])
            with self.assertRaises(RuntimeError):simulate(frames,t,Path(d)/'budget','c5',fps=10,budget_root=d,budget_bytes=1)

if __name__=='__main__':unittest.main()
