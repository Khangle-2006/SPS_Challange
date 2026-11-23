"""Numerical regression and negative controls for the selected video runner."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import contextlib
import io
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from btc_spectral_math import GRID,window_spectrum,supported,avg_spectra
from reader_v4 import evidence,MODELS,predG
from common import load_model
from video_core import main as score
from video_experts import spectrum


class VideoTests(unittest.TestCase):
    def test_frozen_spectral_operator(self):
        x=np.sin(2*np.pi*1.5*np.arange(300)/30)
        p=window_spectrum(x)
        self.assertAlmostEqual(float(p.sum()),1.,places=12)
        self.assertLessEqual(abs(GRID[p.argmax()]-90),.5)

    def test_invalid_frames_reject_a_window(self):
        t=np.arange(301)/30;v=np.ones(301,bool)
        self.assertTrue(supported(t,v,0,10))
        v[150]=False
        self.assertFalse(supported(t,v,0,10))

    def test_static_classical_does_not_produce_a_pulse(self):
        t=np.arange(901)/30
        with np.errstate(all='ignore'):
            self.assertIsNone(avg_spectra(t,np.ones(len(t),bool),np.tile([100.,90.,80.],(len(t),1))))

    def test_five_equal_evidence_inputs(self):
        x=np.exp(-.5*((GRID-90)/3)**2)+1e-6
        z=evidence(x[None],x[None],{n:x[None] for n in MODELS})
        self.assertEqual(z.shape,(1,277))
        self.assertEqual(float(GRID[z.argmax()]),90.)

    def test_expert_recurrent_state_is_reset_for_each_video(self):
        class FakeStatefulExpert:
            state=0
            def process_faces_tensor(self, crops, fps):
                t=np.arange(901)/30.
                self.wave=np.sin(2*np.pi*(1.2+self.state*.1)*t)
                self.state+=1
            def bvp(self, raw=True):return self.wave,np.arange(901)/30.
        m=FakeStatefulExpert();t=np.arange(901)/30.;v=np.ones(901,bool);x=np.zeros((901,1,1,3),np.uint8)
        a,_=spectrum(m,x,t,v);b,_=spectrum(m,x,t,v)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(m.state,2)

    def test_models_load_and_prior_is_not_a_measured_pr(self):
        for f in [None,0,1,2,3,4]:
            m,_=load_model(f)
            for low in [0,1]:
                for task in [False,True]:
                    p=predG(m,np.zeros((1,277)),np.array([low]),np.array([low]),np.array([task]))
                    self.assertTrue(np.isfinite(p).all())
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            (out/'request.json').write_text(json.dumps(dict(fold=None,activity='Resting',frontend='guarded')))
            (out/'frontend.json').write_text(json.dumps(dict(capture_class_low=False,frames=901,valid_frames=901,video_sha256='test')))
            (out/'experts.json').write_text(json.dumps(dict(abstain=True,reason='static_input',weight_sha256={})))
            with contextlib.redirect_stdout(io.StringIO()):score(out)
            r=json.loads((out/'result.json').read_text())
            self.assertTrue(r['abstain']);self.assertIsNone(r['pr_bpm']);self.assertIsInstance(r['fallback_prior_bpm'],float)


if __name__=='__main__': unittest.main()
