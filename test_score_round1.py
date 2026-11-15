import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from score_round1 import score,reference_hr,repeated_loa
from selection_round1 import compare


def refrow(sid,start=0.,hz=1.5):
    t=np.arange(301)/30+start
    return dict(subject_id=sid,recording_id=sid+'_video',window_start_s=start,video_support_valid='true',
                ppg_timestamps_json=json.dumps(t.tolist()),ppg_values_json=json.dumps(np.sin(2*np.pi*hz*t).tolist()))


def write(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class ScoreTests(unittest.TestCase):
    def run_score(self,ref,pred):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)/'reference.csv';p=Path(d)/'output.csv';write(r,ref);write(p,pred);return score(r,p)
    def predictions(self,rows,errors):
        return [dict(subject_id=r['subject_id'],recording_id=r['recording_id'],window_start_s=r['window_start_s'],
                     hr_bpm=reference_hr(r)+e,abstain='false') for r,e in zip(rows,errors)]
    def test_macro_is_not_window_mean(self):
        r=[refrow('a',i*10) for i in range(3)]+[refrow('b')]
        s=self.run_score(r,self.predictions(r,[0,0,0,20]))
        self.assertAlmostEqual(s['subject_macro']['mae'],10)
        self.assertAlmostEqual(s['window_level']['mae'],5)
        self.assertAlmostEqual(s['subject_macro']['rmse'],10)
        self.assertEqual(s['window_level']['error_gt20'],0)
    def test_missing_prediction_cannot_hide_window(self):
        r=[refrow('a'),refrow('a',10)];p=self.predictions(r,[0,0])[:1];s=self.run_score(r,p)
        self.assertEqual(s['counts']['candidate_windows'],2);self.assertEqual(s['coverage'],.5)
    def test_all_abstain_person_makes_macro_undefined(self):
        r=[refrow('a'),refrow('b')];p=self.predictions(r,[0,0]);p[1]['abstain']='true';s=self.run_score(r,p)
        self.assertIsNone(s['subject_macro']['mae']);self.assertEqual(s['coverage'],.5)
    def test_reference_gap_not_interpolated(self):
        r=refrow('a');t=np.array(json.loads(r['ppg_timestamps_json']));v=np.array(json.loads(r['ppg_values_json']));keep=(t<4)|(t>5)
        r['ppg_timestamps_json']=json.dumps(t[keep].tolist());r['ppg_values_json']=json.dumps(v[keep].tolist())
        with self.assertRaisesRegex(ValueError,'reference_gap'):reference_hr(r)
    def test_duplicate_or_extra_predictions_rejected(self):
        r=[refrow('a')];p=self.predictions(r,[0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_score(r,p+p)
        p[0]['recording_id']='wrong'
        with self.assertRaisesRegex(ValueError,'unexpected'):self.run_score(r,p)
    def test_flat_pearson_and_single_subject_loa_are_undefined(self):
        r=[refrow('a'),refrow('a',10)];p=self.predictions(r,[0,0]);p[0]['hr_bpm']=p[1]['hr_bpm']=90.
        s=self.run_score(r,p);self.assertIsNone(s['window_level']['pearson_r']);self.assertIsNone(s['bland_altman']['loa_lower'])
    def test_repeated_measure_known_variance(self):
        b=repeated_loa([0,2,2,4,4,6],['a','a','b','b','c','c'])
        self.assertAlmostEqual(b['within_variance'],2);self.assertAlmostEqual(b['between_variance'],3)
        self.assertAlmostEqual(b['loa_lower'],3-1.96*np.sqrt(5))
    def test_reference_fixed_band(self):
        self.assertLess(abs(reference_hr(refrow('a',hz=.72))-43.2),1.)
        self.assertLess(abs(reference_hr(refrow('a',hz=1.5))-90),.5)
    def test_nonfinite_prediction_is_abstain(self):
        r=[refrow('a')];p=self.predictions(r,[0]);p[0]['hr_bpm']='NaN';s=self.run_score(r,p)
        self.assertEqual(s['coverage'],0);self.assertEqual(s['reasons']['invalid_prediction'],1)
    def test_selection_three_frozen_gates(self):
        def bundle(mae,p95):
            return {n:dict(schema_version='v1',reference_sha256=n,coverage=1.,subject_macro={'mae':mae},window_level={'p95_abs_error':p95}) for n in ['D1_clean','D1_C5','D2_native']}
        self.assertTrue(compare(bundle(10,20),bundle(9,20))['passes_three_numerical_gates'])
        self.assertFalse(compare(bundle(10,20),bundle(9,21))['passes_three_numerical_gates'])
        self.assertFalse(compare(bundle(0,0),bundle(0,0))['passes_three_numerical_gates'])


if __name__=='__main__':unittest.main()
