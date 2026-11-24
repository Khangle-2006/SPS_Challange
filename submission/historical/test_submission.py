import tempfile
import unittest
from pathlib import Path
import numpy as np
import json
import shutil
from submission.mat_writer import write_mat, verify_mat
from submission.metrics import compute_metrics
from submission.repair_bundle_v2 import repair


class SubmissionTests(unittest.TestCase):
    def test_mat73_axes_types_and_unicode(self):
        rng=np.random.default_rng(2027);rgb=rng.integers(0,256,(6,8,3,4),dtype=np.uint8)
        fields={'roi_data':{'forehead':np.zeros((4,4,2)),'roi_type':'bbox_placeholder'},
                'reference_signal':{'ppg':np.arange(10.),'timestamps_s':np.arange(10.)},
                'processed_signal':{'rppg':np.arange(10.),'abstain':np.asarray([False,True])},
                'metrics':{'mae':1.,'loa_lower':'NA'},'params':{'age':'NA','dataset':'tổng hợp'}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.mat'
            info=write_mat(p,fields,frames=(rgb[:,:,:,i] for i in range(4)),shape=rgb.shape)
            verify_mat(p,rgb.shape,info['decoded_rgb_sha256'])
            import mat73
            back=mat73.loadmat(str(p))
            np.testing.assert_array_equal(back['facial_video'],rgb)
            self.assertEqual(back['params']['dataset'],'tổng hợp')
    def test_repeated_measure_variance(self):
        # Three subject biases1,3,5; each pair of residuals -1,+1.
        r=np.arange(6.)+60;p=r+np.asarray([0,2,2,4,4,6])
        m=compute_metrics(p,r,['a','a','b','b','c','c'])
        self.assertAlmostEqual(m['within_subject_variance'],2)
        self.assertAlmostEqual(m['between_subject_variance'],3)
        self.assertAlmostEqual(m['loa_lower'],3-1.96*np.sqrt(5))
    def test_one_subject_no_pseudo_replication(self):
        m=compute_metrics([61,63],[60,62],['a','a'])
        self.assertEqual(m['loa_lower'],'NA')
        self.assertEqual(m['n_subjects'],1)
    def test_missing_and_abstain_remain_denominator(self):
        m=compute_metrics([60,np.nan,61],[60,60,60],['a','a','b'],[False,False,True])
        self.assertAlmostEqual(m['coverage'],1/3)
        self.assertEqual(m['n_windows'],3)
    def test_portable_bundle_repair_preserves_original(self):
        source=Path(__file__).parent
        with tempfile.TemporaryDirectory() as d:
            artifact=Path(d);dst=artifact/'code'/'submission';dst.mkdir(parents=True)
            for name in ['__init__.py','process_round1.py','mat_writer.py','metrics.py']:
                shutil.copyfile(source/name,dst/name)
            (dst/'checkpoints').mkdir();(dst/'checkpoints/baseline_seed2027.pt').write_bytes(b'test-only')
            (artifact/'script_sha256.json').write_text('{}')
            (artifact/'status.json').write_text(json.dumps({'state':'complete'}))
            (artifact/'Round1_Baseline_None.mat').write_bytes(b'untouched test sentinel')
            (artifact/'process_round1.py').write_text("from pathlib import Path\nimport runpy\nrunpy.run_path(str(Path(__file__).parent/'code'/'submission'/'process_round1.py'),run_name='__main__')\n")
            receipt=repair(artifact,source)
            self.assertTrue(receipt['standalone_cli_help_passed'])
            self.assertEqual((artifact/'Round1_Baseline_None.mat').read_bytes(),b'untouched test sentinel')
            self.assertTrue((dst/'requirements.txt').is_file())
            self.assertTrue((artifact/'original_packaging_v1/process_round1.py').is_file())


if __name__=='__main__':unittest.main()
