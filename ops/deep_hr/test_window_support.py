import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from data_adapters import Label,MCD,ExcludedRecording
from window_support import supported_windows


class WindowSupportTests(unittest.TestCase):
    def test_delayed_PPG_and_one_second_gap(self):
        vt=np.arange(1801)/30
        pt=2.3+np.arange(5771)/100
        pt=pt[(pt<27)|(pt>28)]
        r=supported_windows(vt,pt)
        starts=[w['start_s'] for w in r['valid_windows']]
        self.assertAlmostEqual(r['coverage_start_s'],2.3)
        self.assertTrue(all(s>=2.3 for s in starts))
        self.assertTrue(all(s+10<=27 or s>=28 for s in starts))
        self.assertEqual(len(r['rejected_windows']),1)
        self.assertIn('ppg_gap_support',r['rejected_windows'][0]['reasons'])

    def test_shared_clock_unchanged(self):
        t=np.arange(1801)/30
        r=supported_windows(t,t)
        self.assertEqual(len(r['valid_windows']),6)
        self.assertEqual(r['legacy_video_only_windows'],6)
        self.assertEqual(r['rejected_windows'],[])

    def test_NaN_label_and_video_mask_reject_windows(self):
        t=np.arange(901)/30;p=np.sin(t);p[420]=np.nan
        label=Label(p,t,'a.u.')
        valid=np.ones(len(t),bool);valid[750]=False
        r=supported_windows(t,t,valid,label.valid_mask)
        self.assertEqual([w['start_s'] for w in r['valid_windows']],[0.])
        self.assertEqual(len(r['rejected_windows']),2)

    def test_no_overlap_and_duplicates(self):
        self.assertEqual(supported_windows([0,1],[2,3])['candidate_windows'],0)
        t=np.r_[0.,np.arange(601)/30]
        self.assertEqual(len(supported_windows(t,t)['valid_windows']),2)

    def test_MCD_errata_and_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);audit=root/'audit';(audit/'dev60_gate_v3').mkdir(parents=True)
            rows=[]
            for sid,step in [('4952','before'),('7412','after'),('9401','after')]:
                rows.append(dict(patient_id=sid,step=step,camera='FullHDwebcam',video=f'{sid}_{step}.avi',
                                 meta='meta.txt',ppg='ppg.PW',ppg_sync='sync.txt',pulse='99',
                                 upper_ap='120',lower_ap='80',saturation='98'))
            with (root/'db.csv').open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
            (root/'meta.txt').write_text('0 2020-01-01 00:00:00.000000\n1 2020-01-01 00:00:00.040000\n2 2020-01-01 00:00:00.080000\n3 2020-01-01 00:00:00.120000\n4 2020-01-01 00:00:00.160000\n')
            (root/'ppg.PW').write_text('1 2020-01-01 00:00:00.010000\n2 2020-01-01 00:00:00.060000\n3 2020-01-01 00:00:00.110000\n')
            np.savetxt(root/'sync.txt',[[1,.01],[2,.02],[2,.02],[3,.01]])
            gate={'gate':{'candidate_prefix_frames':3},'decoded_corrupt_indices':[], 'source_sha256':'synthetic'}
            (audit/'dev60_gate_v3/9401_after.json').write_text(json.dumps(gate))
            backend=MCD(root,audit)
            for name in ['4952_before','7412_after']:
                with self.assertRaises(ExcludedRecording) as caught:backend.load(name)
                self.assertTrue(caught.exception.metadata['exclusion_reason'])
            r=backend.load('9401_after')
            self.assertEqual(len(r.frames),3)
            np.testing.assert_allclose(r.timestamps,[0,.04,.08])
            self.assertAlmostEqual(r.metadata['fps_effective'],25)
            self.assertEqual(r.metadata['excess_meta_rows_dropped'],2)
            self.assertTrue(r.metadata['pulse']['not_for_hr_label'])
            self.assertNotIn('hr',r.labels)
            self.assertEqual(r.labels['sbp'].scope,'recording')
            self.assertAlmostEqual(r.require_synchronized_ppg().timestamps[0],.01)

if __name__=='__main__':unittest.main()
