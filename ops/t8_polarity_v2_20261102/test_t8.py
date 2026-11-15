"""Tests protect leakage, matching, clocks and failure accounting, without a GPU."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from t8_plan import validate_folds, read, draw, pools, steps_per_epoch, build_index, VARIANTS
from t8_evaluate import reference_row, csv_write, decode_difference
from score_round1 import score, reference_hr
from t8_t1_loader import tests as clock_tests
from t8_aggregate import aggregate

HERE = Path(__file__).resolve().parent


class T8Tests(unittest.TestCase):
    def setUp(self):
        self.spec = read(HERE/'folds_frozen.json')

    def test_original_test_and_each_outer_fold_are_isolated(self):
        populations = validate_folds(self.spec)
        self.assertEqual(len(populations['MCD']), 60)
        broken = copy.deepcopy(self.spec)
        broken['folds'][0]['ubfc_train_subjects'][0] = broken['original_ubfc_test_subjects'][0]
        with self.assertRaises(AssertionError):
            validate_folds(broken)
        broken = copy.deepcopy(self.spec)
        broken['folds'][0]['mcd_train_subjects'][0] = broken['folds'][0]['mcd_val_subjects'][0]
        with self.assertRaises(AssertionError):
            validate_folds(broken)

    def test_matched_draws_even_with_unequal_pools(self):
        pool = {'UBFC2': [('u', float(i)) for i in range(7)],
                'MCD': [('m', float(i)) for i in range(101)]}
        self.assertEqual(steps_per_epoch(pool), 54)
        native = 0
        conditions = set()
        for step in range(1000):
            j = draw(pool, 2027, 0, 0, step, 'joint')
            u = draw(pool, 2027, 0, 0, step, 'ubfc_only')
            self.assertEqual(j[0], u[0])
            self.assertEqual([x[-1] for x in j], [x[-1] for x in u])
            self.assertEqual(j, draw(pool, 2027, 0, 0, step, 'joint'))
            self.assertEqual([x[0] for x in j], ['UBFC2', 'MCD'])
            self.assertEqual([x[0] for x in u], ['UBFC2', 'UBFC2'])
            native += sum(x[-1] == 'native' for x in j)
            conditions.update(x[-1] for x in j)
        self.assertTrue(850 < native < 1150)
        self.assertEqual(conditions, set(VARIANTS))
        self.assertNotEqual(draw(pool, 2027, 0, 0, 0, 'joint'), draw(pool, 2027, 0, 1, 0, 'joint'))

    def test_pool_does_not_include_outer_validation(self):
        rows = []
        for dataset, prefix in [('UBFC2', 'ubfc'), ('MCD', 'mcd')]:
            f = self.spec['folds'][0]
            rows.extend(dict(dataset=dataset, subject_id=s, recording_id=s, eligible_starts=[0., 10.])
                        for s in f[prefix+'_train_subjects']+f[prefix+'_val_subjects'])
        result = pools({'folds': self.spec, 'records': rows}, 0)
        self.assertEqual(len(result['MCD']), 96)
        self.assertEqual(len(result['UBFC2']), 56)
        self.assertTrue({r[0] for r in result['MCD']}.isdisjoint(self.spec['folds'][0]['mcd_val_subjects']))

    def test_incomplete_cache_is_not_training_ready(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'cohort.json').write_text(json.dumps(dict(dataset='UBFC2', variants=VARIANTS,
                recordings=[str(i) for i in range(35)], expected_items=350)))
            (root/'status.json').write_text(json.dumps(dict(state='running', completed=349, total=350, errors=[])))
            with self.assertRaises(AssertionError):
                build_index({'UBFC2': root, 'MCD': root}, self.spec)

    def test_index_preserves_rejected_rows_and_requires_all_variants(self):
        """Simulate full approved cohort; one failed codec window cannot enter either arm."""
        population = validate_folds(self.spec)
        registry = {}
        roots = {d: Path('/fixture')/d for d in ('UBFC2', 'MCD')}
        bad_rid = None
        for ds in roots:
            subjects = sorted(population[ds])
            records = [(s, s) for s in subjects] if ds == 'UBFC2' else [
                (s, s+'_FullHDwebcam_'+c) for s in subjects for c in ('before', 'after')
                if (s, c) != ('7412', 'after')]
            registry[str(roots[ds]/'cohort.json')] = dict(dataset=ds, variants=VARIANTS,
                recordings=[rid for _, rid in records], expected_items=len(records)*10)
            registry[str(roots[ds]/'status.json')] = dict(state='complete', completed=len(records)*10,
                                                       total=len(records)*10, errors=[])
            registry[str(roots[ds]/'fingerprint.json')] = {'dummy': 'hash'}
            for sid, rid in records:
                canonical = dict(valid_windows=[dict(start_s=0., end_s=10., reasons=[])],
                    rejected_windows=[dict(start_s=10., end_s=20., reasons=['ppg_gap_support'])])
                registry[str(roots[ds]/rid/'canonical_windows.json')] = canonical
                for variant in VARIANTS:
                    bad = ds == 'UBFC2' and sid == subjects[0] and variant == 'f15_b800'
                    if bad:
                        bad_rid = rid
                    registry[str(roots[ds]/rid/variant/'complete.json')] = dict(dataset=ds,
                        recording_id=rid, subject_id=('mcd:' if ds == 'MCD' else 'ubfc2:')+sid,
                        variant=variant, fingerprint={'dummy': 'hash'}, encoding={'bitrate_pass': True},
                        canonical_valid_windows=[dict(start_s=0., usable=not bad)])
        with patch('t8_plan.read', side_effect=lambda p: registry[str(p)]), patch('t8_plan.sha', return_value='hash'):
            index = build_index(roots, self.spec, verify_hashes=False)
        self.assertEqual(len(index['records']), 154)
        self.assertEqual(len(index['excluded_training_windows']), 1)
        self.assertEqual(index['coverage']['UBFC2']['train_eligible'], 34)
        rec = next(r for r in index['records'] if r['dataset'] == 'UBFC2' and r['recording_id'] == bad_rid)
        self.assertEqual(rec['eligible_starts'], [])
        self.assertEqual(len(rec['candidates']), 2)
        self.assertTrue(rec['evaluation_usable']['native']['0.0'])

    def test_reference_independent_clock_and_missing_prediction_denominator(self):
        t = .13+np.arange(2600)/100
        meta = dict(ppg_t=t, ppg=np.sin(2*np.pi*1.2*t), ppg_valid=np.ones(len(t), bool))
        record = dict(dataset='MCD', subject_id='fixture', recording_id='fixture_before')
        rows = [reference_row(record, dict(start_s=s, reasons=[]), meta, 'native') for s in (1., 11.)]
        self.assertAlmostEqual(reference_hr(rows[0]), 72., delta=.5)
        with tempfile.TemporaryDirectory() as d:
            r, p = Path(d)/'ref.csv', Path(d)/'pred.csv'
            csv_write(r, rows)
            csv_write(p, [dict(subject_id=rows[0]['subject_id'], recording_id='fixture_before',
                window_start_s=1., hr_bpm=72., abstain=False)])
            scored = score(r, p)
            self.assertEqual(scored['counts']['eligible_windows'], 2)
            self.assertEqual(scored['counts']['missing_output_windows'], 1)
            self.assertEqual(scored['subject_macro']['coverage'], .5)
        # An invalid sample exactly at an endpoint remains invalid, never interpolated away.
        endpoint = float(t[100])
        meta['ppg_valid'][100] = False
        with self.assertRaises(ValueError):
            reference_hr(reference_row(record, dict(start_s=endpoint, reasons=['ppg_invalid_sample']), meta, 'native'))

    def test_difference_decoder_band_and_invalid_waveform(self):
        for bpm in (48., 90., 150.):
            y = np.sin(2*np.pi*(bpm/60)*np.arange(301)/30)
            self.assertAlmostEqual(decode_difference(np.diff(y)), bpm, delta=.5)
        with self.assertRaises(ValueError):
            decode_difference(np.zeros(300))

    def test_independent_clock_loader(self):
        clock_tests()

    def test_oof_aggregation_requires_all_runs_and_pairs_people_across_seeds(self):
        t = np.arange(301)/30
        meta = dict(ppg_t=t, ppg=np.sin(2*np.pi*1.2*t), ppg_valid=np.ones(len(t), bool))
        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d)/'runs', Path(d)/'summary'
            with self.assertRaises(FileNotFoundError):
                aggregate(root, out)
            self.assertFalse(out.exists())
            for fold in self.spec['folds']:
                for seed in (2027, 2028, 2029):
                    for arm in ('ubfc_only', 'joint'):
                        run = root/f"fold{fold['fold']}"/f'seed{seed}'/arm
                        run.mkdir(parents=True)
                        (run/'status.json').write_text(json.dumps(dict(state='complete')))
                        for dataset, prefix in [('UBFC2', 'ubfc'), ('MCD', 'mcd')]:
                            for condition in ('native', 'f10_b500'):
                                dest = run/'evaluation'/(dataset+'_'+condition)
                                dest.mkdir(parents=True)
                                refs, outputs = [], []
                                for sid in fold[prefix+'_val_subjects']:
                                    rec = dict(dataset=dataset, subject_id=sid, recording_id=sid+'_before')
                                    row = reference_row(rec, dict(start_s=0., reasons=[]), meta, condition)
                                    refs.append(row)
                                    outputs.append(dict(subject_id=row['subject_id'], recording_id=row['recording_id'],
                                        window_start_s=0., hr_bpm=reference_hr(row)+(3 if arm == 'ubfc_only' else 1), abstain=False))
                                csv_write(dest/'reference_signal.csv', refs)
                                csv_write(dest/'processed_output.csv', outputs)
            summary = aggregate(root, out)
            for values in summary['S_by_seed'].values():
                self.assertAlmostEqual(values['ubfc_only'], 3.)
                self.assertAlmostEqual(values['joint'], 1.)
            for values in summary['paired_conditions'].values():
                self.assertAlmostEqual(values['mean_delta'], -2.)
                np.testing.assert_allclose(values['subject_bootstrap_CI95'], [-2., -2.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
