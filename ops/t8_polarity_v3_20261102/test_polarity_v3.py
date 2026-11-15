"""Focused CPU checks for the actual correction path and fail-closed launch gates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import polarity_guard as guard
import t8_run as runner
# Runner specifies physical GPU0 for real execution; tests hide it before torch import.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from t8_t1_loader import load_clip
from EfficientPhys_fixed import EfficientPhys


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class CorrectionTests(unittest.TestCase):
    def test_canonical_evaluation_exact_hr_failure_and_export_parity(self):
        import polarity_evaluate as evaluator
        import t8_evaluate as baseline
        from ppg_polarity import adapt_cached_meta
        self.assertIs(evaluator.decode_difference, baseline.decode_difference)
        t = np.arange(1001)/100
        values = np.sin(2*np.pi*1.2*t) + .2*np.sin(2*np.pi*2.4*t)
        record = dict(dataset='MCD', subject_id='subject', recording_id='record')
        window = dict(start_s=0., reasons=[])
        cases = [dict(ppg_t=t, ppg=values, ppg_valid=np.ones(len(t),bool)),
                 dict(ppg_t=t, ppg=np.zeros(len(t)), ppg_valid=np.ones(len(t),bool)),
                 dict(ppg_t=t[:200], ppg=values[:200], ppg_valid=np.ones(200,bool)),
                 dict(ppg_t=t, ppg=values, ppg_valid=np.arange(len(t)) != 50)]
        for dataset in ('UBFC2','MCD'):
            record['dataset'] = dataset
            for meta in cases:
                canonical, receipt = adapt_cached_meta(meta,dataset,source_convention='raw_device')
                raw_row = baseline.reference_row(record,window,meta,'native')
                canonical_row = baseline.reference_row(record,window,canonical,'native')
                parity = evaluator.verify_reference_parity(raw_row,canonical_row,receipt['raw_to_canonical_sign'])
                self.assertEqual(parity['failure'],evaluator.reference_outcome(raw_row)['failure'])
                with tempfile.TemporaryDirectory() as tmp:
                    a,b = Path(tmp)/'raw.csv',Path(tmp)/'canonical.csv'
                    baseline.csv_write(a,[raw_row]); baseline.csv_write(b,[canonical_row])
                    if dataset == 'UBFC2':
                        self.assertEqual(a.read_bytes(),b.read_bytes())
                changed = dict(canonical_row)
                samples = json.loads(changed['ppg_values_json'])
                samples[0] = 123.0
                changed['ppg_values_json'] = json.dumps(samples)
                with self.assertRaisesRegex(RuntimeError, 'beyond declared global sign'):
                    evaluator.verify_reference_parity(raw_row,changed,receipt['raw_to_canonical_sign'])

    def test_lag_receipt_requires_consistent_sign_zero_offset_and_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'lag.json'
            receipt = dict(index_sha256=guard.EXPECTED_INDEX, development_only=True, D3_opened=False,
                offset_seconds=0, canonical_pos_multiplier=-1, MCD_ppg_multiplier=-1,
                decision='retain_zero_offset', passed=True, source_results_sha256='source', summary_sha256='summary',
                fullhd_consistent_polarity_median_frames=2, limitations=['Median does not prove every record synchronized'])
            write(path, receipt)
            binding = dict(receipt_sha256=guard.sha(path),source_results_sha256='source',summary_sha256='summary')
            self.assertTrue(guard.verify_lag_receipt(path,binding)['passed'])
            receipt['canonical_pos_multiplier'] = 1
            write(path, receipt)
            binding['receipt_sha256'] = guard.sha(path)
            with self.assertRaisesRegex(RuntimeError, 'canonical_pos_multiplier'):
                guard.verify_lag_receipt(path,binding)
            receipt['canonical_pos_multiplier'] = -1
            receipt['offset_seconds'] = .1
            write(path, receipt)
            binding['receipt_sha256'] = guard.sha(path)
            with self.assertRaisesRegex(RuntimeError, 'offset_seconds'):
                guard.verify_lag_receipt(path,binding)
            receipt['offset_seconds'] = 0
            receipt['fullhd_consistent_polarity_median_frames'] = 3
            write(path, receipt)
            binding['receipt_sha256'] = guard.sha(path)
            with self.assertRaisesRegex(RuntimeError, 'below3frames'):
                guard.verify_lag_receipt(path,binding)

    def test_optional_joint_history_and_exact_control_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = {'UBFC2':[('u', 0.)], 'MCD':[('m', 0.)]}
            digest = hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()
            control = root/'fold0'/'seed2027'/'ubfc_only'
            provenance = dict(initial_model_sha256='initial', train_pool_sha256=digest,
                              code_sha256=runner.BASELINE_CODE)
            write(control/'provenance.json', provenance)
            write(control/'history.json', [dict(epoch=i,draw_sha256=str(i)) for i in range(1,31)])
            pair = guard.baseline_pair(root, 0, 2027, pool, 'initial', runner.BASELINE_CODE)
            self.assertEqual(len(pair['ubfc_only']), 30)
            self.assertEqual(pair['joint'], [])
            partial_joint = root/'fold0'/'seed2027'/'joint'
            write(partial_joint/'provenance.json', provenance)
            write(partial_joint/'history.json', [dict(epoch=i,draw_sha256=str(i)) for i in range(1,12)])
            pair = guard.baseline_pair(root, 0, 2027, pool, 'initial', runner.BASELINE_CODE)
            self.assertEqual(len(pair['joint']), 11)
            with self.assertRaisesRegex(RuntimeError, 'Initial model'):
                guard.baseline_pair(root, 0, 2027, pool, 'different', runner.BASELINE_CODE)

    def test_real_loader_sign_input_and_cpu_backward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = {'roots': {ds: str(root/ds) for ds in ('UBFC2', 'MCD')}}
            rng = np.random.default_rng(88)
            rgb = rng.normal(size=(301, 36, 36, 3)).astype(np.float32)
            t = np.arange(301)/30
            pt = np.arange(1001)/100
            ppg = np.sin(2*np.pi*1.2*pt) + .2*np.sin(2*np.pi*2.4*pt)
            choices = [(ds, 'fixture', 0., 'native') for ds in ('UBFC2', 'MCD')]
            original = []
            for ds, record, start, variant in choices:
                crop = root/ds/record/variant/'crop'
                crop.mkdir(parents=True)
                np.save(crop/'rgb.npy', rgb)
                np.savez(crop/'meta.npz', t=t, valid=np.ones(301, bool), ppg=ppg,
                         ppg_t=pt, ppg_valid=np.ones(1001, bool))
                original.append(load_clip(crop, start))
            before = {str(p): guard.sha(p) for p in root.rglob('*') if p.is_file()}
            from polarity_loader import load_canonical_clip
            for ds, record, start, variant in choices:
                cx, cy, cw, receipt = load_canonical_clip(root/ds/record/variant/'crop', start, ds)
                expected_sign = -1 if ds == 'MCD' else 1
                raw = original[1 if ds == 'MCD' else 0]
                np.testing.assert_array_equal(cw, expected_sign*raw[2])
                np.testing.assert_array_equal(cy, expected_sign*raw[1])
                self.assertIsInstance(receipt, dict)
            x, y = runner.batch_arrays(index, choices)
            np.testing.assert_array_equal(x, np.stack([v[0] for v in original]))
            np.testing.assert_array_equal(y[0], original[0][1])
            np.testing.assert_array_equal(y[1], -original[1][1])
            self.assertEqual(before, {str(p): guard.sha(p) for p in root.rglob('*') if p.is_file()})
            torch.manual_seed(123)
            model = EfficientPhys(frame_depth=10, img_size=36).cpu()
            model.train()
            pred = model(torch.from_numpy(x[:, :11]))
            target = torch.from_numpy(y[:, :10])
            loss = (pred-target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            grad_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            state_before = torch.get_rng_state().clone()
            logged = {'UBFC2': [], 'MCD': []}
            guard.record_dataset_mse(pred, target, choices, logged)
            self.assertTrue(torch.equal(state_before, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(a, p.grad) for a,p in zip(grad_before, [p for p in model.parameters() if p.grad is not None])))
            self.assertAlmostEqual((logged['UBFC2'][0]+logged['MCD'][0])/2, float(loss.detach()), places=5)

    def test_original_incomplete_and_lock_busy_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'queue.lock').touch()
            write(root/'queue_status.json', {'state':'running','runs':17})
            with self.assertRaisesRegex(RuntimeError, 'missing UBFC-only'):
                guard.acquire_completed_baseline(root, flock=lambda fd: None)
            def busy(fd):
                raise BlockingIOError('active original queue')
            with self.assertRaisesRegex(RuntimeError, 'lock unavailable'):
                guard.acquire_completed_baseline(root, flock=busy)

    def test_complete15controls_without_joint_and_tampered_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'queue.lock').touch()
            write(root/'queue_status.json', {'state':'complete','runs':30,'index_sha256':guard.EXPECTED_INDEX})
            for fold in range(5):
                for seed in guard.SEEDS:
                    for arm in ('ubfc_only',):
                        dest = root/f'fold{fold}'/f'seed{seed}'/arm
                        dest.mkdir(parents=True)
                        (dest/'final.pt').write_bytes(b'fixture only')
                        write(dest/'status.json', {'state':'complete','completed_epochs':30,'checkpoint_sha256':guard.sha(dest/'final.pt')})
                        write(dest/'provenance.json', {'index_sha256':guard.EXPECTED_INDEX,
                              'config':dict(fold=fold,seed=seed,arm=arm,epochs=30)})
                        write(dest/'history.json', [dict(epoch=i) for i in range(1,31)])
            lock, receipts = guard.acquire_completed_baseline(root, flock=lambda fd: None)
            self.assertFalse(lock.closed)
            self.assertEqual(len(receipts), 15)
            lock.close()
            (root/'fold4'/'seed2029'/'ubfc_only'/'final.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError, 'checkpoint mismatch'):
                guard.acquire_completed_baseline(root, flock=lambda fd: None)

    def test_budget_no_double_count_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, corrected = root/'original', root/'corrected'
            for p in (original/'UBFC2', original/'MCD', original/'legacy', corrected, root/'t8_label_diagnostic_20261102'):
                p.mkdir(parents=True, exist_ok=True)
            write(original/'UBFC2'/'status.json', dict(bytes=100,legacy_bytes=20))
            write(original/'MCD'/'status.json', dict(legacy_root=str(original/'legacy')))
            (original/'legacy'/'small').write_bytes(b'12345')
            index = {'roots':{ds:str(original/ds) for ds in ('UBFC2','MCD')}}
            result = guard.storage_guard(index, original, corrected)
            expected = sum(p.stat().st_size for p in original.rglob('*') if p.is_file()) + 120
            self.assertEqual(result['total_bytes'], expected)
            self.assertEqual(len(result['counted_roots_bytes']), 3)
            with patch.object(guard, 'CAP', expected + guard.HEADROOM - 1):
                with self.assertRaisesRegex(RuntimeError, '65GB'):
                    guard.storage_guard(index, original, corrected)
            with self.assertRaisesRegex(RuntimeError, 'separate'):
                guard.storage_guard(index, original, original/'nested')
            with self.assertRaisesRegex(RuntimeError, 'direct sibling'):
                guard.validate_output_path(original, root.parent/'outside_project')
            self.assertFalse((original/'nested').exists())
            self.assertFalse((root.parent/'outside_project').exists())

    def test_dependency_bytes_unchanged(self):
        code = Path(__file__).parent
        for name, digest in runner.BASELINE_CODE.items():
            if name != 't8_run.py':
                self.assertEqual(guard.sha(code/name), digest, name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
