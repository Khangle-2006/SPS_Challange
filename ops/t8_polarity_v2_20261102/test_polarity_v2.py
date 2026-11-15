"""Focused CPU checks for the actual correction path and fail-closed launch gates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
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
            with self.assertRaisesRegex(RuntimeError, 'not complete'):
                guard.acquire_completed_baseline(root, flock=lambda fd: None)
            def busy(fd):
                raise BlockingIOError('active original queue')
            with self.assertRaisesRegex(RuntimeError, 'lock unavailable'):
                guard.acquire_completed_baseline(root, flock=busy)

    def test_complete30_and_tampered_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'queue.lock').touch()
            write(root/'queue_status.json', {'state':'complete','runs':30,'index_sha256':guard.EXPECTED_INDEX})
            for fold in range(5):
                for seed in guard.SEEDS:
                    for arm in ('ubfc_only','joint'):
                        dest = root/f'fold{fold}'/f'seed{seed}'/arm
                        dest.mkdir(parents=True)
                        (dest/'final.pt').write_bytes(b'fixture only')
                        write(dest/'status.json', {'state':'complete','completed_epochs':30,'checkpoint_sha256':guard.sha(dest/'final.pt')})
                        write(dest/'provenance.json', {'index_sha256':guard.EXPECTED_INDEX,
                              'config':dict(fold=fold,seed=seed,arm=arm,epochs=30)})
                        write(dest/'history.json', [dict(epoch=i) for i in range(1,31)])
            lock, receipts = guard.acquire_completed_baseline(root, flock=lambda fd: None)
            self.assertFalse(lock.closed)
            self.assertEqual(len(receipts), 30)
            lock.close()
            (root/'fold4'/'seed2029'/'joint'/'final.pt').write_bytes(b'tampered')
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
