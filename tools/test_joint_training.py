"""Synthetic checks for isolated fitting, bundle integrity and fresh-checkout imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from training.btc_joint import assets
from training.btc_joint.fit import fit_models
from fit_btc_joint import difference, plain
from reader_v4 import GRID, MODELS


class JointTrainingTests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(recording=f'synthetic_{i}', person=i % 158, fold=(i % 158) % 5,
                     pr_bpm=75., task='Resting') for i in range(218)]
        p = root / assets.VIDEOS
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(rows))
        for name in assets.required_paths(rows, ['fit', 'replay']):
            if name == assets.VIDEOS:
                continue
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'synthetic integrity fixture, not participant data')
        manifest = root / 'assets.local.json'
        assets.create_manifest(root, manifest, ['fit', 'replay'])
        return manifest

    def test_tampered_input_and_incomplete_bundle_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self.fixture(root)
            self.assertEqual(len(assets.verify(root, manifest, 'fit')), 218)
            doc = json.loads(manifest.read_text())
            short = dict(doc, files=doc['files'][:-1])
            manifest.write_text(json.dumps(short))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                assets.verify(root, manifest, 'fit')
            manifest.write_text(json.dumps(doc))
            (root / assets.NATIVE).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                assets.verify(root, manifest, 'replay')

    def test_person_cannot_cross_folds_and_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            rows = assets.load_videos(root)
            rows[158]['fold'] = (rows[158]['fold'] + 1) % 5
            (root / assets.VIDEOS).write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'multiple folds'):
                assets.load_videos(root)
            for name in ('../outside', '/tmp/outside', 'C:/outside', 'a\\b'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    assets.resolve(root, name)

    def test_nondev_manifest_denied(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'manifest.json'
            path.write_text(json.dumps(dict(schema_version=1, kind='btc-joint-dev', scope='holdout', videos=218, people=158)))
            with self.assertRaisesRegex(ValueError, 'dev-only'):
                assets.verify(Path(d), path, 'fit')

    def test_outer_validation_labels_do_not_change_fitted_models(self):
        rng = np.random.default_rng(123)
        n = 100
        folds = np.arange(n) % 5
        low = ((np.arange(n) // 5) % 2).astype(int)
        task = ((np.arange(n) // 10) % 2).astype(bool)
        truth = rng.uniform(65, 105, n)
        data = dict(truth=truth.copy(), person=np.arange(n), fold=folds, low=low, task=task,
                    capture=rng.normal(size=(n, 4)), trace=rng.normal(size=(n, 3)))
        for name in ('C', 'P', 'sC', 'sP', *MODELS):
            peak = truth + rng.normal(0, 5, n)
            power = np.exp(-.5 * ((GRID - peak[:, None]) / 7)**2) + .001
            data[name] = power / power.sum(1, keepdims=True)
        before = plain(fit_models(data, outer_folds=(0,)))
        data['truth'][folds == 0] += 30
        after = plain(fit_models(data, outer_folds=(0,)))
        self.assertEqual(before, after)
        self.assertEqual(set(before), {'router_0.json', 'fallback_0.json'})

    def test_parameter_comparison_rejects_changed_values_and_shapes(self):
        self.assertEqual(difference({'x': [1., 2.]}, {'x': [1., 2.]}), 0.)
        for actual in ({'x': [1., 3.]}, {'x': [1.]}, {'y': [1., 2.]}):
            with self.assertRaises(ValueError):
                difference(actual, {'x': [1., 2.]})

    def test_r1_helper_imports_without_site_checkout_or_pythonpath(self):
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        with tempfile.TemporaryDirectory() as cwd:
            r = subprocess.run([sys.executable, str(ROOT/'reports/rerun_stateless_20261123/code/expert_spectrum.py')],
                               cwd=cwd, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == '__main__':
    unittest.main()
