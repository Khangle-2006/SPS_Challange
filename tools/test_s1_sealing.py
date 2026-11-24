"""Safety checks use synthetic files only; never open a holdout manifest."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class SealingTests(unittest.TestCase):
    def test_utf8_authorization_reaches_activity_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);auth=p/'authorization.txt'
            auth.write_text('M\u1ede HOLDOUT S1',encoding='utf-8')
            r=subprocess.run([sys.executable,str(ROOT/'submission/run_holdout.py'),'--repo',str(ROOT),'--manifest',str(p/'holdout_NOT_PRESENT.json'),'--output',str(p/'out'),'--authorization-file',str(auth)],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('Frozen activity lock missing/changed',r.stderr)
            self.assertFalse((p/'out').exists())

    def test_locked_activity_normalization(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('s1_runner',ROOT/'submission/run_holdout.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual([module.cli_activity(x) for x in ['Task1','Task2','Task3','Task','Resting','Walking']],
                         ['Task','Task','Task','Task','Resting','Walking'])
        with self.assertRaises(ValueError):module.cli_activity('unknown')

    def test_holdout_denied_before_opening_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            r=subprocess.run([sys.executable,str(ROOT/'submission/run_holdout.py'),'--repo',str(ROOT),'--manifest',str(Path(d)/'holdout_NOT_PRESENT.json'),'--output',str(Path(d)/'out')],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('Holdout CLOSED',r.stderr)
            self.assertFalse((Path(d)/'out').exists())

    def test_tampered_seal_denied_before_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'PREDICTIONS.json').write_text('[]');(p/'PRE_RUN.json').write_text('{}')
            (p/'SEALED.json').write_text(json.dumps(dict(scope='dev_dryrun',predictions_sha256='wrong',pre_run_sha256='wrong',videos=3)))
            r=subprocess.run([sys.executable,str(ROOT/'submission/score_holdout.py'),'--predictions',d,'--labels',str(p/'labels_NOT_PRESENT.json')],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('seal changed',r.stderr)
            self.assertFalse((p/'SCORE_STARTED.json').exists())


if __name__=='__main__':unittest.main()
