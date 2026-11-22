import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import sys
import repro

class ReproductionSetupTests(unittest.TestCase):
    def test_longest_path_prefix_and_no_cascading(self):
        mapping={'/old':'/new/old','/old/code':'/new/ops'}
        self.assertEqual(repro.relocate("Path('/old/code/a.py'); '/old/data' '/older/file'",mapping),
                         "Path('/new/ops/a.py'); '/new/old/data' '/older/file'")

    def test_rejects_asset_traversal(self):
        for path in ('/tmp/a','reports/../../private_file','reports\\..\\private_file','C:/data/a','.git/config','data/holdout/a'):
            with self.subTest(path=path),self.assertRaises(ValueError):repro.relative_path(path)

    def test_missing_hash_and_nondev_bundle_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'assets.json'; c={'assets_manifest':p}
            p.write_text(json.dumps({'schema_version':1,'scope':'holdout','files':[]}))
            with self.assertRaises(ValueError):repro.assets(c)
            p.write_text(json.dumps({'schema_version':1,'scope':'dev-only','files':[
                {'source':'absent','destination':'data/clip.bin','sha256':'placeholder'}]}))
            self.assertIn('Expected maintainer SHA256',repro.assets(c)[1][0])

    def test_actual_bytes_verified_and_input_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);source=base/'input.bin';source.write_bytes(b'original')
            row={'source':'input.bin','destination':'data/a.bin','sha256':hashlib.sha256(b'original').hexdigest()}
            p=base/'assets.json';p.write_text(json.dumps({'schema_version':1,'scope':'dev-only','files':[row]}))
            self.assertEqual(repro.assets({'assets_manifest':p})[1],[])
            source.write_bytes(b'changed')
            self.assertIn('SHA256 mismatch',repro.assets({'assets_manifest':p})[1][0])
            self.assertEqual(source.read_bytes(),b'changed')

    def test_json_schema_does_not_accept_duplicate_destinations(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'assets.json';r={'source':'a','destination':'data/a','sha256':'0'*64}
            p.write_text(json.dumps({'schema_version':1,'scope':'dev-only','files':[r,r]}))
            with self.assertRaisesRegex(ValueError,'Duplicate'):repro.assets({'assets_manifest':p})

    def test_prepare_isolated_copy_and_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);repo=base/'checkout';repo.mkdir();(repo/'reports').mkdir();(repo/'docs').mkdir()
            script=repo/'reports/example.py';before=b"ROOT = '/data_pool/khangle/facevitals/data'\n"
            script.write_bytes(before)
            (repo/'docs/SOURCE_MANIFEST.json').write_text(json.dumps({'files':[
                {'path':'reports/example.py','sha256':hashlib.sha256(before).hexdigest()}]}))
            subprocess.run(['git','init','-q',str(repo)],check=True)
            subprocess.run(['git','-C',str(repo),'add','reports/example.py'],check=True)
            raw=base/'input.bin';raw.write_bytes(b'unchanged binary')
            manifest=base/'assets.json';manifest.write_text(json.dumps({'schema_version':1,'scope':'dev-only',
                'complete_for':['btc-reader-v3'],'files':[{'source':'input.bin','destination':'data/a.bin',
                'sha256':repro.sha(raw)}]}))
            c={'workspace':base/'prepared','assets_manifest':manifest,
               'python':{p:sys.executable for p in repro.PROFILES},'path_map':{}}
            with patch.object(repro,'ROOT',repo):
                repro.prepare(c)
                self.assertEqual(script.read_bytes(),before)
                self.assertEqual(raw.read_bytes(),b'unchanged binary')
                self.assertEqual((c['workspace']/'data/a.bin').read_bytes(),raw.read_bytes())
                self.assertNotIn('/data_pool/',(c['workspace']/'reports/example.py').read_text())
                self.assertEqual(repro.doctor(c,'btc-reader-v3',prepared=True),[])
                with self.assertRaisesRegex(ValueError,'already exists'):repro.prepare(c)
                (c['workspace']/'data/a.bin').write_bytes(b'corrupt')
                self.assertTrue(any('Prepared file changed' in e for e in repro.doctor(c,'btc-reader-v3',prepared=True)))

if __name__=='__main__':unittest.main()
