import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from check_inference_assets import verify

class AssetIntegrityTests(unittest.TestCase):
    def fixture(self, folder):
        p=Path(folder);(p/'sample.json').write_bytes(b'{}')
        m={'schema_version':1,'files':[{'path':'sample.json','bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}]}
        (p/'MANIFEST.json').write_text(json.dumps(m))
        return p,m

    def test_hash_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p,m=self.fixture(d)
            self.assertEqual(verify(p),1)
            (p/'sample.json').write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):verify(p)

    def test_extra_file_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p,m=self.fixture(d);(p/'unlisted.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Unexpected'):verify(p)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,m=self.fixture(d);m['files'][0]['path']='../sample.json'
            (p/'MANIFEST.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'Invalid manifest path'):verify(p)
