"""Synthetic-only checks of the publication CLI, not a dataset evaluation."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PublicationTests(unittest.TestCase):
    def test_synthetic_pulse_and_static_abstention(self):
        cli = ROOT / 'tools/pos_trace.py'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            source, output = d/'rgb.csv', d/'pred.csv'
            subprocess.run([sys.executable, str(cli), '--make-demo', str(source)], check=True)
            subprocess.run([sys.executable, str(cli), '--input', str(source), '--output', str(output)], check=True)
            with output.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(abs(float(r['PR_bpm'])-90) < 1 for r in rows))
            static = d/'static.csv'
            with static.open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['timestamp_s','red','green','blue','valid'])
                w.writerows((i/30,100,120,80,1) for i in range(301))
            out = d/'static_pred.csv'
            subprocess.run([sys.executable, str(cli), '--input', str(static), '--output', str(out)], check=True)
            with out.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(float(rows[0]['valid']), 0)

if __name__ == '__main__':
    unittest.main()
