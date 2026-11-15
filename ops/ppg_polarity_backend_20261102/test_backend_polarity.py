"""Focused CPU tests for the new backend contract, with synthetic labels only."""
import unittest
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from scipy import signal
from data_adapters import Label, Recording, MCD, UBFC2
from unittest.mock import patch
import source_data_adapters as source
from ppg_polarity import adapt_cached_meta, normalize_recording, CONVENTION, RAW
from morphology_contract import assess_dataset


def fixture(dataset='MCD-rPPG', camera='USB', split='development'):
    t = np.arange(1200)/100
    volume = signal.sawtooth(2*np.pi*1.2*t, width=.2)
    raw = -volume if dataset == 'MCD-rPPG' else volume
    return Recording([None]*len(t), t, {'ppg': Label(raw, t, 'a.u.'),
        'ppg_frame': Label(raw[::2], t[::2], 'a.u.'), 'hr': Label([72], None, 'bpm', 'recording')},
        'synthetic', 'synthetic', {'dataset': dataset, 'camera': camera, 'split': split})


class BackendTests(unittest.TestCase):
    def test_active_backend_real_load_path_fake_all_camera_partition_records(self):
        # Exercise the active MCD.load implementation without real data/frames.
        path = Path(__file__).resolve().parents[1]/'deep_hr'/'data_adapters.py'
        spec = importlib.util.spec_from_file_location('active_adapter_polarity_test', path)
        active = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = active
        spec.loader.exec_module(active)
        for camera in ['USB', 'phone', 'third_camera']:
            for partition in ['development', 'future_D3_metadata_only']:
                backend = active.MCD.__new__(active.MCD)
                backend.root = Path('/synthetic'); backend.audit_dir = Path('/synthetic/audit')
                backend.rows = {'fixture': dict(patient_id='synthetic', camera=camera,
                    step='before', video='fixture.avi', meta='meta.txt', ppg='ppg.txt',
                    ppg_sync='sync.txt', partition=partition)}
                vt = np.array([0, 33333, 66666, 99999], np.int64)
                pt = np.arange(12, dtype=np.int64)*10000
                raw = np.arange(12, dtype=float)+2
                def read_values(p):
                    return (np.zeros(4), vt) if p.name == 'meta.txt' else (raw, pt)
                gate = json.dumps(dict(gate={'candidate_prefix_frames':4}, source_sha256='synthetic'))
                with patch.object(Path, 'read_text', return_value=gate), \
                     patch.object(active, 'read_timed_values', side_effect=read_values), \
                     patch.object(active.np, 'loadtxt', return_value=np.c_[raw[:4], np.zeros(4)]):
                    rec = backend.load('fixture')
                np.testing.assert_array_equal(rec.labels['ppg'].values, -raw)
                np.testing.assert_array_equal(rec.labels['ppg_frame'].values, -raw[:4])
                np.testing.assert_array_equal(rec.labels['ppg'].timestamps, pt/1e6)
                np.testing.assert_array_equal(rec.labels['ppg_frame'].timestamps, vt/1e6)
                self.assertEqual(rec.metadata['ppg_polarity']['raw_to_canonical_sign'], -1)
                np.testing.assert_array_equal(normalize_recording(rec).labels['ppg'].values, -raw)

    def test_global_mcd_all_camera_partition_metadata_and_exactly_once(self):
        # Synthetic metadata only: this does not open any D3 source or labels.
        for camera in ['USB', 'phone', 'third_camera']:
            for split in ['development', 'future_D3_contract_only']:
                raw = fixture(camera=camera, split=split)
                before = raw.labels['ppg'].values.copy()
                with patch.object(source.MCD, 'load', return_value=raw):
                    adapted = MCD.__new__(MCD).load('synthetic')
                again = normalize_recording(adapted)
                for name in ['ppg', 'ppg_frame']:
                    np.testing.assert_array_equal(adapted.labels[name].values, -raw.labels[name].values)
                    np.testing.assert_array_equal(adapted.labels[name].timestamps, raw.labels[name].timestamps)
                    np.testing.assert_array_equal(adapted.labels[name].valid_mask, raw.labels[name].valid_mask)
                    np.testing.assert_array_equal(again.labels[name].values, adapted.labels[name].values)
                np.testing.assert_array_equal(raw.labels['ppg'].values, before)
                self.assertIs(raw.frames, adapted.frames)
                np.testing.assert_array_equal(raw.timestamps, adapted.timestamps)
                self.assertIs(raw.labels['hr'], adapted.labels['hr'])

    def test_cached_waveform_before_diff_and_no_double_flip(self):
        raw = fixture(); label = raw.labels['ppg']
        meta = {'ppg': label.values, 'ppg_t': label.timestamps,
                'ppg_valid': label.valid_mask, 't': np.arange(301)/30, 'rgb': object()}
        adapted, receipt = adapt_cached_meta(meta, 'MCD', source_convention=RAW)
        self.assertIs(adapted['rgb'], meta['rgb'])
        self.assertIs(adapted['ppg_t'], meta['ppg_t'])
        np.testing.assert_array_equal(adapted['ppg'], -meta['ppg'])
        q = np.arange(301)/30
        before_diff = np.diff(np.interp(q, adapted['ppg_t'], adapted['ppg']).astype(np.float32))
        raw_diff = np.diff(np.interp(q, meta['ppg_t'], meta['ppg']).astype(np.float32))
        np.testing.assert_array_equal(before_diff, -raw_diff)
        again, _ = adapt_cached_meta(adapted, 'MCD', source_convention=CONVENTION, provenance=receipt)
        np.testing.assert_array_equal(again['ppg'], adapted['ppg'])
        with self.assertRaises(ValueError):
            adapt_cached_meta(adapted, 'MCD', source_convention=RAW, provenance=receipt)
        with self.assertRaises(ValueError):
            adapt_cached_meta(adapted, 'MCD', source_convention=CONVENTION)

    def test_ubfc_identity_and_explicit_unsupported(self):
        raw = fixture('UBFC2')
        with patch.object(source.UBFC2, 'load', return_value=raw):
            adapted = UBFC2.__new__(UBFC2).load('synthetic')
        np.testing.assert_array_equal(adapted.labels['ppg'].values, raw.labels['ppg'].values)
        for ds in ['PURE', 'UBFC1', 'future_organizer']:
            rec = fixture(ds)
            result = normalize_recording(rec)
            self.assertEqual(result.metadata['ppg_polarity']['status'], 'unverified_requires_adapter_contract')
            with self.assertRaises(ValueError):
                adapt_cached_meta({'ppg': np.ones(100)}, ds, source_convention=RAW)
        rec = fixture('AiPEX'); rec.labels = {'hr': rec.labels['hr']}
        self.assertEqual(normalize_recording(rec).metadata['ppg_polarity']['status'], 'not_applicable_no_ppg')

    def test_morphology_after_adapter_and_orientation_failure(self):
        for ds, tag in [('MCD', 'MCD-rPPG'), ('UBFC2', 'UBFC2')]:
            records = []
            for i in range(3):
                rec = normalize_recording(fixture(tag))
                lab = rec.labels['ppg']
                records.append(dict(ppg=lab.values, ppg_t=lab.timestamps, ppg_valid=lab.valid_mask,
                                    ppg_polarity=rec.metadata['ppg_polarity']))
            report = assess_dataset(records, ds)
            self.assertEqual(report['status'], 'pass', report)
            # Deliberately corrupt canonical waveform sign; detector must catch it.
            corrupted = [{**r, 'ppg': -r['ppg']} for r in records]
            self.assertEqual(assess_dataset(corrupted, ds)['status'], 'fail')
            self.assertEqual(assess_dataset(records[:1], ds)['status'], 'unresolved')
            invalid = [{**r, 'ppg_valid': np.zeros(len(r['ppg']), bool)} for r in records]
            self.assertEqual(assess_dataset(invalid, ds)['status'], 'unresolved')


if __name__ == '__main__':
    unittest.main(verbosity=2)
