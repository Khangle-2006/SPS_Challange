"""Canonical backend waveform references; exact frozen HR decoder/scorer and image path."""
import json
from pathlib import Path
import numpy as np
from t8_plan import window_key, write, sha
from t8_evaluate import reference_row, decode_difference, csv_write
from score_round1 import reference_hr, score
from polarity_loader import load_canonical_clip
from ppg_polarity import adapt_cached_meta, CONVENTION, POLICY_SHA256


def reference_outcome(row):
    try:
        return {'hr_bpm': reference_hr(row), 'failure': None}
    except ValueError as exc:
        return {'hr_bpm': None, 'failure': str(exc)}


def verify_reference_parity(raw, canonical, sign):
    if sign not in (-1, 1):
        raise RuntimeError('Unsupported reference sign')
    if set(raw) != set(canonical):
        raise RuntimeError('Reference fields changed')
    for key in raw:
        if key != 'ppg_values_json' and raw[key] != canonical[key]:
            raise RuntimeError('Reference identity/time/support changed: '+key)
    a = np.asarray(json.loads(raw['ppg_values_json']), float)
    b = np.asarray(json.loads(canonical['ppg_values_json']), float)
    if not np.array_equal(b, sign*a, equal_nan=True):
        raise RuntimeError('Reference waveform differs beyond declared global sign')
    before, after = reference_outcome(raw), reference_outcome(canonical)
    if before != after:
        raise RuntimeError('Reference HR/failure parity changed')
    return dict(recording_id=raw['recording_id'], window_start_s=raw['window_start_s'],
                sign=sign, exact_time_support_and_waveform_sign=True, **after)


def evaluate(model, index, fold_id, out, device):
    import torch
    out = Path(out)
    out.mkdir(exist_ok=False)
    fold = index['folds']['folds'][fold_id]
    results = {}
    model.eval()
    for dataset, prefix in [('UBFC2', 'ubfc'), ('MCD', 'mcd')]:
        records = [r for r in index['records'] if r['dataset'] == dataset
                   and r['subject_id'] in fold[prefix+'_val_subjects']]
        for condition in ('native', 'f10_b500'):
            dest = out / (dataset+'_'+condition)
            dest.mkdir()
            refs, raw_refs, parity_rows, jobs = [], [], [], []
            reference_receipts = {}
            for record in records:
                root = Path(index['roots'][dataset]) / record['recording_id']
                with np.load(root/'native'/'crop'/'meta.npz') as meta:
                    canonical_meta, normalization = adapt_cached_meta(meta, dataset, source_convention='raw_device')
                    reference_receipts[record['recording_id']] = dict(metadata_sha256=sha(root/'native'/'crop'/'meta.npz'), normalization=normalization)
                    for w in record['candidates']:
                        raw_ref = reference_row(record, w, meta, condition)
                        canonical_ref = reference_row(record, w, canonical_meta, condition)
                        parity_rows.append(verify_reference_parity(raw_ref, canonical_ref, normalization['raw_to_canonical_sign']))
                        raw_refs.append(raw_ref)
                        refs.append(canonical_ref)
                        jobs.append((record, w, root/condition/'crop'))
            # Written independently before any prediction or loader outcome.
            csv_write(dest/'reference_signal.csv', refs)
            csv_write(dest/'raw_reference_signal.csv', raw_refs)
            write(dest/'REFERENCE_POLARITY.json', dict(convention=CONVENTION, policy_sha256=POLICY_SHA256,
                raw_reference_sha256=sha(dest/'raw_reference_signal.csv'), canonical_reference_sha256=sha(dest/'reference_signal.csv'),
                normalization_receipts=reference_receipts, window_parity=parity_rows, all_hr_and_failure_parity=True,
                aggregation_contract='Raw control vs canonical joint: verify exact window/nonwave fields, samples with global sign, and HR/failure parity; never waive arbitrary mismatched hashes.'))
            outputs, failures, waves, wave_ids = [], [], [], []
            for ref, (record, w, crop) in zip(refs, jobs):
                row = {k: ref[k] for k in ('subject_id', 'recording_id', 'window_start_s')}
                row.update(hr_bpm='', abstain=True)
                try:
                    if window_key(w) not in record['source_valid_starts']:
                        raise ValueError('canonical_source_rejected')
                    if not record['evaluation_usable'][condition][str(window_key(w))]:
                        raise ValueError('variant_image_or_ppg_support')
                    x, _, _, _ = load_canonical_clip(crop, float(w['start_s']), dataset)
                    with torch.no_grad():
                        wave = model(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
                    row.update(hr_bpm=decode_difference(wave), abstain=False)
                    waves.append(wave)
                    wave_ids.append([record['recording_id'], str(w['start_s'])])
                except ValueError as exc:
                    failures.append({**row, 'reason': str(exc)})
                except Exception as exc:
                    # Preserve completed predictions before surfacing implementation/OOM errors.
                    if outputs:
                        csv_write(dest/'partial_processed_output.csv', outputs)
                    np.savez_compressed(dest/'partial_waveforms.npz', difference=np.asarray(waves), keys=np.asarray(wave_ids))
                    write(dest/'implementation_error.json', dict(window=row, error=str(exc)))
                    raise
                # RuntimeError/OOM/corrupt cache are implementation failures: stop, do not hide.
                outputs.append(row)
            csv_write(dest/'processed_output.csv', outputs)
            np.savez_compressed(dest/'waveforms.npz', difference=np.asarray(waves), keys=np.asarray(wave_ids))
            write(dest/'abstentions.json', failures)
            metrics = score(dest/'reference_signal.csv', dest/'processed_output.csv')
            write(dest/'metrics.json', metrics)
            results[dataset+'_'+condition] = metrics
    write(out/'summary.json', results)
    return results
