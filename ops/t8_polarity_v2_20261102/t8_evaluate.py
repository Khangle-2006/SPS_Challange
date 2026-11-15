"""All canonical rows exported before inference; failures remain abstentions."""
import csv
import json
from pathlib import Path
import numpy as np
from t8_plan import window_key, write
from t8_t1_loader import load_clip
from score_round1 import reference_hr, score


def reference_row(record, window, metadata, condition):
    start = float(window['start_s'])
    t, p = metadata['ppg_t'], metadata['ppg'].astype(float).copy()
    p[~metadata['ppg_valid'].astype(bool)] = np.nan
    lo = max(0, int(np.searchsorted(t, start, side='right'))-1)
    hi = min(len(t), int(np.searchsorted(t, start+10, side='left'))+1)
    reasons = window.get('reasons', [])
    return dict(subject_id=record['dataset']+':'+record['subject_id'],
        recording_id=record['recording_id'], window_start_s=start,
        video_support_valid=not any(r.startswith('video_') for r in reasons),
        ppg_timestamps_json=json.dumps(t[lo:hi].tolist()),
        ppg_values_json=json.dumps([float(x) if np.isfinite(x) else None for x in p[lo:hi]]),
        source_reasons='|'.join(reasons), split='development_OOF', condition=condition)


def decode_difference(prediction):
    y = np.asarray(prediction, dtype=float)
    if y.shape != (300,) or not np.isfinite(y).all() or y.std() < 1e-10:
        raise ValueError('invalid_prediction_waveform')
    # score_round1 applies the exact frozen detrend/filter/argmax operator.
    wave = np.cumsum(y)
    return reference_hr(dict(window_start_s=0.,
        ppg_timestamps_json=json.dumps((np.arange(301)/30).tolist()),
        ppg_values_json=json.dumps(np.r_[wave, wave[-1]].tolist())))


def csv_write(path, rows):
    assert rows
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            refs, jobs = [], []
            for record in records:
                root = Path(index['roots'][dataset]) / record['recording_id']
                with np.load(root/'native'/'crop'/'meta.npz') as meta:
                    for w in record['candidates']:
                        refs.append(reference_row(record, w, meta, condition))
                        jobs.append((record, w, root/condition/'crop'))
            # Written independently before any prediction or loader outcome.
            csv_write(dest/'reference_signal.csv', refs)
            outputs, failures, waves, wave_ids = [], [], [], []
            for ref, (record, w, crop) in zip(refs, jobs):
                row = {k: ref[k] for k in ('subject_id', 'recording_id', 'window_start_s')}
                row.update(hr_bpm='', abstain=True)
                try:
                    if window_key(w) not in record['source_valid_starts']:
                        raise ValueError('canonical_source_rejected')
                    if not record['evaluation_usable'][condition][str(window_key(w))]:
                        raise ValueError('variant_image_or_ppg_support')
                    x, _, _ = load_clip(crop, float(w['start_s']))
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
