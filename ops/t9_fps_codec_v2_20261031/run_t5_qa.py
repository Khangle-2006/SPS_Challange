"""Read completed T5 C5 receipts, keep frozen QA weights and independent clocks."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import csv
import fcntl
import hashlib
import json
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
from qa_mask import QALite
from oval_frontend import OvalFrontend,verify_control
from data_adapters import MCD, UBFC2
from window_support import recording_windows
from t8_t1_loader import supported
from hr_baseline import pos, estimate_hr
from score_round1 import reference_hr, score


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows):
    assert rows
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def approved_ids(dataset, folds, mcd):
    if dataset == 'UBFC2':
        first = folds['folds'][0]
        ids = sorted(set(first['ubfc_train_subjects']+first['ubfc_val_subjects']))
        assert len(ids) == 35 and set(ids).isdisjoint(folds['original_ubfc_test_subjects'])
        return ids
    ids = [Path(r['video']).stem for r in mcd]
    assert len(ids) == len(set(ids)) == 119 and len({r['patient_id'] for r in mcd}) == 60
    assert all(r['camera'] == 'FullHDwebcam' and r['patient_id'] != '4952' and
               not (r['patient_id'] == '7412' and r['step'] == 'after') for r in mcd)
    return ids


def references(record, canonical):
    label = record.require_synchronized_ppg()
    t, p = label.timestamps, label.values.copy()
    p[~label.valid_mask] = np.nan
    rows = []
    for w in sorted(canonical['valid_windows']+canonical['rejected_windows'], key=lambda w:w['start_s']):
        start = w['start_s']
        left = max(0, int(np.searchsorted(t, start, side='right'))-1)
        right = min(len(t), int(np.searchsorted(t, start+10, side='left'))+1)
        while left > 0 and t[left-1] == t[left]:
            left -= 1
        while right < len(t) and t[right] == t[right-1]:
            right += 1
        rows.append(dict(subject_id=record.subject_id, recording_id=record.recording_id,
            window_start_s=start, video_support_valid=not any(r.startswith('video_') for r in w['reasons']),
            ppg_timestamps_json=json.dumps(t[left:right].tolist()),
            ppg_values_json=json.dumps([float(x) if np.isfinite(x) else None for x in p[left:right]]),
            source_reasons='|'.join(w['reasons']), split='development', condition='c5_500k_10fps'))
    return rows


def merge(t, values, valid):
    u, inv, counts = np.unique(t, return_inverse=True, return_counts=True)
    out = np.zeros((len(u),)+values.shape[1:], dtype=np.float64)
    np.add.at(out, inv, np.nan_to_num(values))
    out /= counts.reshape((-1,)+(1,)*(values.ndim-1))
    ok = np.ones(len(u), bool)
    np.logical_and.at(ok, inv, valid)
    return u, out, ok


def mask_for(shape, box, valid):
    mask = np.zeros(shape, np.uint8)
    if valid:
        x1, y1, x2, y2 = [int(v) for v in box]
        if not (0 <= x1 < x2 <= shape[1] and 0 <= y1 < y2 <= shape[0]):
            raise ValueError('Invalid saved compressed ROI bounds')
        mask[y1:y2, x1:x2] = 1
    return mask


def classify(manifest):
    return bool(manifest['kind'] == 'c5' and manifest['fps_requested'] == 10 and
        manifest['mux_fps'] == 10 and manifest['bitrate_limit_kbps'] == 500 and
        manifest['bitrate_pass'] and manifest['actual']['whole_file_kbps'] <= 500)


def process(record, canonical, source, dest, fingerprint, update):
    reference_hash = sha(dest/'reference_signal.csv')
    names = ['evaluation.json', 'manifest.json', 'trace.npz', 'frame_mapping.npz']
    input_hashes = {name: sha(source/name) for name in names}
    manifest = read(source/'manifest.json')
    old_eval = read(source/'evaluation.json')
    assert old_eval['recording_id'] == record.recording_id and old_eval['subject_id'] == record.subject_id
    assert old_eval['protocol_sha256'] == manifest['protocol_sha256']
    assert manifest['kind'] == 'c5' and manifest['fps_requested'] == manifest['mux_fps'] == 10
    video = Path(manifest['video'])
    assert video.parent.resolve() == source.resolve(), 'Do not read an unexpected video path'
    input_hashes['video'] = sha(video)
    z, mapping = np.load(source/'trace.npz'), np.load(source/'frame_mapping.npz')
    t, indices = mapping['timestamps_s'], mapping['source_frame_indices']
    n = len(t)
    assert n == len(indices) == manifest['selected_frames']
    assert len(indices) > 1 and np.all(np.diff(indices) > 0) and indices[0] >= 0 and indices[-1] < len(record.timestamps)
    np.testing.assert_array_equal(z['source_frame_indices'], indices)
    np.testing.assert_allclose(z['timestamps_s'], t, rtol=0, atol=1e-9)
    np.testing.assert_allclose(record.timestamps[indices], t, rtol=0, atol=1e-9)
    assert len(z['rgb']) == len(z['valid']) == len(z['boxes']) == n
    rgb = np.full((n, 2, 3), np.nan, np.float64)
    diagnostic = np.zeros((n, 4), np.float32)
    qa, seen, max_error = QALite(), 0, 0.
    frontend=OvalFrontend()
    oval_points=np.full((n,36,2),np.nan,np.float32)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot open completed T5 video')
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if seen >= n:
                raise ValueError('Unexpected extra decoded frame')
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask,oval_points[seen]=frontend.step(image)
            baseline, weighted, stats = qa.step(image, [mask])
            baseline, weighted = baseline[0], weighted[0]
            max_error=max(max_error,verify_control(image,mask,baseline,weighted,stats))
            rgb[seen] = [baseline, weighted]
            diagnostic[seen] = [stats['gate'], stats['unchanged_fraction'], stats['noise_proxy_DN'] or 0., stats['mean_roi_weight']]
            seen += 1
            if seen % 500 == 0:
                update(seen)
    finally:
        cap.release()
        frontend.close()
        if seen:
            np.savez_compressed(dest/'partial_trace.npz', rgb=rgb[:seen], timestamps_s=t[:seen], diagnostics=diagnostic[:seen],oval_points=oval_points[:seen])
    assert seen == n, f'Incomplete decode {seen}/{n}'
    (dest/'partial_trace.npz').rename(dest/'trace.npz')
    frame_valid = np.asarray(record.metadata.get('frame_valid', np.ones(len(record.timestamps), bool)))[indices]
    tv, values, valid = merge(t, rgb, np.isfinite(rgb).all(axis=(1,2)) & frame_valid)
    canonical_rows = references(record, canonical)
    # Independent reference CSV was already exported before image processing.
    assert sha(dest/'reference_signal.csv') == reference_hash
    rows, outputs, waveforms = [], {'control': [], 'qa': []}, {'control': [], 'qa': []}
    valid_starts = {round(w['start_s'], 9) for w in canonical['valid_windows']}
    replay_max = 0.
    for ref in canonical_rows:
        start = ref['window_start_s']
        rr = dict(subject=record.subject_id, recording_id=record.recording_id, start_s=start,
            step=record.metadata.get('step', 'unspecified'), reference_hr=None, reference_failure=None,
            predictions={}, failures={}, C5_compliant=classify(manifest))
        try:
            if not ref['video_support_valid']:
                raise ValueError('invalid_video_support')
            rr['reference_hr'] = reference_hr(ref)
        except ValueError as exc:
            rr['reference_failure'] = str(exc)
        q = start+np.arange(300)/30
        for arm, name in enumerate(('control', 'qa')):
            rr['predictions'][name] = None
            try:
                if round(start, 9) not in valid_starts:
                    raise ValueError('Canonical source invalid')
                if not supported(tv, valid & np.isfinite(values[:, arm]).all(1), start, start+10):
                    raise ValueError('Video support/gap/ROI')
                sampled = np.stack([np.interp(q, tv, values[:, arm, c]) for c in range(3)], axis=1)
                wave = pos(sampled, 30)
                hr = estimate_hr(wave, 30, band=(.7, 3.))
                rr['predictions'][name] = hr
                waveforms[name].append((start, wave))
            except ValueError as exc:
                rr['failures'][name] = str(exc)
            outputs[name].append(dict(subject_id=record.subject_id, recording_id=record.recording_id,
                window_start_s=start, hr_bpm=rr['predictions'][name] if rr['predictions'][name] is not None else '',
                abstain=rr['predictions'][name] is None))
        rows.append(rr)
    metrics = {}
    for name in ('control', 'qa'):
        csv_write(dest/(name+'_output.csv'), outputs[name])
        np.savez_compressed(dest/(name+'_waveforms.npz'), starts=np.asarray([p[0] for p in waveforms[name]]),
                            waves=np.asarray([p[1] for p in waveforms[name]]))
        metrics[name] = score(dest/'reference_signal.csv', dest/(name+'_output.csv'))
    assert all(sha(source/name) == digest for name, digest in input_hashes.items() if name != 'video')
    return dict(fingerprint=fingerprint, recording_id=record.recording_id, subject_id=record.subject_id,
        rows=rows, metrics=metrics, input_hashes=input_hashes, reference_sha256=reference_hash, C5_compliant=classify(manifest),
        bitrate_kbps=manifest['actual']['whole_file_kbps'], requested_fps=10, mux_fps=manifest['mux_fps'],
        effective_sensor_fps=manifest['effective_sensor_fps'], max_control_RGB_error_DN=max_error,
        frontend='compressed_large_oval_v2', control_validation='independent_masked_RGB_mean_and_gate_closed_identity',
        diagnostics=dict(frames=n, gate_fraction=float(diagnostic[:, 0].mean()),
            unchanged_fraction_mean=float(diagnostic[:, 1].mean()), noise_proxy_DN_median=float(np.median(diagnostic[1:, 2])),
            mean_roi_weight=float(diagnostic[:, 3].mean())))


def main(a):
    cv2.setNumThreads(1)
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if hasattr(os, 'sched_setaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {allowed[min(2, len(allowed)-1)]})
    code = Path(__file__).parent
    frozen = read(code/'SHA256.json')
    assert all(sha(code/name) == digest for name, digest in frozen.items())
    backend = MCD(a.root, a.audit) if a.dataset == 'MCD' else UBFC2(a.root)
    ids = approved_ids(a.dataset, read(code/'folds_frozen.json'), read(code/'mcd_cohort.json'))
    fingerprint = dict(code=frozen, dataset=a.dataset, T5_protocol_sha256=sha(a.t5/'protocol.json'),frontend='compressed_large_oval_v2')
    if (a.output/'fingerprint.json').exists():
        assert read(a.output/'fingerprint.json') == fingerprint
    else:
        save(a.output/'fingerprint.json', fingerprint)
    status = dict(state='preparing_manifest', pid=os.getpid(), total=len(ids), completed=0, pending=[], errors=[])
    save(a.output/'status.json', status)
    # Generate all expected windows from original T1 data before availability/predictions.
    for rid in ids:
        dest = a.output/rid
        dest.mkdir(exist_ok=True)
        record = backend.load(rid)
        canonical = recording_windows(record)
        if (dest/'canonical_windows.json').exists():
            assert read(dest/'canonical_windows.json') == canonical
        else:
            save(dest/'canonical_windows.json', canonical)
            csv_write(dest/'reference_signal.csv', references(record, canonical))
    save(a.output/'cohort.json', dict(dataset=a.dataset, recordings=ids, expected_records=len(ids),
        expected_windows=sum(read(a.output/rid/'canonical_windows.json')['candidate_windows'] for rid in ids)))
    while True:
        status.update(state='running', pending=[], completed=0)
        for rid in ids:
            dest, source = a.output/rid, a.t5/'c5_500k_10fps'/rid
            status.update(recording=rid, frame=0)
            def update(frame):
                status.update(frame=frame, updated_unix=time.time())
                save(a.output/'status.json', status)
            update(0)
            if (dest/'complete.json').exists():
                assert read(dest/'complete.json')['fingerprint'] == fingerprint
                status['completed'] += 1
                continue
            required = ('evaluation.json', 'manifest.json', 'trace.npz', 'frame_mapping.npz')
            if not all((source/f).exists() for f in required):
                status['pending'].append(dict(recording=rid, source_failed=(source/'failure.json').exists()))
                continue
            try:
                if (dest/'trace.npz').exists() or (dest/'partial_trace.npz').exists():
                    raise RuntimeError('Partial result exists; inspect/version continuation, do not overwrite')
                used = sum(p.stat().st_size for p in a.output.parent.rglob('*') if p.is_file())
                cap = 150_000_000 if a.dataset == 'UBFC2' else 850_000_000
                if used > cap-10_000_000:
                    raise RuntimeError('T6 storage reservation150MBA100/850MBP100 includingE0')
                record = backend.load(rid)
                result = process(record, read(dest/'canonical_windows.json'), source, dest, fingerprint, update)
                save(dest/'complete.json', result)
                status['completed'] += 1
                print(json.dumps(dict(recording=rid, completed=status['completed'], total=len(ids))), flush=True)
                if a.limit and status['completed'] >= a.limit:
                    status['state'] = 'smoke_complete'
                    save(a.output/'status.json', status)
                    return
            except Exception:
                status['state'] = 'blocked_error'
                status['errors'].append(dict(recording=rid, traceback=traceback.format_exc()))
                save(a.output/'status.json', status)
                raise
        if status['completed'] == len(ids):
            status['state'] = 'complete'
        else:
            parent = read(a.t5/'status.json')['state']
            status['state'] = 'blocked_missing_T5' if parent.startswith('complete') else 'waiting_T5_receipts'
        save(a.output/'status.json', status)
        if status['state'] != 'waiting_T5_receipts' or not a.watch:
            return
        time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 't5', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--audit', type=Path)
    p.add_argument('--dataset', choices=('MCD', 'UBFC2'), required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
