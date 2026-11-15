"""QA-lite on immutable E0 compressed video and its own saved compressed ROIs."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import fcntl
import hashlib
import json
import time
import traceback
from pathlib import Path
import av
import cv2
import numpy as np
from qa_mask import QALite
from hr_baseline import pos, estimate_hr
from mcd_e0_metrics import sample_grid

POLYS = [[109, 10, 338, 337, 151, 108], [50, 101, 205, 187, 123], [280, 330, 425, 411, 352]]


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
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def masks_for(shape, points, ids):
    masks = []
    for poly in POLYS:
        mask = np.zeros(shape, np.uint8)
        selected = points[[ids.index(i) for i in poly]]
        if np.isfinite(selected).all():
            cv2.fillPoly(mask, [np.rint(selected).astype(np.int32)], 1)
        masks.append(mask)
    return masks


def run_record(row, args, fingerprint, update):
    rid = Path(row['video']).stem
    source = args.e0/rid
    old = read(source/'result.json')
    trace = np.load(source/'trace.npz')
    encoded = source/'h264_500k.mp4'
    receipt = read(source/'encode.json')
    assert sha(encoded) == receipt['encoded_sha256']
    ids = trace['landmark_ids'].tolist()
    lm = trace['landmarks'][:, 1]  # ROI from compressed video, never native.
    baseline_saved = trace['rgb'][:, 3]
    n = len(lm)
    assert n == receipt['frames'] == len(baseline_saved)
    # Use the exact existing timestamp reader; no container-time substitution.
    from mcd_errata_audit import timed
    _, timestamps = timed(args.root/row['meta'])
    t = (timestamps[:n]-timestamps[0])/1e6
    assert len(t) == n and np.all(np.diff(t) > 0)
    rgb = np.full((n, 2, 3, 3), np.nan, np.float32)
    diagnostics = np.zeros((n, 5), np.float32)
    qa = QALite()
    seen = 0
    max_trace_error = 0.
    with av.open(str(encoded)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for frame in container.decode(video=0):
            if seen >= n or frame.is_corrupt:
                raise ValueError('Encoded frame count/corruption')
            image = frame.to_ndarray(format='rgb24')
            baseline, weighted, stats = qa.step(image, masks_for(image.shape[:2], lm[seen], ids))
            baseline = baseline.astype(np.float32)
            expected = baseline_saved[seen]
            if not np.array_equal(np.isfinite(baseline), np.isfinite(expected)):
                raise ValueError('Unweighted trace validity differs from E0')
            good = np.isfinite(baseline)
            if good.any():
                error = float(np.max(np.abs(baseline[good]-expected[good])))
                max_trace_error = max(max_trace_error, error)
                if error > 3e-5:
                    raise ValueError('Unweighted color differs from saved E0; preserve partial')
            rgb[seen, 0], rgb[seen, 1] = baseline, weighted
            diagnostics[seen] = [float(stats['gate']), stats['unchanged_fraction'],
                                 stats['noise_proxy_DN'] or 0., stats['mean_roi_weight'], stats['roi_pixels']]
            seen += 1
            if seen % 500 == 0:
                update(seen)
    assert seen == n
    np.savez_compressed(args.output/(rid+'.npz'), rgb=rgb, t=t, diagnostics=diagnostics,
        diagnostic_names=np.array(['gate', 'unchanged_fraction', 'noise_proxy_DN', 'mean_roi_weight', 'roi_pixels']))
    grid = np.arange(int(np.floor(t[-1]*30))+1)/30
    sampled = []
    for method in range(2):
        values = rgb[:, method].mean(axis=1)
        sampled.append(sample_grid(t, values, np.isfinite(values).all(1), grid))
    results = []
    for window in old['windows']:
        primary = window['reference_hr'] is not None and all(v is not None for v in window['predictions'].values())
        r = dict(subject=row['patient_id'], recording_id=rid, step=row['step'], start_s=window['start_s'],
            reference_hr=window['reference_hr'], primary_E0_paired=primary,
            reference_failure=window['reference_failure'], predictions={}, failures={})
        use = (grid >= r['start_s']) & (grid < r['start_s']+10)
        for method, (x, valid) in zip(('control', 'qa'), sampled):
            try:
                if use.sum() != 300 or not valid[use].all():
                    raise ValueError('Incomplete ROI/time support')
                r['predictions'][method] = estimate_hr(pos(x[use], 30), 30, band=(.7, 3.))
            except ValueError as exc:
                r['predictions'][method] = None
                r['failures'][method] = str(exc)
        previous = window['predictions']['P1R1']
        current = r['predictions']['control']
        if (previous is None) != (current is None) or (previous is not None and abs(previous-current) > 1e-6):
            raise ValueError('Unweighted HR differs from E0; stop before interpreting QA')
        results.append(r)
    return dict(recording_id=rid, fingerprint=fingerprint, rows=results, max_trace_error_DN=max_trace_error,
        diagnostics=dict(frames=n, gate_fraction=float(diagnostics[:, 0].mean()),
            unchanged_fraction_mean=float(diagnostics[:, 1].mean()),
            noise_proxy_DN_median=float(np.median(diagnostics[1:, 2])), mean_roi_weight=float(diagnostics[:, 3].mean())),
        encoded_sha256=receipt['encoded_sha256'], input_trace_sha256=sha(source/'trace.npz'),
        input_result_sha256=sha(source/'result.json'))


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
    assert all(sha(code/file) == digest for file, digest in frozen.items())
    cohort = read(a.e0/'cohort.json')
    assert len(cohort) == 119 and len({r['patient_id'] for r in cohort}) == 60
    assert all(r['patient_id'] != '4952' and not (r['patient_id'] == '7412' and r['step'] == 'after') for r in cohort)
    fingerprint = dict(code=frozen, e0_manifest_sha256=sha(a.e0/'run_manifest.json'), cohort_sha256=sha(a.e0/'cohort.json'))
    fp = a.output/'fingerprint.json'
    if fp.exists():
        assert read(fp) == fingerprint
    else:
        save(fp, fingerprint)
    status = dict(state='running', pid=os.getpid(), completed=0, total=119, errors=[])
    save(a.output/'status.json', status)
    for row in cohort[:a.limit] if a.limit else cohort:
        rid = Path(row['video']).stem
        status.update(recording=rid, frame=0)
        def update(frame):
            status.update(frame=frame, updated_unix=time.time())
            save(a.output/'status.json', status)
        update(0)
        try:
            dest = a.output/(rid+'.json')
            if dest.exists():
                assert read(dest)['fingerprint'] == fingerprint
            else:
                used = sum(p.stat().st_size for p in a.output.parent.rglob('*') if p.is_file())
                if used > 950_000_000:
                    raise RuntimeError('T6 combined output budget1GB')
                save(dest, run_record(row, a, fingerprint, update))
            status['completed'] += 1
            print(json.dumps(dict(recording=rid, completed=status['completed'])), flush=True)
            update(status['frame'])
        except Exception:
            status['state'] = 'blocked_error'
            status['errors'].append(dict(recording=rid, traceback=traceback.format_exc()))
            save(a.output/'status.json', status)
            raise
    status['state'] = 'smoke_complete' if a.limit else 'complete'
    save(a.output/'status.json', status)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'e0', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
