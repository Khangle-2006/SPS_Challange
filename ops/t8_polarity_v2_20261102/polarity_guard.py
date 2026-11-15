"""Fail-closed, CPU-only preflight helpers for the versioned correction."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

CAP = 65_000_000_000
HEADROOM = 100_000_000
EXPECTED_INDEX = '12a87b829ab5e158b430c67701fa0dd978c9a3d592e0060da3e876069108d505'
SEEDS = (2027, 2028, 2029)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_baseline_complete(original_runs, index_sha=EXPECTED_INDEX):
    root = Path(original_runs)
    status = read(root / 'queue_status.json')
    require(status.get('state') == 'complete' and status.get('runs') == 30,
            'BASELINE_WAIT: original 30-run queue not complete')
    require(status.get('index_sha256') == index_sha, 'Baseline queue index mismatch')
    receipts = []
    for fold in range(5):
        for seed in SEEDS:
            for arm in ('ubfc_only', 'joint'):
                dest = root / f'fold{fold}' / f'seed{seed}' / arm
                state = read(dest / 'status.json')
                require(state.get('state') == 'complete' and state.get('completed_epochs') == 30,
                        'BASELINE_WAIT: incomplete original run ' + str(dest))
                require(sha(dest / 'final.pt') == state['checkpoint_sha256'], 'Original final checkpoint mismatch')
                provenance = read(dest / 'provenance.json')
                require(provenance['index_sha256'] == index_sha, 'Original run index mismatch')
                config = provenance['config']
                require((config['fold'], config['seed'], config['arm'], config['epochs']) == (fold, seed, arm, 30),
                        'Original run config mismatch')
                history = read(dest / 'history.json')
                require([h['epoch'] for h in history] == list(range(1, 31)), 'Original history incomplete')
                receipts.append(dict(fold=fold, seed=seed, arm=arm, checkpoint_sha256=state['checkpoint_sha256'],
                                     history_sha256=sha(dest/'history.json'), provenance_sha256=sha(dest/'provenance.json')))
    return receipts


def acquire_original_lock(original_runs, flock=None):
    # r+ requires the original runner's existing lock; no baseline file creation.
    if flock is None:
        import fcntl
        flock = lambda fd: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handle = (Path(original_runs) / 'queue.lock').open('r+')
    try:
        flock(handle)
    except Exception as exc:
        handle.close()
        raise RuntimeError('BASELINE_WAIT: original queue lock unavailable') from exc
    return handle


def acquire_completed_baseline(original_runs, index_sha=EXPECTED_INDEX, flock=None):
    handle = acquire_original_lock(original_runs, flock)
    try:
        receipts = verify_baseline_complete(original_runs, index_sha)
    except Exception:
        handle.close()
        raise
    return handle, receipts


def distinct_roots(roots):
    # Containment is counted once. The corrected output root must be separate,
    # but legacy can already be nested under the original T8 root.
    resolved = sorted(set(Path(p).resolve() for p in roots), key=lambda p: len(p.parts))
    kept = []
    for path in resolved:
        if not any(path == ancestor or ancestor in path.parents for ancestor in kept):
            kept.append(path)
    return kept


def validate_output_path(original_root, corrected_root):
    original_root, corrected_root = Path(original_root).resolve(), Path(corrected_root).resolve()
    require(original_root != corrected_root and original_root not in corrected_root.parents
            and corrected_root not in original_root.parents, 'Corrected output must be separate from original T8 root')
    require(corrected_root.parent == original_root.parent,
            'Corrected output must be a direct sibling under the counted P100 project root')
    return original_root, corrected_root


def storage_guard(index, original_root, corrected_root):
    original_root, corrected_root = validate_output_path(original_root, corrected_root)
    for root in index['roots'].values():
        crop_root = Path(root).resolve()
        require(crop_root == original_root or original_root in crop_root.parents,
                'Unexpected cache outside original T8 root; refuse undercount')
    ubfc = read(Path(index['roots']['UBFC2']) / 'status.json')
    mcd = read(Path(index['roots']['MCD']) / 'status.json')
    legacy = Path(mcd['legacy_root'])
    require(legacy.is_dir(), 'Missing declared MCD legacy root')
    diagnostics = original_root.parent / 't8_label_diagnostic_20261102'
    require(diagnostics.is_dir(), 'Missing T8 diagnostics budget root')
    roots = distinct_roots([original_root, legacy, corrected_root, diagnostics])
    sizes = {}
    for root in roots:
        require(root.is_dir(), 'Missing budget root: ' + str(root))
        sizes[str(root)] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    origin = ubfc['bytes'] + ubfc['legacy_bytes']
    total = sum(sizes.values()) + origin
    require(total <= CAP - HEADROOM, 'STORAGE_WAIT: T8 65GB cap with 100MB headroom')
    return dict(counted_roots_bytes=sizes, A100_UBFC_origin_bytes_including_legacy=origin,
                total_bytes=total, approved_T8_cap_bytes=CAP, guard_headroom_bytes=HEADROOM,
                includes='Entire original T8 root, MCD legacy, diagnostics, separate corrected root, A100 origins; no crop double count')


def project_storage_guard(p100_root, receipt_path, require_fresh=False):
    """Fresh cross-server inventory supplied at launch; P100 remeasured locally.

    A100 figures must count all FaceVitals artifacts, not just T8 UBFC origins.
    No new A100 work is performed by this correction queue.
    """
    receipt = read(receipt_path)
    p100_root = Path(p100_root).resolve()
    require(Path(receipt['p100_root']).resolve() == p100_root, 'Project inventory P100 root mismatch')
    require(receipt['a100_root'] == '/data/khangle/facevitals', 'Project inventory A100 root mismatch')
    require(receipt['measurement_complete'] is True, 'Incomplete project inventory')
    measured = datetime.fromisoformat(receipt['measured_utc'].replace('Z', '+00:00'))
    require(measured.tzinfo is not None, 'Project inventory timestamp needs timezone')
    age = (datetime.now(timezone.utc) - measured).total_seconds()
    require(age >= -60, 'Project inventory timestamp in future')
    if require_fresh:
        require(age <= 3600, 'PROJECT_WAIT: project inventory older than one hour at launch')
    a100_bytes = receipt['a100_bytes']
    measured_p100 = receipt['p100_bytes']
    require(type(a100_bytes) is int and type(measured_p100) is int and min(a100_bytes, measured_p100) >= 0,
            'Invalid project inventory byte count')
    current_p100 = sum(p.stat().st_size for p in p100_root.rglob('*') if p.is_file())
    require(current_p100 + a100_bytes <= 500_000_000_000 - HEADROOM, 'PROJECT_WAIT: project500GB cap with100MB headroom')
    return dict(P100_bytes=current_p100, A100_bytes=a100_bytes, total_bytes=current_p100+a100_bytes,
                cap_bytes=500_000_000_000, receipt_sha256=sha(receipt_path), receipt_measured_utc=receipt['measured_utc'],
                A100_assumption='No A100 writes by this queue; remeasure externally if other jobs change A100 storage')


def baseline_pair(original_runs, fold, seed, pool, initial_model_sha256, baseline_code):
    pool_hash = hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()
    result = {}
    for arm in ('ubfc_only', 'joint'):
        dest = Path(original_runs) / f'fold{fold}' / f'seed{seed}' / arm
        provenance = read(dest / 'provenance.json')
        require(provenance['initial_model_sha256'] == initial_model_sha256, 'Initial model differs from original '+arm)
        require(provenance['train_pool_sha256'] == pool_hash, 'Train pool differs from original '+arm)
        require(provenance['code_sha256'] == baseline_code, 'Baseline source fingerprint mismatch')
        result[arm] = read(dest / 'history.json')
    return result


def record_dataset_mse(pred, target, selections, losses):
    """Detached logging only: never touches the optimization graph or RNG."""
    values = (pred.detach() - target.detach()).square().mean(dim=1).cpu().tolist()
    for selection, value in zip(selections, values):
        losses[selection[0]].append(float(value))
