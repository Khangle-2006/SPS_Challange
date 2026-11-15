"""Frozen T8 cohorts, verified cache index and paired deterministic draws.

This module has no torch dependency and never discovers subjects from raw folders.
"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

VARIANTS = ['native'] + [f'f{f}_b{b}' for f in (10, 15, 30) for b in (300, 500, 800)]
SEEDS = (2027, 2028, 2029)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def validate_folds(spec):
    folds = spec['folds']
    assert [f['fold'] for f in folds] == list(range(5))
    held_test = set(spec['original_ubfc_test_subjects'])
    assert len(held_test) == 7
    populations = {}
    for name, ntrain, nval in [('mcd', 48, 12), ('ubfc', 28, 7)]:
        val_all = []
        population = None
        for fold in folds:
            train = fold[name + '_train_subjects']
            val = fold[name + '_val_subjects']
            assert len(train) == len(set(train)) == ntrain
            assert len(val) == len(set(val)) == nval
            assert set(train).isdisjoint(val)
            current = set(train) | set(val)
            if population is None:
                population = current
            assert current == population
            if name == 'ubfc':
                assert current.isdisjoint(held_test)
            else:
                assert '4952' not in current
            val_all.extend(val)
        assert set(val_all) == population and max(Counter(val_all).values()) == 1
        populations['MCD' if name == 'mcd' else 'UBFC2'] = population
    return populations


def window_key(w):
    return round(float(w['start_s']), 9)


def build_index(roots, spec, verify_hashes=True):
    """Fail closed on incomplete caches. Preserve every canonical evaluation row."""
    populations = validate_folds(spec)
    index = {'version': 't8-index-v1', 'records': [], 'excluded_training_windows': [],
             'roots': {k: str(Path(v).resolve()) for k, v in roots.items()},
             'folds': spec, 'hashes_verified': bool(verify_hashes), 'cache_fingerprints': {}}
    counts = {}
    for dataset in ('UBFC2', 'MCD'):
        root = Path(roots[dataset])
        inventory, status = read(root / 'cohort.json'), read(root / 'status.json')
        assert inventory['dataset'] == dataset and inventory['variants'] == VARIANTS
        assert len(inventory['recordings']) == (35 if dataset == 'UBFC2' else 119)
        assert len(set(inventory['recordings'])) == len(inventory['recordings'])
        assert status['state'] == 'complete' and not status['errors']
        assert status['completed'] == status['total'] == inventory['expected_items']
        assert inventory['expected_items'] == len(inventory['recordings']) * 10
        fingerprint = read(root / 'fingerprint.json')
        index['cache_fingerprints'][dataset] = fingerprint
        seen_subjects, nvalid, ntrain = set(), 0, 0
        for rid in inventory['recordings']:
            canonical = read(root / rid / 'canonical_windows.json')
            candidates = canonical['valid_windows'] + canonical['rejected_windows']
            assert len({window_key(w) for w in candidates}) == len(candidates)
            valid_keys = {window_key(w) for w in canonical['valid_windows']}
            per_variant, subject = {}, None
            for variant in VARIANTS:
                receipt_path = root / rid / variant / 'complete.json'
                receipt = read(receipt_path)
                assert receipt['dataset'] == dataset and receipt['recording_id'] == rid
                assert receipt['variant'] == variant and receipt['fingerprint'] == fingerprint
                namespaced_sid = str(receipt['subject_id'])
                required_prefix = 'mcd:' if dataset == 'MCD' else 'ubfc2:'
                assert namespaced_sid.startswith(required_prefix), namespaced_sid
                sid = namespaced_sid[len(required_prefix):]
                if subject is None:
                    subject = sid
                assert sid == subject and sid in populations[dataset]
                assert not (dataset == 'MCD' and (sid == '4952' or (sid == '7412' and rid.endswith('_after'))))
                if variant != 'native':
                    assert receipt['encoding']['bitrate_pass']
                if verify_hashes:
                    for file, digest in receipt['cache_files'].items():
                        assert sha(root / rid / variant / 'crop' / file) == digest, (rid, variant, file)
                windows = receipt['canonical_valid_windows']
                assert len(windows) == len(valid_keys)
                assert {window_key(w) for w in windows} == valid_keys
                per_variant[variant] = {window_key(w): bool(w['usable']) for w in windows}
            seen_subjects.add(subject)
            eligible = []
            for w in canonical['valid_windows']:
                bad = [v for v in VARIANTS if not per_variant[v][window_key(w)]]
                if bad:
                    index['excluded_training_windows'].append(dict(dataset=dataset, recording_id=rid,
                        subject_id=subject, start_s=w['start_s'], unusable_variants=bad))
                else:
                    eligible.append(float(w['start_s']))
            nvalid += len(valid_keys)
            ntrain += len(eligible)
            index['records'].append(dict(dataset=dataset, subject_id=subject, recording_id=rid,
                canonical_sha256=sha(root / rid / 'canonical_windows.json'),
                eligible_starts=eligible, candidates=sorted(candidates, key=window_key),
                source_valid_starts=sorted(valid_keys),
                evaluation_usable={v: {str(k): ok for k, ok in per_variant[v].items()}
                                   for v in ('native', 'f10_b500')}))
        assert seen_subjects == populations[dataset]
        counts[dataset] = dict(records=len(inventory['recordings']), subjects=len(seen_subjects),
                               canonical_valid=nvalid, train_eligible=ntrain, excluded=nvalid-ntrain)
    index['coverage'] = counts
    return index


def pools(index, fold_id):
    fold = index['folds']['folds'][fold_id]
    result = {}
    for dataset, prefix in [('UBFC2', 'ubfc'), ('MCD', 'mcd')]:
        train, val = set(fold[prefix+'_train_subjects']), set(fold[prefix+'_val_subjects'])
        rows = [(r['recording_id'], start) for r in index['records']
                if r['dataset'] == dataset and r['subject_id'] in train for start in r['eligible_starts']]
        assert rows and train.isdisjoint(val)
        result[dataset] = sorted(rows)
    return result


def steps_per_epoch(pool):
    return math.ceil((len(pool['UBFC2']) + len(pool['MCD'])) / 2)


def draw(pool, seed, fold, epoch, step, arm):
    """Same condition sequence and first UBFC draw in both arms, independent RNGs.

    Separate per-slot generators prevent pool size from changing codec draws.
    """
    assert arm in ('joint', 'ubfc_only')
    batch = []
    for slot in range(2):
        dataset = 'MCD' if arm == 'joint' and slot == 1 else 'UBFC2'
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold, epoch, step, slot, 0]))
        row = pool[dataset][int(rng.integers(len(pool[dataset])))]
        codec_rng = np.random.default_rng(np.random.SeedSequence([seed, fold, epoch, step, slot, 1]))
        variant = 'native' if codec_rng.random() < .5 else VARIANTS[1+int(codec_rng.integers(9))]
        batch.append((dataset, row[0], row[1], variant))
    return batch


def code_fingerprint(directory):
    directory = Path(directory)
    return {p.name: sha(p) for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix in ('.py', '.txt', '.json') and p.name != 'FROZEN_SHA256.json'}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ubfc', type=Path, required=True)
    p.add_argument('--mcd', type=Path, required=True)
    p.add_argument('--folds', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    write(a.output, build_index({'UBFC2': a.ubfc, 'MCD': a.mcd}, read(a.folds)))
