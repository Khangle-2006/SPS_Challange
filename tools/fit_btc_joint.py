"""Fit/export the frozen BTC joint recipe from a verified dev bundle into a NEW directory."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
import numpy as np
import scipy
from joint_core import load_fit, read_spectra, windows
from reader_v4 import MODELS
from training.btc_joint.assets import META, NATIVE, VIDEOS, load_fit_inputs, resolve, sha
from training.btc_joint.fit import fit_models


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def difference(actual, expected, path='model'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('Model fields differ: ' + path)
        return max((difference(actual[k], expected[k], path + '.' + k) for k in expected), default=0.)
    if isinstance(expected, str):
        if actual != expected:
            raise ValueError('Model recipe differs: ' + path)
        return 0.
    try:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    except (TypeError, ValueError):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Model structure differs: ' + path)
        return max((difference(x, y, path) for x, y in zip(actual, expected)), default=0.)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Model array differs: ' + path)
    if not np.allclose(a, b, atol=1e-9, rtol=1e-9):
        raise ValueError(f'Model values differ: {path}, max delta {np.max(abs(a-b))}')
    return float(np.max(abs(a - b))) if a.size else 0.


def verify_predictions(reports, models):
    rows = json.loads(resolve(reports, VIDEOS).read_bytes())
    with resolve(reports, META).open(newline='', encoding='utf-8-sig') as f:
        metadata = {r['recording']: r for r in csv.DictReader(f)}
    with np.load(resolve(reports, NATIVE), allow_pickle=False) as z:
        experts = {m: z[m] for m in MODELS}
    saved = {f: load_fit(f)[0] for f in [None, *range(5)]}
    worst, branch_changes = 0., 0
    errors, people = [], []
    for i, row in enumerate(rows):
        name = row['recording']
        m = metadata[name]
        front = dict(source_bitrate=int(m['bit_rate']), source_width=int(m['width']),
                     source_height=int(m['height']), source_fps=float(m['avg_fps']), source_codec=m['codec'])
        expert = {k: experts[k][i] for k in MODELS}
        with np.load(resolve(reports, 'w2_btc_20261116/export_extracted/blocks/' + name + '.npz'), allow_pickle=False) as trace:
            measured = windows(trace['t'], trace['valid'], trace['rgb_mean'], 10, 2.5)
            if not any(start % 5 == 0 for start, _ in measured):
                raise ValueError('Dev recording lacks required classical support')
            chrom, pos = np.mean([p for _, p in measured], axis=0)
            for fold in (row['fold'], None):
                suffix = 'deploy' if fold is None else str(fold)
                fitted = {kind: models[f'{kind}_{suffix}.json'] for kind in ('router', 'fallback')}
                args = chrom, pos, expert, trace, front, row['task'].startswith('Task')
                actual, branch = read_spectra(fitted, *args)
                expected, saved_branch = read_spectra(saved[fold], *args)
                worst = max(worst, abs(actual - expected))
                branch_changes += int(branch != saved_branch)
                if fold is not None:
                    errors.append(abs(actual - row['pr_bpm']))
                    people.append(row['person'])
        if (i + 1) % 50 == 0:
            print(f'Compared inference on {i+1}/{len(rows)} dev videos', flush=True)
    if worst > 1e-9 or branch_changes:
        raise ValueError(f'Inference parity failed: max delta {worst}, branch changes {branch_changes}')
    errors, people = np.asarray(errors), np.asarray(people)
    mae = float(np.mean([errors[people == p].mean() for p in np.unique(people)]))
    return dict(videos=len(rows), people=len(np.unique(people)), comparisons=2*len(rows),
                oof_person_mae=mae, max_prediction_delta=worst, branch_changes=branch_changes,
                deploy_scores_used_for_evaluation=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--assets-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-shipped', action='store_true', help='Require parameter and OOF/deploy inference parity with all 12 shipped fits')
    args = p.parse_args()
    target = args.output.resolve()
    if target.exists() or target == ROOT / 'models' or ROOT / 'models' in target.parents:
        p.error('Output must be a new directory outside shipped models; no overwrite')
    if (np.__version__, scipy.__version__) != ('2.5.4', '1.18.1'):
        p.error('Frozen fitting requires NumPy 2.5.4 / SciPy 1.18.1. Run tools/setup_env.py joint-fit, then use that interpreter; inference core has different pins.')
    try:
        data = load_fit_inputs(args.reports, args.assets_manifest)
        models = plain(fit_models(data, progress=lambda x: print(x, flush=True)))
        receipt = dict(recipe='Frozen BTC joint', scope='dev-only', refit=True,
                       python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                       new_model_selection=False, holdout_reads=0, assets_manifest_sha256=sha(args.assets_manifest),
                       verified_against_shipped=args.verify_shipped,
                       source_sha256={str(f.relative_to(ROOT)): sha(f) for f in
                           [Path(__file__).resolve(), *sorted((ROOT/'training/btc_joint').glob('*.py')), *sorted((ROOT/'runtime').glob('*.py'))]})
        if args.verify_shipped:
            shipped_manifest = json.loads((ROOT / 'models/btc_joint/manifest.json').read_bytes())
            deltas = {}
            for name, actual in models.items():
                path = ROOT / 'models/btc_joint' / name
                if sha(path) != shipped_manifest[name]:
                    raise ValueError('Shipped model hash changed: ' + name)
                deltas[name] = difference(actual, json.loads(path.read_bytes()), name)
            receipt['max_parameter_delta'] = max(deltas.values())
            receipt['parameters_by_file'] = deltas
            receipt['inference'] = verify_predictions(args.reports, models)
        target.mkdir(parents=True, exist_ok=False)
        (target / '.gitignore').write_text('*\n', encoding='utf-8')
        model_dir = target / 'models'
        model_dir.mkdir()
        for name, model in models.items():
            (model_dir / name).write_text(json.dumps(model, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        hashes = {name: sha(model_dir / name) for name in sorted(models)}
        (model_dir / 'manifest.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
        receipt['export_sha256'] = hashes
        (target / 'FIT_COMPLETE.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('PASS: exported 5 fold fits and deploy fit to ' + str(model_dir))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        p.exit(2, f'BLOCKED: {exc}\n')


if __name__ == '__main__':
    main()
