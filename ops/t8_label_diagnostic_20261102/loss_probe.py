"""CPU fixed training-subject probes; no updates, selection, or historical loss reconstruction."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FOLD, SEED, SUBJECTS = 0, 2027, 8
ARMS = ('ubfc_only', 'joint')
EXPECTED_INDEX = '12a87b829ab5e158b430c67701fa0dd978c9a3d592e0060da3e876069108d505'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code', type=Path, default=Path('/data_pool/khangle/facevitals/code/t8_oval_training_20261031'))
    p.add_argument('--index', type=Path, default=Path('/mnt/samsung_t5/khangle/facevitals/t8_oval_20261031/training_index.json'))
    p.add_argument('--runs', type=Path, default=Path('/mnt/samsung_t5/khangle/facevitals/t8_oval_20261031/runs'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve prior diagnostic output: ' + str(a.output))
    index_sha = sha(a.index)
    assert index_sha == EXPECTED_INDEX, 'Unexpected frozen index'
    frozen = read(a.code / 'FROZEN_SHA256.json')
    for name, digest in frozen.items():
        assert sha(a.code / name) == digest, ('Frozen code changed', name)
    sys.path.insert(0, str(a.code))
    import numpy as np
    import torch
    from t8_plan import pools
    from t8_t1_loader import load_clip
    from EfficientPhys_fixed import EfficientPhys
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    index = read(a.index)
    pool = pools(index, FOLD)
    probes = []
    for dataset in ('UBFC2', 'MCD'):
        record_subject = {r['recording_id']: r['subject_id'] for r in index['records'] if r['dataset'] == dataset}
        by_subject = {}
        for record, start in sorted(pool[dataset]):
            by_subject.setdefault(record_subject[record], (record, start))
        assert len(by_subject) >= SUBJECTS
        for subject in sorted(by_subject)[:SUBJECTS]:
            record, start = by_subject[subject]
            probes.append(dict(dataset=dataset, subject_id=subject, recording_id=record,
                               start_s=start, variant='native'))
    # Validate both immutable final checkpoints before creating output or inference.
    verified = {}
    for arm in ARMS:
        folder = a.runs / f'fold{FOLD}' / f'seed{SEED}' / arm
        status, provenance = read(folder / 'status.json'), read(folder / 'provenance.json')
        assert status['state'] == 'complete'
        checkpoint_sha = sha(folder / 'final.pt')
        assert checkpoint_sha == status['checkpoint_sha256']
        assert provenance['index_sha256'] == index_sha
        assert provenance['code_sha256'] == frozen
        verified[arm] = dict(folder=str(folder), checkpoint_sha256=checkpoint_sha,
                             status_sha256=sha(folder / 'status.json'), provenance=provenance)
    a.output.mkdir(parents=True)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), fold=FOLD, seed=SEED,
                    selection='First eligible native clip by recording/start of first 8 lexicographically sorted distinct training subjects per dataset',
                    probes=probes, index_sha256=index_sha, frozen_code_sha256=frozen,
                    script_sha256=sha(__file__), checkpoints=verified,
                    device='cpu', threads=1, torch_version=torch.__version__,
                    limitations=['Eval-mode final-checkpoint diagnostic, not historical train loss.',
                                 'Native-only fixed train-subject probe; not all augmented training draws.',
                                 'Sign-flipped target MSE is diagnostic only; no label edits or model updates.',
                                 'Aggregate historical train MSE cannot be decomposed by dataset.'])
    write(a.output / 'manifest.json', manifest)
    rows, history_rows = [], []
    for arm in ARMS:
        folder = Path(verified[arm]['folder'])
        checkpoint = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 30
        config = checkpoint['config']
        assert (config['fold'], config['seed'], config['arm']) == (FOLD, SEED, arm)
        assert config == verified[arm]['provenance']['config']
        model = EfficientPhys(frame_depth=10, img_size=36)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        verified[arm]['checkpoint_config'] = config
        verified[arm]['checkpoint_config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        history = read(folder / 'history.json')
        assert [h['epoch'] for h in history] == list(range(1, 31))
        verified[arm]['history_sha256'] = sha(folder / 'history.json')
        write(a.output / (arm + '_aggregate_history.json'), history)
        history_rows.extend(dict(arm=arm, epoch=h['epoch'], aggregate_train_mse=h['train_mse'], elapsed_s=h['elapsed_s']) for h in history)
        for probe in probes:
            crop = Path(index['roots'][probe['dataset']]) / probe['recording_id'] / 'native' / 'crop'
            x, y, _ = load_clip(crop, probe['start_s'])
            with torch.no_grad():
                pred = model(torch.from_numpy(x[None])).numpy()[0]
            assert pred.shape == y.shape == (300,) and np.isfinite(pred).all()
            corr = float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 1e-12 and y.std() > 1e-12 else None
            row = dict(arm=arm, **probe, eval_mse=float(np.mean((pred-y)**2)),
                       sign_flipped_target_mse=float(np.mean((pred+y)**2)),
                       target_pearson_r=corr, prediction_std=float(pred.std()), target_std=float(y.std()),
                       target_sha256=hashlib.sha256(y.tobytes()).hexdigest())
            rows.append(row)
            print(json.dumps(row), flush=True)
        assert sha(folder / 'final.pt') == verified[arm]['checkpoint_sha256']
        del model, checkpoint
    for i in range(len(probes)):
        assert rows[i]['target_sha256'] == rows[i+len(probes)]['target_sha256']
    summaries = []
    for arm in ARMS:
        for dataset in ('UBFC2', 'MCD'):
            selected = [r for r in rows if r['arm'] == arm and r['dataset'] == dataset]
            rs = [r['target_pearson_r'] for r in selected if r['target_pearson_r'] is not None]
            summaries.append(dict(arm=arm, dataset=dataset, clips=len(selected), subjects=len(selected),
                mean_eval_mse=float(np.mean([r['eval_mse'] for r in selected])),
                mean_sign_flipped_target_mse=float(np.mean([r['sign_flipped_target_mse'] for r in selected])),
                median_target_pearson_r=float(np.median(rs)) if rs else None))
    csv_write(a.output / 'per_clip.csv', rows)
    csv_write(a.output / 'aggregate_history.csv', history_rows)
    write(a.output / 'per_clip.json', rows)
    write(a.output / 'summary.json', dict(state='complete', summaries=summaries, limitations=manifest['limitations']))
    write(a.output / 'manifest.json', manifest)
    print(json.dumps(dict(state='complete', output=str(a.output), summaries=summaries)), flush=True)


if __name__ == '__main__':
    main()
