"""Versioned global MCD training polarity correction; joint-only 15-run queue."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import json
import random
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from t8_plan import read, write, sha, code_fingerprint, pools, steps_per_epoch, draw, SEEDS
from t8_t1_loader import load_clip
import polarity_guard as guard

CODE = Path(__file__).resolve().parent
CORRECTION = read(CODE/'CORRECTION.json')
CORRECTION_SHA = sha(CODE/'CORRECTION.json')
BASELINE_CODE = read(CODE/'BASELINE_SHA256.json')
ORIGINAL_RUNS = None
CORRECTED_ROOT = None
PROJECT_RECEIPT = None


def resource_guard():
    """Physical GPU0 only; do not touch anyone else's job or choose GPU1."""
    samples = []
    for i in range(3):
        line = subprocess.check_output(['nvidia-smi', '--id=0',
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        free, util = [int(x.strip()) for x in line.split(',')]
        samples.append({'free_MiB': free, 'utilization_percent': util})
        if free < 6144 or util > 10:
            raise RuntimeError('RESOURCE_WAIT: GPU0 has insufficient free memory or active contention: '+line)
        if i < 2:
            time.sleep(2)
    return samples


def storage_guard(index, output):
    result = guard.storage_guard(index, ORIGINAL_RUNS.parent, CORRECTED_ROOT)
    result['project_budget'] = guard.project_storage_guard(ORIGINAL_RUNS.parent.parent, PROJECT_RECEIPT)
    return result


def seed_all(seed, fold):
    import torch
    actual = seed + 10000*fold
    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)
    torch.cuda.manual_seed_all(actual)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_hash(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(value, path):
    import torch
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def batch_arrays(index, selections):
    """Only training target signs change; cached input and loader remain exact."""
    xs, ys = [], []
    for dataset, record, start, variant in selections:
        crop = Path(index['roots'][dataset])/record/variant/'crop'
        x, y, _ = load_clip(crop, start)
        xs.append(x)
        ys.append(y * CORRECTION['training_target_sign'][dataset])
    return np.stack(xs), np.stack(ys)


def smoke(index, output, index_sha, frozen):
    import torch
    from EfficientPhys_fixed import EfficientPhys
    result_path = output/'GPU_SMOKE.json'
    if result_path.exists():
        result = read(result_path)
        assert result['index_sha256'] == index_sha and result['code_sha256'] == frozen
        assert result['passed']
        return
    pool = pools(index, 0)
    picks = [(ds, *pool[ds][0], 'native') for ds in ('UBFC2', 'MCD')]
    x, y = batch_arrays(index, picks)
    seed_all(2027, 0)
    model = EfficientPhys(frame_depth=10, img_size=36).cuda()
    xx = torch.from_numpy(x).cuda()
    yy = torch.from_numpy(y).cuda()
    model.eval()
    with torch.no_grad():
        together = model(xx)
        separately = torch.cat([model(xx[i:i+1]) for i in range(2)])
    assert together.shape == (2, 300) and torch.isfinite(together).all()
    torch.testing.assert_close(together, separately, rtol=2e-4, atol=2e-5)
    maximum = float((together-separately).abs().max())
    model.train()
    loss = (model(xx)-yy).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    write(result_path, dict(passed=True, index_sha256=index_sha, code_sha256=frozen,
        batch_vs_separate_max_abs=maximum, finite_backward=True,
        actual_cache_picks=picks, torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))
    del model, xx, yy, together, separately, loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(2)


def run_one(index, output, fold, seed, arm, index_sha, frozen):
    assert arm == 'joint'
    import torch
    from EfficientPhys_fixed import EfficientPhys
    from t8_evaluate import evaluate
    dest = output/f'fold{fold}'/f'seed{seed}'/arm
    if dest.exists():
        state = read(dest/'status.json') if (dest/'status.json').exists() else {}
        provenance = read(dest/'provenance.json') if (dest/'provenance.json').exists() else {}
        assert provenance.get('index_sha256') == index_sha and provenance.get('code_sha256') == frozen
        if state.get('state') == 'complete':
            assert sha(dest/'final.pt') == state['checkpoint_sha256']
            return
        raise RuntimeError('Partial run preserved; inspect before versioned continuation: '+str(dest))
    resource = resource_guard()
    storage = storage_guard(index, output)
    pool = pools(index, fold)
    steps = steps_per_epoch(pool)
    seed_all(seed, fold)
    model = EfficientPhys(frame_depth=10, img_size=36).cuda()
    initial = model_hash(model)
    baseline_history = guard.baseline_pair(ORIGINAL_RUNS, fold, seed, pool, initial, BASELINE_CODE)
    dest.mkdir(parents=True)
    config = dict(fold=fold, seed=seed, arm=arm, epochs=30, batch_size=2,
        steps_per_epoch=steps, max_lr=.009, weight_decay=0., selection='final_epoch_30',
        pool_sizes={k: len(v) for k, v in pool.items()}, test_opened=False, D3_opened=False,
        frontend='unified_oval_v2',frontend_sha256=read(Path(__file__).parent/'SOURCE_CACHE_SHA256.json')['oval_crop.py'],
        landmark_sha256=read(Path(__file__).parent/'SOURCE_CACHE_SHA256.json')['oval_frontend.py'],
        correction_manifest=CORRECTION, correction_manifest_sha256=CORRECTION_SHA,
        baseline_source_sha256=BASELINE_CODE, variant='t8-polarity-v2-20261102')
    write(dest/'provenance.json', dict(config=config, index_sha256=index_sha,
        code_sha256=frozen, initial_model_sha256=initial, resource=resource, storage=storage,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        correction_manifest_sha256=CORRECTION_SHA, baseline_runs=str(ORIGINAL_RUNS),
        project_budget_receipt_sha256=sha(PROJECT_RECEIPT),
        train_pool_sha256=hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()))
    opt = torch.optim.AdamW(model.parameters(), lr=.009, weight_decay=0.)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=.009, epochs=30, steps_per_epoch=steps)
    history = []
    for epoch in range(30):
        model.train()
        storage_guard(index, output)
        start_time, losses = time.time(), []
        dataset_losses = {'UBFC2': [], 'MCD': []}
        control_order = hashlib.sha256()
        order = hashlib.sha256()
        codec_counts = {}
        for step in range(steps):
            selection = draw(pool, seed, fold, epoch, step, arm)
            order.update(json.dumps(selection).encode())
            control_selection = draw(pool, seed, fold, epoch, step, 'ubfc_only')
            control_order.update(json.dumps(control_selection).encode())
            assert selection[0] == control_selection[0]
            assert [v[3] for v in selection] == [v[3] for v in control_selection]
            for ds, _, _, variant in selection:
                key = ds+':'+variant
                codec_counts[key] = codec_counts.get(key, 0)+1
            x, y = batch_arrays(index, selection)
            opt.zero_grad(set_to_none=True)
            pred = model(torch.from_numpy(x).cuda())
            loss = (pred-torch.from_numpy(y).cuda()).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError('Non-finite gradient')
            opt.step()
            schedule.step()
            losses.append(float(loss.detach()))
            guard.record_dataset_mse(pred, torch.from_numpy(y).to(pred.device), selection, dataset_losses)
        assert order.hexdigest() == baseline_history['joint'][epoch]['draw_sha256'], 'Joint draw history differs'
        assert control_order.hexdigest() == baseline_history['ubfc_only'][epoch]['draw_sha256'], 'Control draw history differs'
        history.append(dict(epoch=epoch+1, train_mse=float(np.mean(losses)),
            dataset_train_mse={ds: float(np.mean(values)) for ds, values in dataset_losses.items()},
            dataset_train_clips={ds: len(values) for ds, values in dataset_losses.items()},
            baseline_control_draw_sha256=control_order.hexdigest(),
            elapsed_s=time.time()-start_time, draw_sha256=order.hexdigest(), codec_counts=codec_counts))
        save_checkpoint(dict(epoch=epoch+1, model=model.state_dict(), optimizer=opt.state_dict(),
            scheduler=schedule.state_dict(), history=history, config=config,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), dest/'last.pt')
        write(dest/'history.json', history)
        write(dest/'status.json', dict(state='training', completed_epochs=epoch+1))
        print(json.dumps(dict(fold=fold, seed=seed, arm=arm, **history[-1])), flush=True)
    expected = model_hash(model)
    save_checkpoint(dict(model=model.state_dict(), config=config, epoch=30), dest/'final.pt')
    model.load_state_dict(torch.load(dest/'final.pt', map_location='cuda', weights_only=False)['model'])
    assert model_hash(model) == expected
    write(dest/'status.json', dict(state='training_complete_evaluation_pending', completed_epochs=30,
        checkpoint_sha256=sha(dest/'final.pt'), model_sha256=expected))
    evaluate(model, index, fold, dest/'evaluation', 'cuda')
    write(dest/'status.json', dict(state='complete', completed_epochs=30,
        checkpoint_sha256=sha(dest/'final.pt'), model_sha256=expected, no_outer_val_selection=True))
    del model, opt, schedule, pred, loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(2)


def main(a):
    import fcntl
    global ORIGINAL_RUNS, CORRECTED_ROOT, PROJECT_RECEIPT
    ORIGINAL_RUNS = a.original_runs.resolve()
    CORRECTED_ROOT = a.output.resolve()
    PROJECT_RECEIPT = a.project_budget_receipt.resolve()
    assert ORIGINAL_RUNS.name == 'runs'
    # Reject invalid/nested/outside-project paths before mkdir or queue.lock creation.
    guard.validate_output_path(ORIGINAL_RUNS.parent, CORRECTED_ROOT)
    # The exclusive original lock remains held throughout the entire v2 queue.
    # This rejects an active original runner before any GPU calls or mutations.
    baseline_lock, receipts = guard.acquire_completed_baseline(ORIGINAL_RUNS)
    try:
        frozen = read(CODE/'FROZEN_SHA256.json')
        assert code_fingerprint(CODE) == frozen
        index = read(a.index)
        index_sha = sha(a.index)
        assert index_sha == guard.EXPECTED_INDEX and index['hashes_verified']
        assert index['folds'] == read(CODE/'folds_frozen.json')
        cache_fingerprint = read(CODE/'SOURCE_CACHE_SHA256.json')
        assert all(fp == cache_fingerprint for fp in index['cache_fingerprints'].values())
        for name, digest in BASELINE_CODE.items():
            if name != 't8_run.py':
                assert frozen[name] == digest, ('Unapproved dependency change', name)
        assert CORRECTION['training_target_sign'] == {'UBFC2': 1, 'MCD': -1}
        assert CORRECTION['index_sha256'] == index_sha
        # Validate cross-server project inventory freshness before touching GPU.
        guard.project_storage_guard(ORIGINAL_RUNS.parent.parent, PROJECT_RECEIPT, require_fresh=True)
        a.output.mkdir(parents=True, exist_ok=True)
        lock = (a.output/'queue.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            storage = storage_guard(index, a.output)
            binding = dict(index_sha256=index_sha, correction_manifest=CORRECTION,
                correction_manifest_sha256=CORRECTION_SHA, code_sha256=frozen,
                original_completed_runs=receipts, storage=storage,
                project_budget_receipt=read(PROJECT_RECEIPT),
                project_budget_receipt_sha256=sha(PROJECT_RECEIPT))
            preflight = a.output/'PREFLIGHT.json'
            if preflight.exists():
                old = read(preflight)
                assert old['index_sha256'] == index_sha and old['code_sha256'] == frozen
                assert old['correction_manifest_sha256'] == CORRECTION_SHA
            else:
                write(preflight, binding)
            if a.preflight_only:
                return
            resource_guard()
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.cuda.set_per_process_memory_fraction(.35, 0)
            # No unchanged smoke rerun: exact input/model/decoder, target sign
            # application and finite backward tested separately on CPU.
            for fold in range(5):
                for seed in SEEDS:
                    write(a.output/'queue_status.json', dict(state='running', pid=os.getpid(),
                        fold=fold, seed=seed, arm='joint', index_sha256=index_sha,
                        correction_manifest_sha256=CORRECTION_SHA))
                    run_one(index, a.output, fold, seed, 'joint', index_sha, frozen)
            write(a.output/'queue_status.json', dict(state='complete', runs=15,
                index_sha256=index_sha, correction_manifest_sha256=CORRECTION_SHA))
        finally:
            lock.close()
    finally:
        baseline_lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--original-runs', type=Path, required=True)
    p.add_argument('--project-budget-receipt', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    # Refusal must not alter original artifacts or mark a new queue as running.
    main(args)
