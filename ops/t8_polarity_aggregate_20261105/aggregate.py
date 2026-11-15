"""Mixed-root OOF aggregation; never modifies frozen training bundles or raw outputs."""
import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def main():
    p = argparse.ArgumentParser()
    for name in ('code', 'controls', 'joint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.code))
    from t8_plan import read, write, sha, SEEDS
    from t8_evaluate import csv_write
    from polarity_evaluate import verify_reference_parity
    from score_round1 import score, key, stats
    from polarity_guard import EXPECTED_INDEX, require
    from ppg_polarity import POLICY_SHA256

    require(not a.output.exists(), 'Preserve existing aggregate')
    require(sha(a.code/'FROZEN_SHA256.json') == 'd994b335b6c482e80943c081d7a6aa1288308a6868e673e6be579f3b4d1b4205', 'Frozen manifest mismatch')
    frozen = read(a.code/'FROZEN_SHA256.json')
    for name, digest in frozen.items():
        require(sha(a.code/name) == digest, 'Frozen code mismatch '+name)
    require(read(a.joint/'queue_status.json')['state'] == 'complete', 'Joint queue incomplete')
    def rows(path):
        with path.open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    conditions = ('UBFC2_f10_b500', 'MCD_native', 'UBFC2_native', 'MCD_f10_b500')
    s_conditions = conditions[:3]
    run_receipts, parity_receipts, grouped, fold_metrics, loss_rows = [], [], {}, [], []
    # Validate each source and recompute reference sign/time/support/HR/failure parity.
    # Identical CSV hashes across seeds allow reuse of the exact parity calculation.
    parity_cache = {}
    for fold in range(5):
        for seed in SEEDS:
            runs = {arm: root/f'fold{fold}'/f'seed{seed}'/arm
                    for arm, root in [('ubfc_only', a.controls), ('joint', a.joint)]}
            provs, histories = {}, {}
            for arm, run in runs.items():
                state, prov, history = (read(run/name) for name in ('status.json', 'provenance.json', 'history.json'))
                require(state['state'] == 'complete' and state['completed_epochs'] == 30, 'Incomplete run')
                require(sha(run/'final.pt') == state['checkpoint_sha256'], 'Checkpoint mismatch')
                require(prov['index_sha256'] == EXPECTED_INDEX, 'Index mismatch')
                cfg = prov['config']
                require((cfg['fold'], cfg['seed'], cfg['arm'], cfg['epochs'], cfg['selection']) == (fold, seed, arm, 30, 'final_epoch_30'), 'Protocol mismatch')
                require(not cfg['test_opened'] and not cfg['D3_opened'], 'Closed cohort violation')
                require([h['epoch'] for h in history] == list(range(1, 31)), 'History mismatch')
                require(all(np.isfinite(h['train_mse']) for h in history), 'Nonfinite loss')
                if arm == 'joint':
                    require(prov['code_sha256'] == frozen, 'Joint provenance mismatch')
                    require(prov['correction_manifest_sha256'] == sha(a.code/'CORRECTION.json'), 'Correction mismatch')
                    loss_rows.extend(dict(fold=fold, seed=seed, **h) for h in history)
                provs[arm], histories[arm] = prov, history
                run_receipts.append(dict(fold=fold, seed=seed, arm=arm, checkpoint_sha256=state['checkpoint_sha256'], history_sha256=sha(run/'history.json'), provenance_sha256=sha(run/'provenance.json')))
            require(provs['joint']['initial_model_sha256'] == provs['ubfc_only']['initial_model_sha256'], 'Initial model differs')
            require(provs['joint']['train_pool_sha256'] == provs['ubfc_only']['train_pool_sha256'], 'Train pool differs')
            require(provs['joint']['config']['baseline_source_sha256'] == provs['ubfc_only']['code_sha256'], 'Baseline code differs')
            require(provs['joint']['config']['steps_per_epoch'] == provs['ubfc_only']['config']['steps_per_epoch'], 'Steps differ')
            for u, j in zip(histories['ubfc_only'], histories['joint']):
                require(j['baseline_control_draw_sha256'] == u['draw_sha256'], 'Matched control draws differ')
            for condition in conditions:
                ud, jd = (runs[arm]/'evaluation'/condition for arm in ('ubfc_only', 'joint'))
                receipt = read(jd/'REFERENCE_POLARITY.json')
                raw_hash, canonical_hash = sha(jd/'raw_reference_signal.csv'), sha(jd/'reference_signal.csv')
                require(sha(ud/'reference_signal.csv') == raw_hash == receipt['raw_reference_sha256'], 'Raw reference hash mismatch')
                require(canonical_hash == receipt['canonical_reference_sha256'], 'Canonical hash mismatch')
                require(receipt['policy_sha256'] == POLICY_SHA256 and receipt['all_hr_and_failure_parity'], 'Polarity receipt mismatch')
                signature = (raw_hash, canonical_hash)
                if signature not in parity_cache:
                    raw, canonical = rows(jd/'raw_reference_signal.csv'), rows(jd/'reference_signal.csv')
                    require(len(raw) == len(canonical), 'Reference length mismatch')
                    sign = -1 if condition.startswith('MCD') else 1
                    checked = [verify_reference_parity(r, c, sign) for r, c in zip(raw, canonical)]
                    # CSV encodes the numeric window start as text; receipt JSON retains float.
                    for item in checked:
                        item['window_start_s'] = float(item['window_start_s'])
                    parity_cache[signature] = checked
                require(parity_cache[signature] == receipt['window_parity'], 'Parity rows differ')
                parity_receipts.append(dict(fold=fold, seed=seed, condition=condition, raw_sha256=raw_hash, canonical_sha256=canonical_hash, windows=len(parity_cache[signature]), verified=True))
                metrics = {}
                for arm, directory in [('ubfc_only', ud), ('joint', jd)]:
                    metric = read(directory/'metrics.json')
                    require(metric['reference_sha256'] == sha(directory/'reference_signal.csv'), 'Metric reference mismatch')
                    require(metric['output_sha256'] == sha(directory/'processed_output.csv'), 'Prediction hash mismatch')
                    require(metric['scorer_sha256'] == sha(a.code/'score_round1.py'), 'Scorer mismatch')
                    metrics[arm] = metric
                    fold_metrics.append(dict(fold=fold, seed=seed, arm=arm, condition=condition, counts=metric['counts'], subject_macro=metric['subject_macro'], window_level=metric['window_level']))
                    grouped.setdefault((seed, arm, condition), []).append(directory)
                ur, jr = metrics['ubfc_only']['windows'], metrics['joint']['windows']
                fields = ('subject_id', 'recording_id', 'window_start_s', 'reference_hr_bpm', 'reference_eligible', 'reference_reasons')
                require([[r[k] for k in fields] for r in ur] == [[r[k] for k in fields] for r in jr], 'Scored reference parity differs')
            print('Validated fold', fold, 'seed', seed, flush=True)
    a.output.mkdir(parents=True)
    write(a.output/'VALIDATION.json', dict(run_receipts=run_receipts, reference_parity=parity_receipts, aggregate_code_sha256=sha(__file__), checked_utc=datetime.now(timezone.utc).isoformat()))
    all_metrics = {}
    for (seed, arm, condition), dirs in grouped.items():
        refs, predictions, subjects = [], [], set()
        for directory in dirs:
            rr = rows(directory/'reference_signal.csv')
            ss = {r['subject_id'] for r in rr}
            require(subjects.isdisjoint(ss), 'Outer subject leakage')
            subjects |= ss
            refs.extend(rr)
            predictions.extend(rows(directory/'processed_output.csv'))
        require(len(subjects) == (60 if condition.startswith('MCD') else 35), 'Incomplete cohort')
        require(len({key(r) for r in refs}) == len(refs), 'Duplicate windows')
        dest = a.output/f'seed{seed}_{arm}_{condition}'
        dest.mkdir()
        csv_write(dest/'reference_signal.csv', refs)
        csv_write(dest/'processed_output.csv', predictions)
        metric = score(dest/'reference_signal.csv', dest/'processed_output.csv')
        write(dest/'metrics.json', metric)
        all_metrics[seed, arm, condition] = metric
        print('Scored', seed, arm, condition, flush=True)
    summary = dict(scope='Full 5-fold 3-seed development OOF; 35 UBFC and MCD dev60; test/D3 closed', conditions={}, S_by_seed={}, S_definition='mean of subject-macro MAE for UBFC C5, MCD native, UBFC native', selection='All final epoch30 models; no outcome selection; original unharmonized joint excluded')
    for condition in conditions:
        result, deltas = {}, []
        for arm in ('ubfc_only', 'joint'):
            ms = [all_metrics[s, arm, condition] for s in SEEDS]
            result[arm] = dict(MAE=float(np.mean([m['subject_macro']['mae'] for m in ms])), subject_macro_P95=float(np.mean([m['subject_macro']['p95_abs_error'] for m in ms])), window_P95_by_seed=[m['window_level']['p95_abs_error'] for m in ms], gt20_by_seed=[m['window_level']['error_gt20'] for m in ms], counts_by_seed=[m['counts'] for m in ms], coverage_by_seed=[m['coverage'] for m in ms])
        for seed in SEEDS:
            u, j = (all_metrics[seed, arm, condition]['per_subject'] for arm in ('ubfc_only', 'joint'))
            require(set(u) == set(j), 'Subject pairing differs')
            require(all(u[s]['mae'] is not None and j[s]['mae'] is not None for s in u), 'Undefined subject MAE')
            deltas.append([j[s]['mae']-u[s]['mae'] for s in sorted(u)])
        d = np.asarray(deltas)
        rng = np.random.default_rng(20261031)
        boot = d[:, rng.integers(d.shape[1], size=(10000, d.shape[1]))].mean(axis=(0, 2))
        result['paired'] = dict(mean_joint_minus_control=float(d.mean()), CI95=np.quantile(boot, [.025, .975]).tolist(), subjects=sorted(u), delta_by_seed=d.tolist(), method='10000 paired person bootstraps, all seeds retained together')
        summary['conditions'][condition] = result
    for arm in ('ubfc_only', 'joint'):
        values = [float(np.mean([all_metrics[s, arm, c]['subject_macro']['mae'] for c in s_conditions])) for s in SEEDS]
        summary['S_by_seed'][arm] = values
    summary['S_mean'] = {arm: float(np.mean(v)) for arm, v in summary['S_by_seed'].items()}
    write(a.output/'SUMMARY.json', summary)
    write(a.output/'FOLD_METRICS.json', fold_metrics)
    write(a.output/'DATASET_LOSS_CURVES.json', loss_rows)
    print(json.dumps(dict(S_mean=summary['S_mean'], conditions={c: {k: v for k, v in r.items() if k != 'paired'} for c, r in summary['conditions'].items()})), flush=True)


if __name__ == '__main__':
    main()
