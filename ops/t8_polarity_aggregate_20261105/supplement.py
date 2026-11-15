"""Independent macro cross-check, before/after strata, loss and encoding receipts."""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

code, root, index_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(code))
from score_round1 import stats
from t8_plan import read, write, sha

summary = read(root/'SUMMARY.json')
folds = read(root/'FOLD_METRICS.json')
strata, persons, checks = {}, [], []
for p in sorted(root.glob('seed*/metrics.json')):
    name = p.parent.name
    seed, arm, condition = name.split('_', 1)[0], None, None
    for candidate in ('ubfc_only', 'joint'):
        if name.startswith(seed+'_'+candidate+'_'):
            arm, condition = candidate, name[len(seed)+len(candidate)+2:]
    m = read(p)
    f = [x for x in folds if x['seed'] == int(seed[4:]) and x['arm'] == arm and x['condition'] == condition]
    expected = np.average([x['subject_macro']['mae'] for x in f], weights=[x['counts']['subjects'] for x in f])
    assert abs(expected-m['subject_macro']['mae']) < 1e-10
    for count in ('candidate_windows', 'eligible_windows', 'accepted_windows'):
        assert sum(x['counts'][count] for x in f) == m['counts'][count]
    checks.append(dict(run=name, macro_from_folds=float(expected), passed=True))
    for sid, v in m['per_subject'].items():
        persons.append(dict(seed=int(seed[4:]), arm=arm, condition=condition, subject=sid, **v))
    if condition.startswith('MCD'):
        for stage in ('before', 'after'):
            rr = [r for r in m['windows'] if r['recording_id'].endswith('_'+stage)]
            by_person = defaultdict(list)
            for r in rr:
                by_person[r['subject_id']].append(r)
            pp = {}
            for sid, sub in by_person.items():
                good = [r for r in sub if r['accepted']]
                pp[sid] = stats([r['prediction_hr_bpm'] for r in good], [r['reference_hr_bpm'] for r in good])
            assert all(x['mae'] is not None for x in pp.values())
            strata[name+'_'+stage] = dict(subjects=len(pp), candidates=len(rr), eligible=sum(r['reference_eligible'] for r in rr), accepted=sum(r['accepted'] for r in rr), macro_MAE=float(np.mean([x['mae'] for x in pp.values()])), per_person=pp)
loss = read(root/'DATASET_LOSS_CURVES.json')
loss_summary = {ds: {str(ep): float(np.mean([r['dataset_train_mse'][ds] for r in loss if r['epoch']==ep])) for ep in (1, 10, 20, 30)} for ds in ('UBFC2', 'MCD')}
index = read(index_path)
encodings = []
for record in index['records']:
    p = Path(index['roots'][record['dataset']])/record['recording_id']/'f10_b500'/'complete.json'
    r = read(p)
    e = r['encoding']
    assert e['fps_requested'] == 10 and e['bitrate_pass'] and e['actual']['whole_file_kbps'] <= 500
    encodings.append(dict(dataset=record['dataset'], recording=record['recording_id'], receipt_sha256=sha(p), target_fps=10, mux_fps=e['mux_fps'], effective_sensor_fps=e['effective_sensor_fps'], measured_whole_file_kbps=e['actual']['whole_file_kbps']))
encoding_summary = {ds: {k: [min(x[k] for x in encodings if x['dataset']==ds), max(x[k] for x in encodings if x['dataset']==ds)] for k in ('effective_sensor_fps', 'measured_whole_file_kbps')} for ds in ('UBFC2', 'MCD')}
c = summary['conditions']
s_reduction = 1-summary['S_mean']['joint']/summary['S_mean']['ubfc_only']
clean_increase = c['UBFC2_native']['joint']['MAE']/c['UBFC2_native']['ubfc_only']['MAE']-1
c5_p95_delta = c['UBFC2_f10_b500']['joint']['subject_macro_P95']-c['UBFC2_f10_b500']['ubfc_only']['subject_macro_P95']
gates = dict(S_reduction=s_reduction, S_reduction_ge10pct=s_reduction>=.1, D1_clean_relative_increase=clean_increase, D1_clean_increase_le5pct=clean_increase<=.05, C5_subject_macro_P95_delta=c5_p95_delta, C5_P95_nonincrease=c5_p95_delta<=0, note='Descriptive joint-vs-control check; no outcome selection. 72-vs36 gate remains unevaluated because72 not run.')
write(root/'SUPPLEMENT.json', dict(independent_fold_macro_checks=checks, before_after=strata, dataset_loss_epoch_means=loss_summary, encoding_summary=encoding_summary, encoding_receipts=encodings, descriptive_gates=gates))
write(root/'PER_PERSON.json', persons)
print(json.dumps(dict(loss=loss_summary, encoding=encoding_summary, gates=gates)), flush=True)
