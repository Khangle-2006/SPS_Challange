"""Independently recompute saved metrics with Python's standard library."""
import argparse
import hashlib
import json
import math
import statistics as st
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
root = a.directory
lock_path = root / 'frozen_protocol.json'
lock = json.loads(lock_path.read_text())
digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
output = {}
for split in ('val', 'test'):
    report = json.loads((root / (split + '_results.json')).read_text())
    assert report['frozen_protocol_sha256'] == digest
    result = []
    identities = None
    for item in report['seeds']:
        rows = item['records']
        ids = [(r['subject'], r['start_s'], r['gt_hr'], r['pos_hr']) for r in rows]
        if identities is None:
            identities = ids
        assert ids == identities
        errors = [abs(r['pred_hr'] - r['gt_hr']) if r['pred_hr'] is not None else 135. for r in rows]
        assert all(math.isclose(e, r['abs_error'], abs_tol=1e-10) for e, r in zip(errors, rows))
        subjects = sorted({r['subject'] for r in rows})
        per_subject = {s: st.mean(e for e, r in zip(errors, rows) if r['subject'] == s) for s in subjects}
        values = dict(window_MAE=st.mean(errors), window_RMSE=math.sqrt(st.mean(e*e for e in errors)),
                      subject_macro_MAE=st.mean(per_subject.values()))
        for key, value in values.items():
            assert math.isclose(value, item[key], abs_tol=1e-10)
        pos_errors = [abs(r['pos_hr'] - r['gt_hr']) for r in rows]
        result.append(dict(seed=item['seed'], best_epoch=item['best_epoch'], **values,
                           invalid=item['invalid'], per_subject_MAE=per_subject,
                           better_than_POS=sum(e < pe for e, pe in zip(errors, pos_errors)),
                           worse_than_POS=sum(e > pe for e, pe in zip(errors, pos_errors)),
                           tied_with_POS=sum(e == pe for e, pe in zip(errors, pos_errors))))
    assert math.isclose(st.mean(pos_errors), report['pos']['window_MAE'], abs_tol=1e-10)
    assert math.isclose(math.sqrt(st.mean(e*e for e in pos_errors)), report['pos']['window_RMSE'], abs_tol=1e-10)
    for metric, aggregate in report['aggregate'].items():
        values = [r[metric] for r in result]
        assert math.isclose(st.mean(values), aggregate['mean'], abs_tol=1e-10)
        assert math.isclose(st.stdev(values), aggregate['sample_sd'], abs_tol=1e-10)
    output[split] = dict(seeds=result, pos=report['pos'], aggregate=report['aggregate'])
assert set(output['val']['seeds'][0]['per_subject_MAE']).isdisjoint(output['test']['seeds'][0]['per_subject_MAE'])
output['audit'] = 'PASS: metrics independently recomputed; paired windows match; validation/test subjects disjoint.'
(root / 'independent_metric_audit.json').write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
