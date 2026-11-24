"""Read labels only AFTER complete predictions are sealed. No inference, refit or threshold selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(rows,labels):
    good=[r for r in rows if not r['abstain'] and r['pr_bpm'] is not None]
    base=dict(videos=len(rows),predicted=len(good),coverage=len(good)/max(len(rows),1))
    if not good:return dict(**base,mae=None)
    ps=sorted({labels[r['id']]['person'] for r in good});errors=[];mse=[];bias=[];yp=[];pp=[]
    for person in ps:
        rr=[r for r in good if labels[r['id']]['person']==person]
        y=np.array([labels[r['id']]['pr'] for r in rr],float);p=np.array([r['pr_bpm'] for r in rr],float);d=p-y
        errors.append(np.abs(d).mean());mse.append((d*d).mean());bias.append(d.mean());yp.append(y.mean());pp.append(p.mean())
    errors=np.array(errors);bias=np.array(bias);rng=np.random.default_rng(0);ix=rng.integers(0,len(ps),(10000,len(ps)))
    lo,hi=np.quantile(errors[ix].mean(1),[.025,.975]);r=float(np.corrcoef(yp,pp)[0,1]) if len(ps)>1 and np.std(yp)>0 and np.std(pp)>0 else None
    sd=float(np.std(bias,ddof=1)) if len(ps)>1 else None
    return dict(**base,people=len(ps),mae=float(errors.mean()),rmse=float(np.mean(np.sqrt(mse))),rmse_root_person_mse=float(np.sqrt(np.mean(mse))),pearson_person_means=r,
                mae_ci95=[float(lo),float(hi)],bias=float(bias.mean()),ba_person_mean_loa=None if sd is None else [float(bias.mean()-1.96*sd),float(bias.mean()+1.96*sd)],
                ba_definition='Bland-Altman on person-mean predictions/reference, not window-level repeated-measure LoA',bootstrap=10000,seed=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--authorization-file',type=Path);a=p.parse_args()
    root=a.predictions;seal=json.loads((root/'SEALED.json').read_text())
    if seal['scope']!='dev_dryrun' and (not a.authorization_file or a.authorization_file.read_text(encoding='utf-8').strip()!='M\u1ede HOLDOUT S1'):p.error('Holdout scoring not authorized')
    if sha(root/'PREDICTIONS.json')!=seal['predictions_sha256'] or sha(root/'PRE_RUN.json')!=seal['pre_run_sha256']:p.error('Prediction seal changed')
    rows=json.loads((root/'PREDICTIONS.json').read_text())
    if len(rows)!=seal['videos']:p.error('Incomplete prediction set')
    for row in rows:
        path=(root/row['result']).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=row['sha256']:p.error('Per-video prediction changed')
    # Exclusive score lock prevents re-reading the outcomes under modified settings.
    with (root/'SCORE_STARTED.json').open('x') as f:json.dump(dict(seal_sha256=sha(root/'SEALED.json')),f)
    labels=json.loads(a.labels.read_text())
    if set(labels)!={r['id'] for r in rows}:raise ValueError('Labels must exactly match sealed cohort, with no additional people')
    if any(set(v)!={'person','pr'} for v in labels.values()):raise ValueError('Expected person/pr fields only')
    result=dict(scope=seal['scope'],all=metrics(rows,labels),by_file_type={t:metrics([r for r in rows if r['file_type']==t],labels) for t in sorted({r['file_type'] for r in rows})},labels_sha256=sha(a.labels),seal_sha256=sha(root/'SEALED.json'))
    with (root/'SCORE.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
