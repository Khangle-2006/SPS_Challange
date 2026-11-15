"""Independent Round1 scorer. Inputs are self-contained CSVs, not notebook state.

reference_signal.csv: one expected window per row, with subject_id, recording_id,
window_start_s, video_support_valid, ppg_timestamps_json, ppg_values_json.
processed_output.csv: subject_id, recording_id, window_start_s, hr_bpm, abstain.
The reference window manifest must be exported independently of predictions.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

VERSION='round1-self-score-v1-20261030'
REF_FIELDS={'subject_id','recording_id','window_start_s','video_support_valid','ppg_timestamps_json','ppg_values_json'}
OUT_FIELDS={'subject_id','recording_id','window_start_s','hr_bpm','abstain'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(value):
    s=str(value).strip().lower()
    if s not in ('true','false','1','0'):raise ValueError('Boolean must be true/false/1/0')
    return s in ('true','1')


def key(row):
    subject=str(row['subject_id']).strip();record=str(row['recording_id']).strip()
    start=float(row['window_start_s'])
    if not subject or not record or not np.isfinite(start):raise ValueError('Invalid window identity')
    return subject,record,round(start,9)


def read_csv(path,required):
    with open(path,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not required<=set(reader.fieldnames or []):
            raise ValueError(f'{path}: missing fields {sorted(required-set(reader.fieldnames or []))}; use canonical reference export, do not infer windows from predictions')
        result={}
        for row in reader:
            if row.get('split','').upper()=='D3':raise ValueError('D3 locked: not authorized by this development scorer')
            k=key(row)
            if k in result:raise ValueError('Duplicate window key: '+str(k))
            result[k]=row
    return result


def reference_hr(row):
    t=np.asarray(json.loads(row['ppg_timestamps_json']),float)
    p=np.asarray(json.loads(row['ppg_values_json']),float)
    start=float(row['window_start_s']);end=start+10
    if t.ndim!=1 or p.shape!=t.shape or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<0):
        raise ValueError('invalid_reference_time')
    if not np.isfinite(p).all():raise ValueError('invalid_reference_sample')
    if t[0]>start+1e-9 or t[-1]<end-1e-9:raise ValueError('incomplete_reference_coverage')
    bad=(np.diff(t)>.2+1e-12)&(t[:-1]<end-1e-10)&(t[1:]>start+1e-10)
    if bad.any():raise ValueError('reference_gap')
    u,inv,n=np.unique(t,return_inverse=True,return_counts=True)
    summed=np.zeros(len(u));np.add.at(summed,inv,p);p=summed/n;t=u
    w=np.interp(start+np.arange(300)/30,t,p)
    if w.std()<1e-10:raise ValueError('flat_reference')
    n=len(w);d=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n),format='csc')
    trend=spsolve(sparse.eye(n,format='csc')+100**2*(d.T@d),w)
    y=signal.sosfiltfilt(signal.butter(1,[.7,3.],btype='bandpass',fs=30,output='sos'),w-trend)
    f,power=signal.periodogram(y,fs=30,window='hann',nfft=4096,detrend=False)
    sel=(f>=.7)&(f<=3.);power=power[sel];f=f[sel]
    if not np.isfinite(power).all() or power.sum()<=1e-30:raise ValueError('invalid_reference_spectrum')
    return float(60*f[np.argmax(power)])


def stats(pred,ref):
    p=np.asarray(pred,float);r=np.asarray(ref,float);e=p-r;n=len(e)
    if not n:return dict(n=0,mae=None,rmse=None,pearson_r=None,p95_abs_error=None,error_gt20=0,bias=None)
    return dict(n=n,mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),
                pearson_r=float(np.corrcoef(p,r)[0,1]) if n>=2 and p.std()>0 and r.std()>0 else None,
                p95_abs_error=float(np.quantile(abs(e),.95,method='linear')),error_gt20=int(np.sum(abs(e)>20)),bias=float(e.mean()))


def repeated_loa(errors,subjects):
    e=np.asarray(errors,float);s=np.asarray(subjects,str);groups=np.unique(s);n=len(e);k=len(groups)
    result={'bias':float(e.mean()) if n else None,'loa_lower':None,'loa_upper':None,
            'method':'Bland-Altman2007 changing-quantity paired ANOVA, unequal repeats',
            'status':'insufficient_subjects_or_within_df','assumptions':'independent residuals within person; homogeneous within variance; approximately normal errors; not automatically verified'}
    if k<2 or n<=k:return result
    sizes=np.array([np.sum(s==g) for g in groups]);means=np.array([e[s==g].mean() for g in groups])
    msw=sum(np.sum((e[s==g]-m)**2) for g,m in zip(groups,means))/(n-k)
    msb=np.sum(sizes*(means-e.mean())**2)/(k-1);n0=(n-np.sum(sizes**2)/n)/(k-1)
    raw=(msb-msw)/n0;between=max(0.,raw);sd=np.sqrt(msw+between)
    result.update(loa_lower=float(e.mean()-1.96*sd),loa_upper=float(e.mean()+1.96*sd),
                  within_variance=float(msw),between_variance=float(between),between_variance_clipped=bool(raw<0),
                  status='estimated_assumptions_unverified')
    return result


def score(reference_path,output_path):
    reference=read_csv(reference_path,REF_FIELDS);output=read_csv(output_path,OUT_FIELDS)
    if not reference:raise ValueError('Empty reference window manifest')
    extra=set(output)-set(reference)
    if extra:raise ValueError('Output contains unexpected windows: '+str(sorted(extra)[:3]))
    rows=[];reasons=Counter();by_subject=defaultdict(list)
    for k,r in sorted(reference.items()):
        ref=None;reason=[]
        if not boolean(r['video_support_valid']):reason.append('invalid_video_support')
        try:ref=reference_hr(r)
        except ValueError as exc:reason.append(str(exc))
        eligible=not reason
        o=output.get(k);pred=None;abstain=True;prediction_reason=None
        if o is None:prediction_reason='missing_output'
        else:
            explicit=boolean(o['abstain'])
            try:pred=float(o['hr_bpm'])
            except (ValueError,TypeError):pred=None
            if pred is None or not np.isfinite(pred):pred=None;prediction_reason='invalid_prediction'
            elif explicit:prediction_reason='explicit_abstain'
            else:abstain=False
        accepted=eligible and not abstain
        row=dict(subject_id=k[0],recording_id=k[1],window_start_s=k[2],reference_hr_bpm=ref,
                 prediction_hr_bpm=pred,reference_eligible=eligible,accepted=accepted,
                 reference_reasons=reason,prediction_reason=prediction_reason)
        rows.append(row);by_subject[k[0]].append(row)
        reasons.update(reason)
        if prediction_reason:reasons[prediction_reason]+=1
    good=[r for r in rows if r['accepted']];eligible=[r for r in rows if r['reference_eligible']]
    subjects={}
    for sid,sub in by_subject.items():
        valid=[r for r in sub if r['reference_eligible']];accepted=[r for r in sub if r['accepted']]
        subjects[sid]={**stats([r['prediction_hr_bpm'] for r in accepted],[r['reference_hr_bpm'] for r in accepted]),
                       'candidate_windows':len(sub),'eligible_windows':len(valid),
                       'coverage':len(accepted)/len(valid) if valid else None}
    evaluable=[s for s in subjects.values() if s['eligible_windows']>0]
    missing_subjects=[sid for sid,s in subjects.items() if s['eligible_windows']>0 and s['n']==0]
    unscorable=[sid for sid,s in subjects.items() if not s['eligible_windows']]
    macro_ok=bool(evaluable) and not missing_subjects and not unscorable
    correlations=[s['pearson_r'] for s in subjects.values() if s['pearson_r'] is not None]
    pooled=stats([r['prediction_hr_bpm'] for r in good],[r['reference_hr_bpm'] for r in good])
    return {'schema_version':VERSION,'reference_sha256':sha(reference_path),'output_sha256':sha(output_path),
            'scorer_sha256':sha(__file__),'invalid_policy':'abstain_with_coverage',
            'counts':{'candidate_windows':len(rows),'eligible_windows':len(eligible),'accepted_windows':len(good),
                      'reference_invalid_windows':len(rows)-len(eligible),'subjects':len(subjects),'missing_output_windows':len(set(reference)-set(output))},
            'subject_macro':{'mae':float(np.mean([s['mae'] for s in evaluable])) if macro_ok else None,
                             'rmse':float(np.mean([s['rmse'] for s in evaluable])) if macro_ok else None,
                             'p95_abs_error':float(np.mean([s['p95_abs_error'] for s in evaluable])) if macro_ok else None,
                             'coverage':float(np.mean([s['coverage'] for s in evaluable])) if evaluable else None,
                             'pearson_median':float(np.median(correlations)) if correlations else None,
                             'pearson_undefined_subjects':len(subjects)-len(correlations)},
            'window_level':pooled,'bland_altman':repeated_loa([r['prediction_hr_bpm']-r['reference_hr_bpm'] for r in good],[r['subject_id'] for r in good]),
            'coverage':len(good)/len(eligible) if eligible else 0.,'coverage_all_candidates':len(good)/len(rows),
            'reference_coverage':len(eligible)/len(rows),'subjects_with_no_accepted_prediction':missing_subjects,
            'subjects_with_no_eligible_reference':unscorable,'reasons':dict(reasons),'per_subject':subjects,'windows':rows,
            'interpretation':'Undefined values are null, not zero. Accuracy is selective and must be reported with coverage. Counts refer to complete reference manifest, not proof of cohort completeness.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--metrics',help='Default: score_round1_metrics.json beside output CSV')
    a=p.parse_args();result=score(a.reference,a.output)
    path=Path(a.metrics) if a.metrics else Path(a.output).with_name('score_round1_metrics.json')
    if path.exists():raise FileExistsError('Preserve prior scores; use new --metrics path')
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['counts','subject_macro','window_level','coverage']},indent=2))


if __name__=='__main__':main()
