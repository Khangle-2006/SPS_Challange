"""Aggregate every receipt/failure; no dropping failed condition-recordings."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from simulate import PROTOCOL,write_json,sha256


def metrics(rows,key='old_hr'):
    good=[r for r in rows if r.get(key) is not None and r.get('reference_hr') is not None]
    if not good:return {'windows':0,'subject_macro_mae_bpm':None,'p95_absolute_error_bpm':None,'error_gt20_fraction':None}
    err=np.asarray([abs(r[key]-r['reference_hr']) for r in good]);subjects=collections.defaultdict(list)
    for r,e in zip(good,err):subjects[r['subject_id']].append(float(e))
    return {'windows':len(good),'subjects':len(subjects),'subject_macro_mae_bpm':float(np.mean([np.mean(v) for v in subjects.values()])),
      'window_mae_bpm':float(err.mean()),'p95_absolute_error_bpm':float(np.percentile(err,95)),
      'error_gt20_count':int((err>20).sum()),'error_gt20_fraction':float((err>20).mean())}


def report(root):
    root=Path(root);cohort=json.loads((root/'cohort.json').read_text());results={};allrows=[]
    expected=len(cohort['recordings'])
    for cond in PROTOCOL['conditions']:
        rows=[];failures=[];manifests=[];candidates=0;receipts=0;reasons=collections.Counter();v2rows=[]
        for name in cohort['recordings']:
            p=root/cond/name
            if (p/'manifest.json').exists():manifests.append(json.loads((p/'manifest.json').read_text()))
            if not (p/'evaluation.json').exists():
                failures.append({'recording':name,'error':json.loads((p/'failure.json').read_text())['error'] if (p/'failure.json').exists() else 'pending'})
                continue
            d=json.loads((p/'evaluation.json').read_text());receipts+=1;candidates+=d['candidate_windows']
            for r in d['rows']:
                row={**r,'subject_id':d['subject_id'],'recording_id':name,'condition':cond};rows.append(row);allrows.append(row)
                if r['failure']:reasons[r['failure']]+=1
                if r.get('v2',{}).get('hr_bpm') is not None and not r['v2'].get('abstain',True):
                    v2rows.append({**row,'v2_hr':r['v2']['hr_bpm']})
        bitrate=[m for m in manifests if m['bitrate_limit_kbps'] is not None]
        passed=sum(m['bitrate_pass'] for m in bitrate)
        results[cond]={'expected_recordings':expected,'evaluated_recordings':receipts,'missing_or_failed':failures,
          'candidate_windows_in_evaluated_recordings':candidates,'successful_windows':sum(r['failure'] is None for r in rows),
          'coverage_evaluated_recordings_only':sum(r['failure'] is None for r in rows)/candidates if candidates else None,
          'failure_reasons':dict(reasons),'old_decoder':metrics(rows),
          'v2_diagnostic_selective':metrics(v2rows,'v2_hr'),
          'bitrate_files':len(bitrate),'bitrate_pass_files':passed,
          'bitrate_pass_fraction_expected_cohort':passed/expected if bitrate else None,
          'bitrate_acceptance':passed/expected>=.95 if bitrate else None,
          'bitrate_range_kbps':[min(m['actual']['whole_file_kbps'] for m in bitrate),max(m['actual']['whole_file_kbps'] for m in bitrate)] if bitrate else None,
          'qp_available_files':sum(m['qp']['mean_qp'] is not None for m in manifests)}
    result={'dataset':cohort['dataset'],'protocol_sha256':sha256(root/'protocol.json'),'conditions':results,
      'bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),'limitation':'Development cohort, not untouched test. MCD C5 is double compression. C1 includes fixed storage codec. Metrics separate datasets; missing evaluations remain explicit.'}
    write_json(root/'summary.json',result);write_json(root/'all_windows.json',allrows)
    lines=['# T5 constraint simulation — '+cohort['dataset'],'','Protocol SHA256: `'+result['protocol_sha256']+'`','',
      '| Condition | Recordings | Valid/candidate windows | Subject macro MAE | P95 absolute error | >20 bpm | C5 bitrate passes |',
      '|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'NA' if x is None else f'{x:.3f}'
    for c,d in results.items():
        m=d['old_decoder'];lines.append(f"| {c} | {d['evaluated_recordings']}/{expected} | {d['successful_windows']}/{d['candidate_windows_in_evaluated_recordings']} | {fmt(m['subject_macro_mae_bpm'])} | {fmt(m['p95_absolute_error_bpm'])} | {fmt(m['error_gt20_fraction'])} | {d['bitrate_pass_files']}/{expected if d['bitrate_files'] else 'NA'} |")
    lines+=['',result['limitation'],'','The coverage denominator above covers evaluated recordings only; pending/failed recordings are explicitly listed in summary.json and cannot satisfy completion. V2 is diagnostic and remains disabled as the default. P95 is pooled window absolute error; MAE is averaged within subject, then across subjects.']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');report(ap.parse_args().root)
