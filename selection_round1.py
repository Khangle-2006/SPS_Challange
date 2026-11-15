"""Frozen internal selection gates. This is not the SPS score or a D3 evaluator."""
import argparse,json
from pathlib import Path
import numpy as np
NAMES=('D1_clean','D1_C5','D2_native')


def compare(baseline,candidate):
    if set(baseline)!=set(NAMES) or set(candidate)!=set(NAMES):raise ValueError('Need exactly D1_clean,D1_C5,D2_native')
    coverage={};warnings=[]
    for name in NAMES:
        b,c=baseline[name],candidate[name]
        if b['schema_version']!=c['schema_version']:raise ValueError('Scoring protocol mismatch')
        if b['reference_sha256']!=c['reference_sha256']:raise ValueError('Different reference manifests: '+name)
        coverage[name]={'baseline':b['coverage'],'candidate':c['coverage']}
        if b['coverage']!=c['coverage']:warnings.append(name+': selective coverage differs; accuracy is not all-window performance')
        if any(x['subject_macro']['mae'] is None for x in [b,c]):
            return {'evaluable':False,'reason':'undefined macro MAE: '+name,'coverage':coverage}
    sb=float(np.mean([baseline[n]['subject_macro']['mae'] for n in NAMES]))
    sc=float(np.mean([candidate[n]['subject_macro']['mae'] for n in NAMES]))
    bclean=baseline['D1_clean']['subject_macro']['mae'];cclean=candidate['D1_clean']['subject_macro']['mae']
    bp=baseline['D1_C5']['window_level']['p95_abs_error'];cp=candidate['D1_C5']['window_level']['p95_abs_error']
    if bp is None or cp is None:return {'evaluable':False,'reason':'undefined C5 P95','coverage':coverage}
    gates={'S_reduction_ge10pct':sb>0 and sc<=.9*sb,
           'clean_increase_le5pct':cclean<=1.05*bclean,'C5_p95_not_increased':cp<=bp}
    return {'evaluable':True,'S_baseline':sb,'S_candidate':sc,'relative_reduction':1-sc/sb if sb>0 else None,
            'gates':gates,'passes_three_numerical_gates':all(gates.values()),'coverage':coverage,'warnings':warnings,
            'scope':'Numerical gates only; requires separately verified OOF subjects, cohort completeness, fixed seeds/config and constraint manifests. No automatic deployment or D3 access.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    def load(path):
        path=Path(path);mapping=json.loads(path.read_text());return {k:json.loads((path.parent/v).read_text()) for k,v in mapping.items()}
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(compare(load(a.baseline),load(a.candidate)),indent=2,allow_nan=False))
