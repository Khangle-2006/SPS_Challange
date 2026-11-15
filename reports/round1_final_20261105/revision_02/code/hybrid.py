"""Fixed label-free window selection. This script never fits a threshold."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from spectral import snr
from metrics import person_metrics,aggregate,csv_write,write_json,table
p=argparse.ArgumentParser()
for n in ('root','code','controls','joint'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.code));from t8_evaluate import decode_difference
w=json.loads((a.root/'frozen_data/FROZEN_WINDOWS.json').read_text())
ps=json.loads((a.root/'frozen_data/POS_SNR.json').read_text());pos={(r['condition'],r['person'],r['recording'],r['start']):r for r in ps}
condition_dirs={'UBFC C5':'UBFC2_f10_b500','UBFC clean':'UBFC2_native','MCD native':'MCD_native','MCD-C5':'MCD_f10_b500'}
waves={};receipts=[];out=[];selection=[]
for r in w:
    if r['branch'] not in ('ubfc_only','v3_joint'):continue
    branch=r['branch'];ident=(r['condition'],r['subject_id'],r['recording_id'],r['window_start_s']);group=(branch,r['fold'],r['seed'],r['condition'])
    if group not in waves:
        rr=a.controls if branch=='ubfc_only' else a.joint;arm='ubfc_only' if branch=='ubfc_only' else 'joint'
        path=rr/f'fold{r["fold"]}'/f'seed{r["seed"]}'/arm/'evaluation'/condition_dirs[r['condition']]/'waveforms.npz'
        with np.load(path) as z:
            waves[group]={(str(k[0]),float(k[1])):v.copy() for k,v in zip(z['keys'],z['difference'])}
        receipts.append(dict(branch=branch,fold=r['fold'],seed=r['seed'],condition=r['condition'],waveforms_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pred=None;which='abstain';ms=None;pp=pos.get(ident);pv=pp.get('snr') if pp else None
    if r['accepted']:
        diff=waves[group][r['recording_id'],r['window_start_s']]
        assert decode_difference(diff)==r['prediction_hr_bpm'],'Waveform/prediction mismatch'
        ms=snr(np.cumsum(np.asarray(diff,float)))
        # Selection consumes SNR and candidateHR only; no referenceHR is consulted.
        if ms is not None and (pv is None or ms>=pv):pred=r['prediction_hr_bpm'];which='model'
        elif pv is not None:pred=pp['hr'];which='POS'
    name='hybrid_'+branch
    out.append(dict(r,branch=name,prediction_hr_bpm=pred,accepted=bool(r['reference_eligible'] and pred is not None),prediction_reason=None if pred is not None else 'no_valid_source_or_common_support'))
    selection.append(dict(person=r['subject_id'],condition=r['condition'],fold=r['fold'],seed=r['seed'],branch=name,recording=r['recording_id'],start=r['window_start_s'],model_snr=ms,pos_snr=pv,selected=which))
write_json(a.root/'HYBRID_WINDOWS.json',out);write_json(a.root/'HYBRID_WAVEFORM_HASHES.json',receipts);write_json(a.root/'HYBRID_SELECTION.json',selection)
rows=person_metrics(w+out);summary=aggregate(rows)
write_json(a.root/'ALL_BRANCH_BOOTSTRAP.json',summary)
csv_write(a.root/'hybrid_per_person.csv',[r for r in rows if r['branch'].startswith('hybrid')])
text='# Hybrid POS/model OOF results\n\n'+table(summary,['ubfc_only','v3_joint','hybrid_ubfc_only','hybrid_v3_joint'])+'\n\n'
for b,parent in [('hybrid_ubfc_only','ubfc_only'),('hybrid_v3_joint','v3_joint')]:
    text+=f"## {b} versus {parent}\n\n```json\n"+json.dumps(summary['paired'][b+' minus '+parent],indent=2)+'\n```\n\n'
text+='## Adoption gates versus v3_joint\n\n```json\n'+json.dumps(summary['gates'],indent=2)+'\n```\n\nRule fixed before results. SNR is bandpeak±6bpm power divided by remaining0.7–3Hz bandpower. Ties selectmodel. No labels enterselection. There is no existing numeric SNR threshold in V3; unchanged support and finite/nonflat gates remain, with abstention when neither source is valid. Coverage is retained for all candidate windows. P95 is meanpersonwindowP95 afterseed averaging.\n'
(a.root/'hybrid_report.md').write_text(text)
print(json.dumps(dict(recommended=summary['recommended_branch'],S={b:r['S'] for b,r in summary['branches'].items()},gates=summary['gates'])),flush=True)
