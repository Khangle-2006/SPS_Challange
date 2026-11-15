"""Frozen V3 scorer and reference bytes reused by strict hashes, no CSV duplication."""
import csv,json
from pathlib import Path
import numpy as np
from t8_plan import read,write,sha,window_key
from t8_evaluate import csv_write,decode_difference
from polarity_loader import load_canonical_clip
from score_round1 import score
def evaluate(model,index,fold,seed,out,v3,device):
    import torch
    out=Path(out);out.mkdir(exist_ok=False);model.eval();summary={}
    for ds,prefix in [('UBFC2','ubfc'),('MCD','mcd')]:
        records=[r for r in index['records'] if r['dataset']==ds and r['subject_id'] in index['folds']['folds'][fold][prefix+'_val_subjects']]
        for variant in ('native','f10_b500'):
            condition=ds+'_'+variant;dst=out/condition;dst.mkdir()
            base=Path(v3)/f'fold{fold}'/f'seed{seed}'/'joint'/'evaluation'/condition
            ref=base/'reference_signal.csv'; m=read(base/'metrics.json');pol=read(base/'REFERENCE_POLARITY.json')
            assert sha(ref)==m['reference_sha256']==pol['canonical_reference_sha256']
            assert sha(base/'raw_reference_signal.csv')==pol['raw_reference_sha256']
            assert pol['all_hr_and_failure_parity']
            assert m['scorer_sha256']==sha(Path(__import__('score_round1').__file__))
            refs=list(csv.DictReader(ref.open()))
            jobs=[(r,w) for r in records for w in r['candidates']]
            assert len(refs)==len(jobs)
            write(dst/'REFERENCE_REUSE.json',dict(path=str(ref),sha256=sha(ref),raw_path=str(base/'raw_reference_signal.csv'),raw_sha256=pol['raw_reference_sha256'],polarity_receipt_sha256=sha(base/'REFERENCE_POLARITY.json'),strict_unchanged_reference=True))
            outputs=[];failures=[];waves=[];keys=[]
            for row,(r,w) in zip(refs,jobs):
                assert (row['subject_id'],row['recording_id'],float(row['window_start_s']))==(ds+':'+r['subject_id'],r['recording_id'],float(w['start_s']))
                pred={k:row[k] for k in ('subject_id','recording_id','window_start_s')};pred.update(hr_bpm='',abstain=True)
                try:
                    if window_key(w) not in r['source_valid_starts']:raise ValueError('canonical_source_rejected')
                    if not r['evaluation_usable'][variant][str(window_key(w))]:raise ValueError('variant_image_or_ppg_support')
                    x,_,_,_=load_canonical_clip(Path(index['roots'][ds])/r['recording_id']/variant/'crop',float(w['start_s']),ds)
                    with torch.no_grad(): wave=model(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
                    pred.update(hr_bpm=decode_difference(wave),abstain=False);waves.append(wave);keys.append([r['recording_id'],str(w['start_s'])])
                except ValueError as exc:failures.append({**pred,'reason':str(exc)})
                outputs.append(pred)
            csv_write(dst/'processed_output.csv',outputs)
            np.savez_compressed(dst/'waveforms.npz',difference=np.asarray(waves),keys=np.asarray(keys))
            write(dst/'abstentions.json',failures);metrics=score(ref,dst/'processed_output.csv')
            assert metrics['reference_sha256']==m['reference_sha256']
            write(dst/'metrics.json',metrics);summary[condition]=metrics
    write(out/'summary.json',summary)
