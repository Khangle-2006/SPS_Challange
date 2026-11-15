import argparse,json
from pathlib import Path
from metrics import person_metrics,csv_write,write_json,table
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
w=json.loads((r/'frozen_data/FROZEN_WINDOWS.json').read_text());s=json.loads((r/'FROZEN_BOOTSTRAP.json').read_text())
rows=person_metrics(w);csv_write(r/'mcd_c5_per_person.csv',[x for x in rows if x['condition']=='MCD-C5'])
native={(x['subject_id'],x['recording_id'],x['window_start_s'],x['branch'],x['seed']):x for x in w if x['condition']=='MCD native'}
loss=[];seen=set()
for x in w:
    if x['condition']!='MCD-C5':continue
    n=native[x['subject_id'],x['recording_id'],x['window_start_s'],x['branch'],x['seed']]
    assert n['reference_hr_bpm']==x['reference_hr_bpm'] and n['reference_eligible']==x['reference_eligible']
    if n['accepted'] and not x['accepted']:loss.append(dict(person=x['subject_id'],recording=x['recording_id'],start=x['window_start_s'],branch=x['branch'],seed=x['seed'],reason=x['prediction_reason']))
    seen.add((x['subject_id'],x['recording_id'],x['window_start_s']))
write_json(r/'MCD_C5_WINDOW_LOSSES.json',loss)
read=s['paired']['v3_joint minus ubfc_only']['MCD-C5'];passA=read['CI95'][1]<0
write_json(r/'MCD_C5_READING.json',dict(delta=read,UBFC_specific_pattern_supported=passA,alternative_not_demonstrated=not passA))
text='# MCD-C5 matched evaluation\n\nFixed reading declared before results: deltaCIupper<0 supports the UBFC-specific C5 pattern; otherwise V3 has not demonstrated strong-compression robustness.\n\n'+table(s)+'\n\n'+json.dumps(dict(reading=read,UBFC_specific_pattern_supported=passA,unique_MCD_windows=len(seen),new_losses=len(loss)),indent=2)+'\n\nMCD-C5 is a separate diagnostic condition and never entersS or branch selection. All15 existing checkpoints per branch are reused through their verified complete evaluation artifacts. Reference windows and HR are identical toMCDnative. Frame/support losses and classical failures remain explicit.\n\nMCD source MPEG-4 followed by H.264 is double compression; this premise must be confirmed by SOURCE_CODEC_VERIFICATION.json, not inferred from the filename. A favorable MCD-C5 result supports the specified dataset-specific pattern without proving the physical cause of UBFC degradation.\n'
(r/'mcd_c5_report.md').write_text(text);print(json.dumps(read),flush=True)
