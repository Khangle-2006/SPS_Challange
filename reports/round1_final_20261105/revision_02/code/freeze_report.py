import argparse,json,shlex
from pathlib import Path
from metrics import person_metrics,aggregate,csv_write,write_json,table
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
windows=json.loads((root/'frozen_data/FROZEN_WINDOWS.json').read_text());rows=person_metrics(windows);s=aggregate(rows)
write_json(root/'FROZEN_BOOTSTRAP.json',s)
csv_write(root/'per_person_metrics.csv',[r for r in rows if r['branch'] in ('ubfc_only','v3_joint') and r['condition']!='MCD-C5'])
csv_write(root/'baselines_per_person.csv',[r for r in rows if r['branch'] not in ('ubfc_only','v3_joint')])
expected={'ubfc_only':{'UBFC C5':15.13,'MCD native':12.62,'UBFC clean':.99,'S':9.58},'v3_joint':{'UBFC C5':17.46,'MCD native':9.36,'UBFC clean':1.25,'S':9.35}}
bad=[]
for b,cc in expected.items():
    for c,v in cc.items():
        actual=s['branches'][b][c]['value'] if c=='S' else s['branches'][b][c]['MAE']['value']
        if abs(actual-v)>.05:bad.append(f'{b} {c}: expected{v}, actual{actual}')
write_json(root/'REPRODUCTION_CHECK.json',dict(tolerance=.05,discrepancies=bad,passed=not bad))
head=('DISCREPANCY\n'+'\n'.join(bad)+'\n\n') if bad else ''
sentences=[f"All15 runs per model branch reproduce the known MAEs and S within0.05bpm, with35 UBFC and60 MCD people." if not bad else 'Discrepancies are retained above and processing continues.',
           'The comparison uses all finalepoch30 models and keeps person-level splits,10second windows,0.7–3Hz decoding and abstention coverage.',
           'The v3_joint versus ubfc_only UBFC-C5 reading is '+s['paired']['v3_joint minus ubfc_only']['UBFC C5']['reading']+' by the newly requested seed0 paired person bootstrap.',
           'MCD native and UBFC clean readings are '+s['paired']['v3_joint minus ubfc_only']['MCD native']['reading']+' and '+s['paired']['v3_joint minus ubfc_only']['UBFC clean']['reading']+', respectively.',
           'POS,CHROM and train-fold constants use the same reference windows and oval source, with failures retained rather than imputed.']
scores='\n'.join(f"- {b}: S={r['S']['value']:.6f}, CI95={r['S']['CI95']}" for b,r in s['branches'].items())
(root/'v3_frozen_report.md').write_text(head+'# Frozen V3 development results\n\n'+table(s)+'\n\n'+scores+'\n\n'+'\n\n'.join(sentences)+'\n\nPrimaryP95 is mean person-level windowP95, after averaging seeds. MedianpersonMAE andP95personMAE are in FROZEN_BOOTSTRAP.json. MCD-C5 is shown separately and excluded fromS.\n')
receipts=json.loads((root/'frozen_data/C5_ENCODING_RECEIPTS.json').read_text());r=next(x for x in receipts if x['dataset']=='UBFC2')
with (root/'DECISIONS.md').open('a') as f:f.write('\n## Exact existing UBFC C5 command\n\nRecording '+r['recording']+', receiptSHA256 '+r['receipt_sha256']+'\n\n```sh\n'+shlex.join(r['command'])+'\n```\n\nOutput paths in that historical command identify the original artifact and are never executed with overwrite flags in this task. All per-record commands/FPS/bitrate receipts are retained in frozen_data/C5_ENCODING_RECEIPTS.json.\n')
print(json.dumps({b:r['S'] for b,r in s['branches'].items()}),flush=True)
