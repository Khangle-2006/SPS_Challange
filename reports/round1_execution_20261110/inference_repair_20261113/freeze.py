from common_repair import *
import csv
import numpy as np
assert not (P/'FROZEN.json').exists()
btc=read(O/'dev_manifest.json');split=read(O/'btc_split.json');assert sha(O/'btc_split.json')=='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
base=list(csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()));base=[r for r in base if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'];ids={r['recording'] for r in base};assert len(ids)==218
jobs=read(E/'C5_JOBS.json');jobs=[j for j in jobs if j['dataset']!='BTC' or j['recording'] in ids]
idx=read(R/'t8_oval_20261031/training_index.json');refs=read(B/'public_comparison/REFERENCE_COMPACT.json');out=[]
meta={v['recording']:v for v in btc['videos']}
for j in jobs:
 j=dict(j);ds=j['dataset'];rid=j['recording']
 if ds=='BTC':
  v=meta[rid];assert split['people'][v['person']]['partition']=='dev';j['metadata']=v
  cs=read(O/'predictions'/v['drive_id']/'native/classical.json')['rows'];j['windows']=[dict(start=s,truth=v['pr_bpm'],eligible=True) for s in sorted({r['start'] for r in cs if r['scheme']=='10s'})]
  j['crop']=str(O/'dev'/v['drive_id']/'native');j['source_info']=str(O/'dev'/v['drive_id']/'SOURCE.json')
 else:
  ref=next(r for r in refs['records'] if r['dataset']==ds and r['recording_id']==rid and r['variant']=='native');j['windows']=[dict(start=w['window_start_s'],truth=w['reference_hr_bpm'],eligible=w['reference_eligible']) for w in ref['windows']]
  j['crop']=str(Path(idx['roots'][ds])/rid/'native/crop')
 j['key']=ds+'_'+rid;out.append(j)
assert len(out)==372 and len({r['person'] for r in out if r['dataset']=='BTC'})==158
save(P/'MANIFEST.json',out)
save(P/'FROZEN.json',dict(utc=now(),manifest_sha256=sha(P/'MANIFEST.json'),protocol_sha256=sha(P/'PROTOCOL.md'),split_sha256=sha(O/'btc_split.json'),reference_sha256=sha(B/'public_comparison/REFERENCE_COMPACT.json'),baseline_sha256=sha(B/'VIDEO_PREDICTIONS.csv'),padding_checkpoints=read(VIS/'FROZEN.json')['checkpoints'],budget=budget(True),code={p.name:sha(p) for p in P.glob('*.py')}))
print('FROZEN',len(out))
