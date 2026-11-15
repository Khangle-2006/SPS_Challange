import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from vis_common import *
import csv,numpy as np
assert not (P/'SELECTION.json').exists(),'Frozen selection already exists'
idx=read(R/'t8_oval_20261031/training_index.json');side=read(E/'v5_multiview_20261112/OFFSETS_V2.json');rng=np.random.default_rng(0);rows=[]
for f,fold in enumerate(idx['folds']['folds']):
 for person in sorted(rng.choice(sorted(fold['ubfc_val_subjects']),4,replace=False)):
  rec=next(r for r in idx['records'] if r['dataset']=='UBFC2' and r['recording_id']==person)
  cache=Path(idx['roots']['UBFC2'])/person/'native/crop';receipt=read(cache.parent/'complete.json');source=Path('/data_pool/khangle/facevitals/datasets/UBFC-rPPG/DATASET_2')/person/'vid.avi'
  rows.append(dict(group='UBFC',domain='UBFC',recording=person,person=person,fold=f,cache=str(cache),source=str(source),source_sha256=receipt['source_sha256'],starts=rec['eligible_starts'],ppg_sign=1))
 eligible=[]
 for person in sorted(fold['mcd_val_subjects']):
  pairs=[]
  for step in ['before','after']:
   full=next((r for r in idx['records'] if r['dataset']=='MCD' and r['recording_id']==f'{person}_FullHDwebcam_{step}'),None)
   iri=next((r for r in side if r['person']==person and r['camera']=='IriunWebcam' and r['step']==step),None)
   if full and iri:pairs.append((full,iri))
  if pairs:eligible.append((person,pairs[0]))
 assert len(eligible)>=4
 for j in sorted(rng.choice(len(eligible),4,replace=False)):
  person,(full,iri)=eligible[j];rid=full['recording_id'];cache=Path(idx['roots']['MCD'])/rid/'native/crop';receipt=read(cache.parent/'complete.json')
  rows.append(dict(group='MCD FullHD',domain='MCD FullHD',recording=rid,person=person,fold=f,cache=str(cache),source=str(R/'datasets/MCD-rPPG/video'/f'{rid}.avi'),source_sha256=receipt['source_sha256'],starts=full['eligible_starts'],ppg_sign=-1))
  rows.append(dict(group='MCD Iriun',domain='MCD Iriun',recording=iri['recording'],person=person,fold=f,source=str(R/'datasets/MCD-rPPG'/iri['raw_row']['video']),source_sha256=iri['source_sha256'],raw_row=iri['raw_row'],ppg_sign=-1))
dev=read(OLD/'dev_manifest.json')['videos'];base=[r for r in csv.DictReader((B/'VIDEO_PREDICTIONS.csv').open()) if r['condition']=='native' and r['method']=='calibrated_median_POS_CHROM_V3'];assert len(base)==218
base=sorted(base,key=lambda r:(abs(float(r['prediction'])-float(r['truth'])),r['recording']))
for label,rr in [('BTC best',base[:20]),('BTC worst',base[-20:])]:
 for a in rr:
  v=next(v for v in dev if v['recording']==a['recording']);rows.append(dict(group=label,domain='BTC '+v['group'],recording=v['recording'],person=v['person'],fold=v['fold'],source=v['path'],source_sha256=v['sha256'],drive_id=v['drive_id'],cache=str(OLD/'dev'/v['drive_id']/'native'),pr=float(v['pr_bpm']),calibrated=float(a['prediction']),calibrated_abs_error=abs(float(a['prediction'])-float(a['truth'])),ppg_sign=None,task=v['task']))
assert len(rows)==100 and len({r['recording'] for r in rows})==100
for g in sorted({r['group'] for r in rows}):
 group=sorted([r for r in rows if r['group']==g],key=lambda r:r['recording'])
 for i,r in enumerate(group):r['evm']=i<3
rows=sorted(rows,key=lambda r:(r['group'],r['recording']))
for i,r in enumerate(rows):r['key']=f'{i:03d}_{r["recording"]}'
save(P/'SELECTION.json',rows)
cp={str(f):dict(path=str(checkpoint(f)),sha256=sha(checkpoint(f))) for f in range(5)}
save(P/'FROZEN.json',dict(utc=now(),selection_sha256=sha(P/'SELECTION.json'),protocol_sha256=sha(P/'PROTOCOL.md'),checkpoints=cp,index_sha256=sha(R/'t8_oval_20261031/training_index.json'),dev_sha256=sha(OLD/'dev_manifest.json'),baseline_sha256=sha(B/'VIDEO_PREDICTIONS.csv'),code={p.name:sha(p) for p in P.glob('*.py')},budget=budget(fresh=True)))
print('LOCKED',len(rows),[(g,sum(r['group']==g for r in rows)) for g in sorted({r['group'] for r in rows})])
