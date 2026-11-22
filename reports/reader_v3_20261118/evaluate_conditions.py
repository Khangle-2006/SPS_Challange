"""CPU-only transformed-data evaluation using FROZEN native outer-fold models.
No refit; original-source capture class; no holdout. Run after C_COMPLETE.
"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import argparse,json,csv,hashlib,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'source'))
import lib_btc as lib
from reader_models import pmed

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--reports-root',required=True,type=Path);a.add_argument('--numeric-root',required=True,type=Path);a.add_argument('--source-metadata',required=True,type=Path);a.add_argument('--out',required=True,type=Path);args=a.parse_args()
 import fcntl
 args.out.mkdir(parents=True,exist_ok=True);lock=(args.out/'evaluate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (args.out/'COMPLETE.json').exists()
 R=args.reports_root;W=R/'w2_btc_20261116';C=R/'reader_w2b_20261116';V=R/'round1_v4_20261105/btc_zeroshot';E=R/'round1_execution_20261110'
 assert (C/'C_COMPLETE.json').exists(),'C1/C4 extraction still incomplete: inspect workers, do not duplicate'
 videos=json.loads((args.numeric_root/'VIDEOS.json').read_text());assert len(videos)==218
 mapping=json.loads((W/'PRIVATE_EXPORT_MAPPING.json').read_text())['video'];assert set(mapping.values())=={v['recording'] for v in videos}
 inv={v:k for k,v in mapping.items()};dev={v['recording']:v for v in json.loads((V/'dev_manifest.json').read_text())['videos'] if v['recording'] in mapping}
 meta={r['recording']:r for r in csv.DictReader(args.source_metadata.open())};assert all(v['recording'] in meta for v in videos)
 foldmodels=json.loads((P/'FOLD_MODELS.json').read_text())
 base=list(csv.DictReader((P/'reproduction/reader_v3/OOF_PREDICTIONS.csv').open()));base={r['recording']:r for r in base}
 sys.path.insert(0,str(E));from report_metrics import metric,belongs
 rows=[];exclusions=[];proof=[]
 for condition in ['c1_050','c1_025','c1_0125','c4_QVGA','c4_VGA','c5','c5prime_default','c5prime_bf0']:
  current=[]
  for v in videos:
   anon=v['recording'];real=inv[anon];source=dev[real]
   assert float(source['pr_bpm'])==float(v['pr_bpm']) and source['fold']==v['fold'] and source['task']==v['task']
   if condition.startswith(('c1_','c4_')):d=C/'constraints'/condition/real
   elif condition=='c5':d=V/'dev'/source['drive_id']/condition
   else:
    root=E/'c5prime/BTC'/real/condition.split('_')[-1];receipt=json.loads((root/'COMPLETE.json').read_text())
    if receipt['state']!='complete' or not receipt['bitrate_pass']:
     exclusions.append(dict(recording=anon,condition=condition,reason=receipt.get('reason','bitrate_above500')));continue
    d=root/'crop'
   if not (d/'meta.npz').exists():exclusions.append(dict(recording=anon,condition=condition,reason='meta_unavailable'));continue
   receipt=json.loads((d/'COMPLETE.json').read_text());assert sha(d/'meta.npz')==receipt['meta_sha256']
   with np.load(d/'meta.npz',allow_pickle=False) as n:
    with np.errstate(all='ignore'):s=lib.avg_spectra(n['t'],n['valid'],n['rgb_mean'])
   z=None if s is None else (lib.zlog(s[0])+lib.zlog(s[1]))/2
   if z is not None and not np.isfinite(z).all():z=None
   width=int(meta[anon]['width']);height=int(meta[anon]['height']);cls='low' if width*height<1800000 else 'high'
   m=foldmodels[int(v['fold'])];assert v['person'] in m['validation_people'] and v['person'] not in m['train_people']
   value=pmed(np.zeros(len(lib.GRID)) if z is None else z,np.array(m['prior']),m['beta_by_class'][cls])+m['offset_by_cell'][cls+('_task' if v['task'].startswith('Task') else '_not_task')]
   common=dict(recording=anon,person=v['person'],fold=v['fold'],task=v['task'],group=v['group'],truth=float(v['pr_bpm']),condition=condition)
   current.append(dict(common,method='reader_v3',prediction=float(value),abstain=z is None,source_class=cls,source_width=width,source_height=height))
   for method,key in [('constant','constant'),('calibrated','calibrated'),('reader_native','reader_v3')]:current.append(dict(common,method=method,prediction=float(base[anon][key])))
   proof.append(dict(recording=anon,condition=condition,meta_sha256=receipt['meta_sha256'],source_width=width,source_height=height,source_class=cls))
  rows.extend(current);(args.out/'STATUS.json').write_text(json.dumps(dict(condition=condition,videos=len(current)//4)))
 metrics=[]
 for cond in sorted({r['condition'] for r in rows}):
  for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
   subset=[r for r in rows if r['condition']==cond and belongs(r,group)]
   batch=[dict(a,condition=cond,group=group) for a in metric(subset,['constant','calibrated','reader_native','reader_v3'])]
   look={m:{r['recording']:r for r in subset if r['method']==m} for m in ['constant','calibrated','reader_native','reader_v3']}
   for result in batch:
    paired=[]
    for person in sorted({r['person'] for r in subset}):
     ids=[k for k,r in look[result['method']].items() if r['person']==person]
     paired.append(np.mean([abs(look[result['method']][k]['prediction']-look[result['method']][k]['truth'])-abs(look['reader_native'][k]['prediction']-look['reader_native'][k]['truth']) for k in ids]))
    dd=np.array(paired);ix=np.random.default_rng(0).integers(len(dd),size=(10000,len(dd)))
    result.update(delta_native=float(dd.mean()),delta_native_CI=np.quantile(dd[ix].mean(1),[.025,.975]).tolist())
   metrics.extend(batch)
 for name,value in [('PREDICTIONS',rows),('METRICS',metrics),('EXCLUSIONS',exclusions),('SOURCE_CLASS_BINDING',proof)]:
  (args.out/(name+'.json')).write_text(json.dumps(value,indent=2))
 (args.out/'COMPLETE.json').write_text(json.dumps(dict(passed=True,holdout_evaluations=0,refit=False,fold_models_sha256=sha(P/'FOLD_MODELS.json'),source_metadata_sha256=sha(args.source_metadata))))
if __name__=='__main__':main()
