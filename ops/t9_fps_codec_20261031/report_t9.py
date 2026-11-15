import argparse,csv,json
from pathlib import Path
import numpy as np
from score_round1 import score

def read(p):return json.loads(p.read_text())
def load(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bootstrap(values):
 a=np.asarray(list(values.values()),float);rng=np.random.default_rng(20261031)
 return dict(subjects=len(a),mean=float(a.mean()),CI95=np.quantile(rng.choice(a,size=(10000,len(a)),replace=True).mean(1),[.025,.975]).tolist(),per_subject=values) if len(a) else dict(subjects=0,mean=None,CI95=None)
def main(a):
 state=read(a.input/'status.json');cohort=read(a.input/'cohort.json');assert state['state']=='complete' and state['completed']==35 and not state['errors']
 records=[read(a.input/r/'complete.json') for r in cohort['recordings']];assert len({json.dumps(r['fingerprint'],sort_keys=True) for r in records})==1
 a.output.mkdir(exist_ok=False,parents=True);refs=[];outputs={m:[] for m in ['native','raw10','h26430','h26410']}
 for r in cohort['recordings']:
  refs+=load(a.input/r/'reference_signal.csv')
  for m in outputs:outputs[m]+=load(a.input/r/(m+'_output.csv'))
 assert len(refs)==207;write(a.output/'reference_signal.csv',refs);results={}
 for m,rows in outputs.items():
  write(a.output/(m+'_output.csv'),rows);results[m]=score(a.output/'reference_signal.csv',a.output/(m+'_output.csv'));(a.output/(m+'_metrics.json')).write_text(json.dumps(results[m],indent=2))
 def keyed(result):return {(r['subject_id'],r['recording_id'],r['window_start_s']):r for r in result['windows']}
 maps={m:keyed(v) for m,v in results.items()};assert all(set(v)==set(maps['native']) for v in maps.values())
 paired={}
 for name,weights in {'fps_only':{'raw10':1,'native':-1},'codec_at30':{'h26430':1,'native':-1},'codec_at10':{'h26410':1,'raw10':-1},'interaction':{'h26410':1,'raw10':-1,'h26430':-1,'native':1}}.items():
  per={};n=0
  for key in maps['native']:
   if all(maps[m][key]['accepted'] for m in weights):
    values=[maps[m][key]['reference_hr_bpm'] for m in weights];assert len(set(values))==1
    delta=sum(w*abs(maps[m][key]['prediction_hr_bpm']-maps[m][key]['reference_hr_bpm']) for m,w in weights.items());per.setdefault(key[0],[]).append(delta);n+=1
  paired[name]=dict(windows=n,**bootstrap({s:float(np.mean(v)) for s,v in per.items()}))
 summary=dict(records=35,candidates=207,metrics={m:{k:v[k] for k in ['counts','subject_macro','window_level','reference_coverage']} for m,v in results.items()},paired=paired,raw10_above15_flag=results['raw10']['subject_macro']['mae'] is not None and results['raw10']['subject_macro']['mae']>15,interpretation='35dev POS diagnostic; threshold is investigation trigger, not proof of a bug; no T8 outcome or untouched test',per_record_stream_counts={r['recording_id']:r['counts'] for r in records})
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 names=list(results);ax[0].bar(names,[results[m]['subject_macro']['mae'] for m in names]);ax[0].set(ylabel='Subject macro MAE (bpm)',title='Same35development subjects; inspect coverage')
 vals=list(paired['interaction']['per_subject'].values());ax[1].hist(vals,bins=15);ax[1].axvline(0,color='black',linestyle='--');ax[1].set(xlabel='Interaction of absolute error (bpm)',ylabel='Subjects',title='Only common accepted windows')
 fig.savefig(a.output/'comparison.png',dpi=180);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
