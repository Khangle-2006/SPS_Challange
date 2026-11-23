"""Read-only inventory of exact old cohorts; all new files stay in OUT."""
from expert_spectrum import *
import collections
assert not (OUT/'PLAN.json').exists(),'Existing R1 plan: inspect rather than overwrite'
for n in ['tmp','cache','logs','status','receipts','source','gates','native','btc_conditions','public_native','public_cond']:(OUT/n).mkdir(parents=True,exist_ok=True)
import y05_openrppg as Y5
_,keys=Y5.btc_keys();assert len(keys)==218 and len(set(keys))==218
R=W.parent/'round1_execution_20261110/inference_repair_20261113'
manifest=json.loads((R/'MANIFEST.json').read_text());assert collections.Counter(j['dataset'] for j in manifest)=={'BTC':218,'UBFC2':35,'MCD':119}
public=[j for j in manifest if j['dataset'] in ('UBFC2','MCD')]
save(OUT/'PUBLIC_DEV_MANIFEST.json',public);save(OUT/'BTC_KEYS.json',keys)
bc=['heavy','split10','track_c1_g25p','aidtrack_c1_g25gv01','c2_vit','c4_vga','c4_qvga','c6_us1','track_c7_blue']
pc=['native','c4_qvga','c6_us1','aidtrack_c1_g25gv01','c2_vit','track_c1_g25p','track_c7_blue','c1_g50p','c1_g12p','c1_g25gv001','c7_white','c5_split10']
tasks=[]
for name in MODELS:
 for c in bc:
  old=W/'out/y11'/c/(name+'.npz')
  with np.load(old) as z:
   assert list(z['keys'])==keys;done=z['done'].astype(bool);finite=np.isfinite(z['S']).all(1)
  if c=='heavy':
   with np.load(W/'out/y05'/(name+'.npz')) as z:ids=np.flatnonzero(~np.isfinite(z['S']).all(1)).tolist()
  else:ids=[i for i,k in enumerate(keys) if (W/('out/x10' if c=='split10' else 'out/x15/'+c)/(k+'.npz')).exists()]
  assert set(np.flatnonzero(done))==set(ids),(name,c,'old completed cohort differs')
  tasks.append(dict(id='btc__'+name+'__'+c,kind='btc',model=name,condition=c,ids=ids,old=str(old),old_sha256=sha(old),expected_output=int(finite.sum()),reuse_native=c=='heavy'))
 for c in pc:
  src=W/('out/x10' if c in ('native','c5_split10') else 'out/x16/'+c)
  jobs=[j for j in public if (src/(j['key']+'.npz')).exists()]
  assert len(jobs)==(154 if c in ('native','c5_split10') else 73),(name,c,len(jobs))
  oldbase=W/('out/y12' if c=='native' else 'out/y13/'+c)/name
  counts={};receipts={}
  for j in jobs:
   p=oldbase/(j['key']+'.npz')
   with np.load(p) as z:
    n=int(np.isfinite(z['spec']).all(1).sum());counts[j['dataset']]=counts.get(j['dataset'],0)+n
   receipts[j['key']]=dict(sha256=sha(p),output_windows=n)
  tasks.append(dict(id='public__'+name+'__'+c,kind='public',model=name,condition=c,keys=[j['key'] for j in jobs],old_base=str(oldbase),old_receipts=receipts,expected_windows=counts))
save(OUT/'PLAN.json',dict(models=MODELS,btc_conditions=bc,public_conditions=pc,tasks=tasks,holdout_reads=0,public_manifest_sha256=sha(R/'MANIFEST.json')))
sources={}
for name in ['y05_openrppg.py','y11_openrppg_conditions.py','y12_openrppg_public.py','y13_openrppg_public_conditions.py','z13_public_p4_prefilter.py','z14_p4m_freeze.py','z17_selfwarm.py','z18_selfwarm_score.py','y01_btc_improve.py','y06_reader_quality.py','y08_fps_sim.py','x05_public_score.py','y02_public_improve.py','y07_public_prior.py']:
 p=W/'code'/name;sources[str(p)]=sha(p);(OUT/'source'/name).write_bytes(p.read_bytes())
for name in ['READER_V4_DEFINITION.txt','DEVIATIONS.txt','P4M_DEFINITION.txt']:
 p=W/'logs'/name;sources[str(p)]=sha(p);(OUT/'source'/name).write_bytes(p.read_bytes())
save(OUT/'SOURCE_HASHES.json',sources);save(OUT/'WEIGHTS.json',verify_weights())
print(json.dumps(dict(tasks=len(tasks),btc_videos=218,public_records=154,public_constraint_records=73,source_hashes=len(sources))))
