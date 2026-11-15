"""Read-only artifact validation and paired mechanism tables; no experiment changes."""
from audit_clock import *
from report_clock import loadrows,writecsv
from collections import defaultdict,Counter

def main():
 plan=read(P/'PLAN.json'); checks=[];groups=defaultdict(list); artifacts=[]
 assert sha(P/'PLAN.json')=='ba51ddb8e585942a4491400eb70f8c7ae3722182caccc41d3885a9c6d5b2786c'
 assert plan['bindings']==binding()
 for version,key in [('V2','v2_sha'),('V3','v3_sha'),('V4','v4_sha')]:
  a=read(P/f'AMENDMENT_{version}.json')
  assert sha(P/f'run_{version.lower()}.py')==a[key]
 assert sha(V/'btc_zeroshot/btc_split.json')==SPLIT
 assert (P/'REAL_COMPLETE.json').exists() and (P/'SYNTH_COMPLETE.json').exists()
 for kind in ['real','synthetic']:
  paths=list((P/kind).glob('*/*/f*/RESULT.json')) if kind=='real' else list((P/kind).glob('*/*/RESULT.json'))
  encoded=0;before=0
  for path in paths:
   a=read(path);d=path.parent
   assert (d/'traces.npz').is_file() and (d/'spectra.npz').is_file()
   with np.load(d/'spectra.npz') as z:
    for r in a['rows']:
     if r.get('peak_bpm') is not None:
      key=f"{r['roi']}_{r['window']}_{r['clock']}_{r['method']}"
      assert np.isfinite(z[key+'_p']).all() and np.isfinite(z[key+'_f']).all()
   if a['bf'] is None:before+=1;continue
   encoded+=1
   assert sha(d/'video.mp4')==a['encoded_sha']
   g=read(d/'GOP.json'); n=30*a['fs']+1
   assert len(g['frames'])==n and np.allclose(a['PTS'],np.arange(n)/a['fs'],atol=2e-5)
   if kind=='real':
    assert a['roi_fixed'] and a['source_sha']==next(j['source_sha'] for j in plan['videos'] if j['recording']==a['recording'])
    key=(kind,a['dataset'],a['recording'],a['fs'])
   else:key=(kind,a['scene'],a['seed'],a['pulse_bpm'],a['fs'])
   groups[key].append(a)
   artifacts.append(dict(path=str(d),encoded_sha=a['encoded_sha'],input_rgb_sha=a['input_rgb_sha'],frames=n,bitrate_pass=a['bitrate_pass']))
  assert (encoded,before)==((66,22) if kind=='real' else (120,40))
  checks.append(dict(kind=kind,encoded=encoded,before=before))
 for key,aa in groups.items():
  assert sorted(a['bf'] for a in aa)==[0,3,4]
  assert len({a['input_rgb_sha'] for a in aa})==1
  if key[0]=='real':
   zz=[np.load(Path(next(x['path'] for x in artifacts if x['encoded_sha']==a['encoded_sha']))/'traces.npz') for a in aa]
   for z in zz[1:]:assert np.array_equal(z['source_t'],zz[0]['source_t']) and np.array_equal(z['sampled_t'],zz[0]['sampled_t'])
   for z in zz:z.close()
 for j in plan['videos']:
  d=P/'real'/j['dataset']/j['recording']
  if j['recording']=='1474_FullHDwebcam_after':
   assert read(d/'EXCLUDED.json')['reason']=='source_clock_gap_repeated_selection' and not list(d.glob('f*/RESULT.json'))
  else:assert (d/'COMPLETE.json').is_file()
 writecsv(P/'VERIFIED_ARTIFACTS.csv',artifacts)
 validation=dict(utc=now(),status='PASS',counts=checks,planned_sources=12,evaluated_sources=11,excluded_source='1474_FullHDwebcam_after',shared_input_groups=len(groups),encoded_sha_verified=len(artifacts),frozen_plan_sha=sha(P/'PLAN.json'),report_code_sha={f:sha(P/f) for f in ['report_clock.py','final_checks.py','summarize_final.py']},holdout_evaluations=0,scope_guard=guard(False),note='Original/V2/V3 errors retained. Completed V2/V3 results reused unchanged by V4; MCD source gap excluded without replacement. Source SHA checked by worker once per video; encoded SHA checked again here.')
 real,_=loadrows('real');lookup={(r['dataset'],r['recording'],r['fs'],r['bf'],r['roi'],r['clock'],r['method'],r['window']):r for r in real}
 paired=[];coverage=[];beforeafter=[];roi=[]
 for r in real:
  if r['clock']!='source_clock' or r['bf'] is None or r['roi']!='oval36':continue
  key=(r['dataset'],r['recording'],r['fs'],r['bf'],r['roi'],r['clock'],r['method'],r['window'])
  common={k:r[k] for k in ['dataset','recording','fs','bf','method','window']}
  if r['fs']==8:
   b=lookup.get(key[:2]+(10,)+key[3:])
   if b and r.get('peak_bpm') is not None and b.get('peak_bpm') is not None:
    cyc=r.get('cycle_bpm') and b.get('cycle_bpm')
    paired.append(common|dict(peak8=r['peak_bpm'],peak10=b['peak_bpm'],delta_peak=b['peak_bpm']-r['peak_bpm'],cycle8=r.get('cycle_bpm'),cycle10=b.get('cycle_bpm'),eligible=bool(cyc),follows_both=bool(cyc and abs(r['peak_bpm']-r['cycle_bpm'])<=2 and abs(b['peak_bpm']-b['cycle_bpm'])<=2)))
  b=lookup.get(key[:3]+(None,)+key[4:])
  if b and r.get('peak_bpm') is not None and b.get('peak_bpm') is not None:
   cycle=r.get('cycle_bpm')
   beforeafter.append(common|dict(before_peak=b['peak_bpm'],after_peak=r['peak_bpm'],cycle=cycle,before_near=bool(cycle and abs(b['peak_bpm']-cycle)<=2),after_near=bool(cycle and abs(r['peak_bpm']-cycle)<=2),before_wave_sd=b['raw_wave_sd'],after_wave_sd=r['raw_wave_sd']))
  b=lookup.get(key[:4]+('skin',)+key[5:])
  if b and r.get('peak_bpm') is not None and b.get('peak_bpm') is not None:roi.append(common|dict(oval36_peak=r['peak_bpm'],skin_peak=b['peak_bpm'],abs_difference=abs(b['peak_bpm']-r['peak_bpm'])))
 for ds in ['UBFC2','MCD','BTC']:
  for fs in [8,10]:
   for bf in [None,0,3,4]:
    for method in ['G','POS','CHROM']:
     rr=[r for r in real if r['dataset']==ds and r['fs']==fs and r['bf']==bf and r['method']==method and r['roi']=='oval36' and r['clock']=='source_clock']
     coverage.append(dict(dataset=ds,fs=fs,bf=bf,method=method,expected_windows=12,observed=len(rr),valid=sum(r['prediction'] is not None for r in rr),reasons=json.dumps(dict(Counter(r['reason'] for r in rr if r['reason'])))))
 writecsv(P/'PAIRED_FPS_MOVEMENT.csv',paired);writecsv(P/'BEFORE_AFTER.csv',beforeafter);writecsv(P/'ROI_COMPARISON.csv',roi);writecsv(P/'COVERAGE.csv',coverage)
 details=list(csv.DictReader((P/'WINDOW_PREDICTIONS.csv').open()));refcov=[]
 for ds in ['UBFC2','MCD','BTC']:
  rr=[r for r in details if r['dataset']==ds]
  refcov.append(dict(dataset=ds,observed_method_window_rows=len(rr),prediction_present=sum(bool(r['prediction']) for r in rr),reference_present=sum(bool(r['truth']) for r in rr),both_present=sum(bool(r['prediction']) and bool(r['truth']) for r in rr),note='Repeated across methods and configurations, not independent windows. Public reference support can fail at time0; never shifted to repair.'))
 writecsv(P/'REFERENCE_COVERAGE.csv',refcov)
 for r in csv.DictReader((P/'BTC_BASELINES_MATCHED.csv').open()):
  assert abs(float(r['abs_error'])-abs(float(r['prediction'])-float(r['truth'])))<1e-10
 save(P/'FINAL_VALIDATION.json',validation)
 print(json.dumps(validation))

if __name__=='__main__':main()
