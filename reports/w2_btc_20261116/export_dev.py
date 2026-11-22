import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import csv,json,hashlib,subprocess,zipfile,fcntl,traceback
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;B=R/'round1_v4_20261105/btc_zeroshot';E=R/'round1_execution_20261110';Y=R/'round1_bayes_20261108'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def main():
 lease=(P/'export.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=P/'numeric_dev';assert not out.exists(),'Inspect partial, no automatic overwrite';out.mkdir()
 raw=list(csv.DictReader(open(E/'inference_repair_20261113/report_original/BTC_VIDEOS.csv')))
 cohort={r['recording'] for r in raw if r['condition']=='native'};assert len(cohort)==218
 manifest=json.load(open(B/'dev_manifest.json'));vs=[v for v in manifest['videos'] if v['recording'] in cohort];assert len(vs)==218
 people=sorted({v['person'] for v in vs});assert len(people)==158
 vm={k:f'video{i+1:03}' for i,k in enumerate(sorted(cohort))};pm={k:f'person{i+1:03}' for i,k in enumerate(people)}
 mapping={**vm,**pm};forbidden={v['drive_id'] for v in manifest['videos']}
 def clean(x):
  if isinstance(x,dict):return {mapping.get(k,k):clean(v) for k,v in x.items() if k not in ['name','patient_name','path','drive_id','sheet_row','url','source_path']}
  if isinstance(x,list):return [clean(a) for a in x]
  if isinstance(x,str):
   if x in mapping:return mapping[x]
   if x in forbidden or x.startswith(('/mnt/','/data/','/data_pool/')):return '[removed]'
   for a,b in sorted(mapping.items(),key=lambda z:-len(z[0])):x=x.replace(a,b)
  return x
 save(P/'PRIVATE_EXPORT_MAPPING.json',{'video':vm,'person':pm})
 records=[]
 for v in vs:
  assert sum(f.stat().st_size for f in P.rglob('*') if f.is_file())<1_000_000_000,'STOP_NEW_SCOPE_1GB'
  d=B/'dev'/v['drive_id']/'native';receipt=json.load(open(d/'COMPLETE.json'))
  assert sha(d/'crop.mkv')==receipt['crop_sha256'] and sha(d/'meta.npz')==receipt['meta_sha256']
  with np.load(d/'meta.npz',allow_pickle=False) as z:meta={k:z[k].copy() for k in ['t','valid','rgb_mean','boxes']}
  data=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(d/'crop.mkv'),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
  rgb=np.frombuffer(data,np.uint8).reshape(-1,36,36,3);assert len(rgb)==len(meta['t'])
  grid=rgb.reshape(len(rgb),6,6,6,6,3)
  blocks=np.rint(128*grid.mean(axis=(2,4))).astype(np.uint16)
  fill=np.rint(255*np.any(grid!=0,axis=-1).mean(axis=(2,4))).astype(np.uint8)
  target=out/'blocks'/(vm[v['recording']]+'.npz');target.parent.mkdir(exist_ok=True)
  np.savez_compressed(target,blocks=blocks,fill=fill,**meta)
  with np.load(target) as z:
   for k in meta:assert np.array_equal(z[k],meta[k],equal_nan=True)
  roi=Y/'traces'/v['recording']/'native/trace.npz';target=out/'roi'/(vm[v['recording']]+'.npz');target.parent.mkdir(exist_ok=True)
  target.write_bytes(roi.read_bytes());assert sha(target)==sha(roi)
  records.append(dict(recording=vm[v['recording']],person=pm[v['person']],fold=v['fold'],frames=len(rgb),group=v['group'],task=v['task'],pr_bpm=v['pr_bpm']))
  save(P/'EXPORT_STATUS.json',{'complete_videos':len(records),'total':218,'frames':sum(x['frames'] for x in records)})
 files=[B/'ALL_METRICS.csv',B/'VIDEO_PREDICTIONS.csv',B/'CALIBRATION_OOF.json',Y/'VIDEO_PREDICTIONS.csv',Y/'METRICS.json',Y/'CALIBRATION_FITS.json',Y/'domain_sweep_20261109/PBV_SIGNATURES.json',E/'inference_repair_20261113/report_original/WINDOWS.csv',E/'inference_repair_20261113/report_original/BTC_VIDEOS.csv',E/'sweep/fold2/history.json',E/'sweep/fold2/NATIVE_REPORT.md',E/'sweep/fold2/WEAK_COLLAPSE_GATE.json']
 for src in files:
  dst=out/'tables'/src.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True)
  if src.suffix=='.json':save(dst,clean(json.load(open(src))))
  elif src.suffix=='.csv':
   with src.open() as f:reader=csv.DictReader(f);fields=[k for k in reader.fieldnames if k not in ['name','path','drive_id','sheet_row']];rows=[clean(r) for r in reader if not r.get('recording') or r['recording'] in cohort]
   with dst.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
  else:dst.write_text(clean(src.read_text()))
 save(out/'tables/dev_manifest.json',{'videos':[clean({k:v[k] for k in ['recording','person','fold','group','skin','task','lighting','pr_bpm']}) for v in vs],'video_count':218,'people_count':158})
 plan=json.load(open(E/'PLAN_ELIGIBLE.json'));run=next(r for r in plan['runs'] if r['fold']==2)
 save(out/'tables/fold2_plan.json',{k:plan[k] for k in ['epochs','learning_rate']}|{'btc_aux_weight':plan['btc_aux_weight']['2'],'steps_per_epoch':run['steps_per_epoch']})
 save(out/'VIDEOS.json',records)
 (out/'README.md').write_text('# BTC dev numeric export\n\n218 videos,158 people; no holdout, face images or direct names. Video/person codes are local pseudonyms; re-identification map is not included. Not anonymous medical data: PR and derived physiology remain sensitive.\n\nblocks=round(128*RGB block mean),uint16[T,6,6,3]; decode by /128. fill=round(255*fraction pixels with any nonzero RGB channel),uint8[T,6,6]; decode by /255. Spatial order is row,column. Metadata t/valid/rgb_mean/boxes copied exactly. ROI NPZ copied byte-for-byte. PR is a video scalar, not a synchronized waveform. t in seconds.\n\nCSV recording rows restricted to218 videos; JSON calibration and aggregate tables retain historical fit/cohort scope and must not be assumed matched automatically. Direct identifiers stripped/replaced. No new model scoring. Original fold2 plan weight0.1 is historical, not W2 weight1.0.\n\nSHA256 for each payload file is in SHA256.json. ZIP part manifests are in delivery receipt. Combine extracted parts into one folder.\n')
 save(out/'SHA256.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
 video_fold={r['recording']:r['fold'] for r in records}
 payload=sorted((f for f in out.rglob('*') if f.is_file()),key=lambda f:(video_fold.get(f.stem,-1),str(f)))
 dest=P/'delivery';dest.mkdir();parts=[];group=[];size=0;previous=None
 # Conservative uncompressed bound ensures every ZIP stays below25MB, including overhead.
 for f in payload:
  fold=video_fold.get(f.stem,-1)
  if group and (size+f.stat().st_size>24_000_000 or fold!=previous):
   parts.append(group);group=[];size=0
  assert f.stat().st_size<24_000_000,'Single artifact exceeds part cap; inspect before splitting'
  group.append(f);size+=f.stat().st_size;previous=fold
 if group:parts.append(group)
 receipts=[]
 for i,fs in enumerate(parts):
  z=dest/f'BTC_DEV_NUMERIC_part{i+1:02}.zip'
  with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as a:
   for f in fs:a.write(f,str(f.relative_to(out)))
  with zipfile.ZipFile(z) as a:
   assert a.testzip() is None
   for f in fs:assert hashlib.sha256(a.read(str(f.relative_to(out)))).hexdigest()==sha(f)
  assert z.stat().st_size<=25_000_000;receipts.append({'file':z.name,'bytes':z.stat().st_size,'sha256':sha(z)})
 save(P/'EXPORT_COMPLETE.json',{'videos':218,'people':158,'frames':sum(x['frames'] for x in records),'parts':receipts,'holdout_accessed':False})
if __name__=='__main__':
 try:main()
 except Exception:save(P/'EXPORT_ERROR.json',{'traceback':traceback.format_exc()});raise
