"""AiPEX forehead ROI, no landmarks; assumed acquisition15Hz, never fit from labels."""
import os,sys,csv,json,subprocess,hashlib
from pathlib import Path
import numpy as np
import cv2
from common import P,R,read,sha,save,guard
ROOT=Path('/data_pool/khangle/facevitals/datasets/AiPEX')
def subjects():
 prov=read(R/'aipex_diagnostic_20261031/provenance.json');out=[]
 for group in ['India','Sierra_Leone']:
  label=ROOT/'Ground_Truth'/(group+'_GT.csv');assert sha(label)==prov['labels'][group]
  with label.open(encoding='utf-8-sig') as f:
   for row in list(csv.reader(f))[1:]:
    if row and row[0].strip():out.append(dict(person=group+':'+row[0].strip(),group=group,subject=row[0].strip(),pr=float(row[1])))
 assert len(out)==88 and len({r['person'] for r in out})==88
 return out
def prepare():
 from common import lock
 lease=lock('aipex_cache.lock');cv2.setNumThreads(1);out=[]
 for r in subjects():
  source=ROOT/'Data'/r['group']/(r['subject']+'_forehead.avi');receipt=read(source.with_name(source.name+'.complete.json'));assert sha(source)==receipt['sha256']
  d=P/'aipex'/r['group']/r['subject'];d.mkdir(parents=True,exist_ok=True)
  if (d/'COMPLETE.json').exists():assert read(d/'COMPLETE.json')['source_sha256']==receipt['sha256'];continue
  guard();target=d/'crop.npz';assert not target.exists(),'preserve partial'
  raw=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(source),'-vf','scale=36:36:flags=area','-filter_threads','1','-threads','1','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
  assert len(raw)%(36*36*3)==0;rgb=np.frombuffer(raw,np.uint8).reshape(-1,36,36,3)
  with target.open('wb') as f:np.savez_compressed(f,rgb=rgb)
  g=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=pict_type','-of','json',str(source)]));types=[f['pict_type'] for f in g['frames']];assert len(types)==len(rgb)
  centers=[];periods={}
  for typ in ['I','P']:
   ii=np.where(np.array(types)==typ)[0];period=float(np.median(np.diff(ii))) if len(ii)>1 else None;periods[typ]=period
   if period:
    centers += [k*15/period for k in range(1,7) if .7<=k*15/period<=3]
  save(d/'COMPLETE.json',dict(**r,source_sha256=receipt['sha256'],crop_sha256=sha(target),frames=len(rgb),fps_assumed=15,acquisition_fps_verified=False,ROI='forehead only, fixed before outcomes, no landmark',artifact_hz=sorted(set(centers)),periods_frames=periods,holdout_evaluations=0));out.append(r['person'])
  save(P/'AIPEX_STATUS.json',dict(completed=len(list((P/'aipex').glob('*/*/COMPLETE.json'))),total=88))
 save(P/'AIPEX_COMPLETE.json',dict(subjects=88,ROI='forehead only',assumed_fs=15))
class AiPEX:
 def __init__(self):
  assert (P/'AIPEX_COMPLETE.json').exists();self.rows=subjects();self.cache={}
 def sample(self,rng,weak,n=2):
  choices=[r for r in self.rows if not weak or 42<=r['pr']<=180];ix=rng.choice(len(choices),n,replace=False);xs=[];prs=[];hz=[];ids=[]
  for i in ix:
   r=choices[i];d=P/'aipex'/r['group']/r['subject'];a=read(d/'COMPLETE.json')
   if r['person'] not in self.cache:
    assert sha(d/'crop.npz')==a['crop_sha256']
    with np.load(d/'crop.npz') as z:self.cache[r['person']]=z['rgb'].copy()
   rgb=self.cache[r['person']];t=np.arange(len(rgb))/15;assert t[-1]>=10
   start=float(rng.uniform(0,max(0,t[-1]-10)));q=start+np.arange(301)/30;j=np.searchsorted(t,q,side='right').clip(1,len(t)-1);w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
   x=rgb[j-1].astype(np.float32)*(1-w[:,None,None,None])+rgb[j].astype(np.float32)*w[:,None,None,None];assert x.std()>1e-6
   xs.append(((x-x.mean())/x.std()).transpose(0,3,1,2));prs.append(r['pr'] if weak else None);hz.append(a['artifact_hz']);ids.append(dict(dataset='AiPEX',person=r['person'],start=start))
  return np.stack(xs),prs,hz,ids
if __name__=='__main__':prepare()
