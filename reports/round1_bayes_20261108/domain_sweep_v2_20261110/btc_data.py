"""Read only enumerated dev records, verify cache hashes, retain source timestamps."""
import json,sys,subprocess
from pathlib import Path
from collections import OrderedDict
import numpy as np
from codec_provider import ROOT,sha,read
from t8_t1_loader import supported
P=Path(__file__).resolve().parent;BTC=P.parent.parent/'round1_v4_20261105/btc_zeroshot'
SPLIT='97944f717b7fc166bb8a0e91c791c7ad6b7e451cffed564f40232dde3ec8e221'
class BTCData:
 def __init__(self,fold,max_bytes=1_000_000_000):
  assert sha(BTC/'btc_split.json')==SPLIT
  split=read(BTC/'btc_split.json');manifest=read(BTC/'dev_manifest.json');assert manifest['split_sha256']==SPLIT
  self.rows=manifest['videos'];assert len(self.rows)==221
  assert all(split['people'][v['person']]['partition']=='dev' for v in self.rows)
  self.fold=fold;self.cache=OrderedDict();self.bytes=0;self.max_bytes=max_bytes;self.hashes={};self.train=[];self.exclusions={}
  for v in self.rows:
   if v['fold']==fold:continue
   for start in self.starts(v,'native'):
    self.train.append((v,start))
  self.people=sorted({v['person'] for v,s in self.train})
  self.by_person={person:[x for x in self.train if x[0]['person']==person] for person in self.people}
 def folder(self,v,condition):
  assert v in self.rows and condition in ['native','c5']
  return BTC/'dev'/v['drive_id']/condition
 def starts(self,v,condition):
  return sorted({r['start'] for r in self.candidates(v,condition) if r['scheme']=='10s' and r['support_valid']})
 def candidates(self,v,condition):
  source=BTC/'dev'/v['drive_id']/'SOURCE.json';s=read(source);assert s['source_sha256']==v['sha256'];self.hashes[str(source)]=sha(source)
  if s['source_span_s']<10:
   self.exclusions[v['recording']]='source shorter than 10 seconds; no padding';return []
  f=BTC/'predictions'/v['drive_id']/condition/'classical.json';self.hashes[str(f)]=sha(f);return read(f)['rows']
 def load(self,v,condition):
  key=v['drive_id'],condition
  if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
  d=self.folder(v,condition);receipt=read(d/'COMPLETE.json')
  for name,field in [('crop.mkv','crop_sha256'),('meta.npz','meta_sha256')]:
   assert sha(d/name)==receipt[field];self.hashes[str(d/name)]=receipt[field]
  with np.load(d/'meta.npz') as m:t=m['t'].copy();valid=m['valid'].copy()
  b=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(d/'crop.mkv'),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
  assert len(b)==len(t)*36*36*3
  rgb=np.frombuffer(b,np.uint8).reshape(len(t),36,36,3)
  while self.cache and self.bytes+rgb.nbytes>self.max_bytes:
   _,old=self.cache.popitem(last=False);self.bytes-=old[0].nbytes
  assert rgb.nbytes<=self.max_bytes;self.cache[key]=rgb,t,valid;self.bytes+=rgb.nbytes
  return rgb,t,valid
 def clip(self,v,start,condition='native'):
  rgb,t,valid=self.load(v,condition);q=start+np.arange(301)/30
  if not supported(t,valid,q[0],q[-1]):raise ValueError('BTC image support rejected')
  j=np.searchsorted(t,q,side='right').clip(1,len(t)-1);w=((q-t[j-1])/(t[j]-t[j-1])).astype(np.float32)
  x=rgb[j-1].astype(np.float32)*(1-w[:,None,None,None])+rgb[j].astype(np.float32)*w[:,None,None,None]
  if not np.isfinite(x).all() or x.std()<1e-6:raise ValueError('BTC constant/nonfinite clip')
  return ((x-x.mean())/x.std()).transpose(0,3,1,2).copy()
 def aux(self,rng,weak):
  people=[p for p in self.people if any(42<=v['pr_bpm']<=180 or not weak for v,s in self.by_person[p])]
  chosen=rng.choice(people,4,replace=False);xs=[];prs=[];freq=[];ids=[]
  for person in chosen:
   choices=[z for z in self.by_person[person] if not weak or 42<=z[0]['pr_bpm']<=180]
   v,start=choices[int(rng.integers(len(choices)))];assert v['fold']!=self.fold
   xs.append(self.clip(v,start));prs.append(v['pr_bpm'] if weak else None);ids.append((v['person'],v['recording'],start))
   g=P.parent/'native_gop'/(v['recording']+'.json');centers=[]
   if g.exists():
    a=read(g);assert a['source_sha256']==v['sha256'];centers=[z['hz'] for z in a['artifact_frequencies'] if .7<=z['hz']<=3];self.hashes[str(g)]=sha(g)
   freq.append(centers)
  return np.stack(xs),prs,freq,ids
