"""Reuse frozen public codec cache; separate canonical, aligned sideview cache."""
import sys,hashlib,json
import numpy as np
from runtime import *
sys.path.insert(0,str(E));from training_provider import EncoderProvider,DiskProvider as OriginalDiskProvider
class DiskProvider(OriginalDiskProvider):
    def prepare(self,ds,rid,start,recipe,lowres,fold):
        key=json.dumps([ds,rid,float(start),recipe,bool(lowres),fold],separators=(',',':'));digest=hashlib.sha256(key.encode()).hexdigest()
        assert (E/'training_cache'/str(fold)/(digest+'.json')).exists(),'Missing public cache: refuse unexpected training-time encoding'
        return super().prepare(ds,rid,start,recipe,lowres,fold)
class SideProvider(EncoderProvider):
    def __init__(self,bounds):
        index=read(old.p.INDEX);index['roots']['MCD_SIDE']=str(P/'side_native');super().__init__(index,bounds,300_000_000)
        self.side={r['recording_id']:r for r in read(P/'SIDE_INDEX.json')}
        self.manifest['records'] += [dict(dataset='MCD_SIDE',recording=k,source=v['source']) for k,v in self.side.items()]
    def metadata(self,ds,rid):
        assert ds=='MCD_SIDE'
        if (ds,rid) not in self.meta:
            d=P/'side_native'/rid/'native/crop';r=self.side[rid];assert sha(d/'meta.npz')==r['cache_files']['meta.npz'];assert r['polarity']['convention']=='blood_volume_positive_v1' and r['polarity']['raw_to_canonical_sign']==-1
            with np.load(d/'meta.npz') as z:raw={k:z[k].copy() for k in z.files}
            assert np.array_equal(raw['t'],raw['original_t']),'Sideview source ordinal/points mismatch; inspect before encoding'
            self.meta[ds,rid]=(raw,raw) # Already canonical and time-shifted, NEVER flip/shift again.
        return self.meta[ds,rid]
    def prepare(self,ds,rid,start,recipe,lowres,fold):
        key=json.dumps([ds,rid,float(start),recipe,bool(lowres),fold],separators=(',',':'));digest=hashlib.sha256(key.encode()).hexdigest();d=P/'side_aug'/str(fold);file=d/(digest+'.npz');rp=d/(digest+'.json')
        if rp.exists():
            r=read(rp);assert r['key']==key and sha(file)==r['sha256']
            with np.load(file) as z:t,rgb,valid=z['times'].copy(),z['rgb'].copy(),z['valid'].copy()
            _,m=self.metadata(ds,rid);return t,rgb,valid,m,r['encoding']
        assert not file.exists(),'Partial cache exists; preserve/inspect'
        budget();t,rgb,valid,m,enc=super().prepare(ds,rid,start,recipe,lowres,fold);d.mkdir(parents=True,exist_ok=True)
        with file.open('wb') as f:np.savez_compressed(f,times=t,rgb=rgb,valid=valid)
        save(rp,dict(key=key,sha256=sha(file),encoding=enc,offset_receipt_sha256=self.side[rid]['offset_receipt_sha256']))
        return t,rgb,valid,m,enc
