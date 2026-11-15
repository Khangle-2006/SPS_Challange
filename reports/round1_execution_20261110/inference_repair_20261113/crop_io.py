from common_repair import *
import numpy as np

def crop_path(j,c):
 if c.startswith('c5prime'):
  root=E/'c5prime'/j['dataset']/j['recording']/c.split('_',1)[1];a=read(root/'COMPLETE.json')
  if a['state']!='complete' or not a['bitrate_pass']:raise ValueError('C5prime_source_ineligible')
  return root/'crop'
 if c=='native':return Path(j['crop'])
 if j['dataset']=='BTC':return O/'dev'/j['drive_id']/'c5'
 return Path(j['crop']).parent.parent/'f10_b500/crop'

def cached(j,c):
 d=crop_path(j,c);m=dict(np.load(d/'meta.npz'))
 if (d/'rgb.npy').exists():
  a=read(d.parent/'complete.json')['cache_files']
  for n in ['rgb.npy','meta.npz']:assert sha(d/n)==a[n]
  rgb=np.asarray(np.load(d/'rgb.npy',mmap_mode='r'))
 else:
  a=read(d/'COMPLETE.json');assert sha(d/'crop.mkv')==a['crop_sha256'] and sha(d/'meta.npz')==a['meta_sha256']
  b=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(d/'crop.mkv'),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']);rgb=np.frombuffer(b,np.uint8).reshape(-1,36,36,3)
 assert len(rgb)==len(m['t']);return rgb,m,d

def source_path(j,c):
 if c=='native':return Path(j['source']),j['source_sha256']
 if c.startswith('c5prime'):
  d=E/'c5prime'/j['dataset']/j['recording']/c.split('_',1)[1];return d/'video.mp4',read(d/'COMPLETE.json')['encoded_sha256']
 if j['dataset']=='BTC':
  d=O/'dev'/j['drive_id']/'c5_encoded';return d/'video.mp4',sha(d/'video.mp4')
 a=read(crop_path(j,c).parent/'complete.json')['encoding'];path=Path(a['video'])
 if not path.exists() and j['dataset']=='UBFC2':path=P/'legacy_sources'/(j['recording']+'.mp4')
 return path,a['video_sha256_at_reuse_or_creation']
