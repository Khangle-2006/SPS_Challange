from infer_original import *
from crop_io import cached
from scipy.stats import spearmanr
import cv2,copy
sys.path.insert(0,str(VIS));from model_maps import power

def selection():
 public=[dict(r) for r in read(VIS/'SELECTION.json') if not r['group'].startswith('BTC')];base=[r for r in read(E/'final_priorities_20261111/padding_evaluation/BTC_VIDEO.json') if r['condition']=='native'];jobs={j['recording']:j for j in manifest() if j['dataset']=='BTC'}
 base=[r for r in base if r['recording'] in jobs];assert len(base)==218
 ordered=sorted(base,key=lambda r:(abs(r['prediction']-r['truth']),r['recording']));rows=public
 for group,part in [('BTC padding best',ordered[:20]),('BTC padding worst',ordered[-20:])]:
  for r in part:
   j=jobs[r['recording']];rows.append(dict(j,group=group,domain='BTC '+j['metadata']['group'],padding_prediction=r['prediction'],padding_error=abs(r['prediction']-r['truth']),truth=r['truth']))
 assert len(rows)==100
 for i,r in enumerate(rows):r['visual_key']=f"{i:03d}_"+r['recording']
 return rows

def geometry(j,m,start):
 a=read(P/'orientation'/j['key']/'COMPLETE.json');v=np.load(P/'orientation'/j['key']/'landmarks.npz');s=read(j['source_info'])['stream'];w,h=int(s['width']),int(s['height']);rot=next((float(d['rotation']) for d in s.get('side_data_list',[]) if 'rotation' in d),0)
 if abs(rot)%180==90:w,h=h,w
 scales=np.array([w/a['scale'][0],h/a['scale'][1]]);regions=[]
 for pts,t in zip(v['points'],v['times']):
  if not (start<=t<=start+10) or not np.isfinite(pts).all():continue
  box=m['boxes'][np.argmin(abs(m['t']-t))];x0,y0,x1,y1=box
  if x1<=x0 or y1<=y0:continue
  xy=(pts*scales-[x0,y0])/[x1-x0,y1-y0]*36;oval=np.zeros((36,36),np.uint8)
  ov=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];cv2.fillConvexPoly(oval,cv2.convexHull(np.rint(xy[ov]).astype(np.int32)),1);kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5));inside=cv2.erode(oval,kernel);outside=cv2.dilate(oval,kernel);eye=np.zeros_like(oval)
  for poly in [[33,160,158,133,153,144],[362,385,387,263,373,380],[61,40,37,0,267,270,291,321,314,17,84,91]]:cv2.fillConvexPoly(eye,cv2.convexHull(np.rint(xy[poly]).astype(np.int32)),1)
  eye=(eye>0)&(inside>0);masks=np.stack([(inside>0)&~eye,(outside-inside)>0,eye,outside==0]);assert np.all(masks.sum(0)==1);regions.append(masks)
 return np.mean(regions,axis=0) if regions else np.full((4,36,36),np.nan)

def input_for(r):
 if not r['group'].startswith('BTC'):
  d=VIS/'data'/r['key'];a=np.load(d/'clips.npz');raw=a['rgb'][0];region=np.load(d/'geometry.npz')['regions'];return raw,region,float(a['starts'][0])
 rgb,m,d=cached(r,'native')
 for w in r['windows']:
  try:x,raw=clip(rgb,m['t'],m['valid'],w['start']);return raw,geometry(r,m,w['start']),w['start']
  except ValueError:continue
 raise ValueError('NO_SUPPORTED_VISUAL_WINDOW')

STAGES=[['final_dense_2'],['final_dense_1'],['motion_conv4','apperance_att_conv2'],['motion_conv3'],['motion_conv2','apperance_att_conv1'],['motion_conv1','batch_norm']]
def run(r,inv,device='cuda',out_override=None):
 out=out_override or P/'visual'/r['visual_key'];assert not out.exists();out.mkdir(parents=True);raw,region,start=input_for(r);entry=next(e for e in inv if e['kind']=='padding' and e['fold']==r['fold']);model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));model.load_state_dict(torch.load(entry['path'],map_location='cpu',weights_only=True)['model'],strict=True);model=model.eval().to(device);x=(raw-raw.mean())/raw.std();x=torch.from_numpy(x.transpose(0,3,1,2)[None].copy()).to(device)
 with torch.no_grad():d=model(x)[0].cpu().numpy();static=model(x[:,:1].repeat(1,301,1,1,1))[0].cpu().numpy()
 assert all(v is None for v in all_decodes(static).values());frequency=decode_difference(d)/60;model.double();x=x.double().requires_grad_(True)
 def grad():
  p=power(model(x),frequency).sum();g=torch.autograd.grad(p,x)[0];return g.abs().sum((0,1,2)).detach().cpu().numpy(),g.sum((0,1,2)).detach().cpu().numpy(),float(p.detach())
 maps=[];signed=[];powers=[];g,sg,p=grad();maps.append(g);signed.append(sg);powers.append(p);occ=np.zeros((6,6))
 with torch.no_grad():
  for row in range(6):
   for col in range(6):
    z=x.detach().clone();z[:,:,:,row*6:(row+1)*6,col*6:(col+1)*6]=0;occ[row,col]=p-float(power(model(z),frequency))
 torch.manual_seed(0)
 if device=='cuda':torch.cuda.manual_seed_all(0)
 comparisons=[]
 for stage,layers in enumerate(STAGES):
  for name in layers:getattr(model,name).reset_parameters()
  g,sg,p=grad();maps.append(g);signed.append(sg);powers.append(p);rho=float(spearmanr(maps[0].ravel(),g.ravel()).statistic) if g.std()>0 and maps[0].std()>0 else None
  comparisons.append(dict(stage=stage+1,cumulative_layers=sum(STAGES[:stage+1],[]),spearman=rho,raw_L1=float(np.mean(abs(g-maps[0]))),mass_ratio=float(g.sum()/max(maps[0].sum(),1e-30)),power=p))
 area=region.mean((1,2));regional={}
 for name,heat in [('gradient',maps[0]),('occlusion_absolute',np.repeat(np.repeat(abs(occ),6,0),6,1))]:
  if np.isfinite(region).all() and heat.sum()>0:
   mass=(region*heat).sum((1,2))/heat.sum();regional[name]=dict(area=area.tolist(),mass=mass.tolist(),mass_per_area=np.divide(mass,area,out=np.zeros_like(mass),where=area>0).tolist())
 np.savez_compressed(out/'MAPS.npz',gradient_raw=maps,gradient_signed=signed,occlusion_signed=occ,mean_rgb=raw.mean(0),regions=region,difference=d,power=powers)
 save(out/'COMPLETE.json',dict(utc=now(),recording=r['recording'],group=r['group'],fold=r['fold'],start=start,fixed_target_hz=frequency,checkpoint_sha256=entry['sha256'],randomization=comparisons,regional=regional,static_pass=True,static_std=float(static.astype(float).std()),maps_sha256=sha(out/'MAPS.npz'),no_training=True));assert sha(entry['path'])==entry['sha256'];del model,x;torch.cuda.empty_cache()

def main():
 lease=lock('visual_repair.lock');assert not (P/'VISUAL_START.json').exists();rows=selection();save(P/'VISUAL_SELECTION.json',rows);save(P/'VISUAL_START.json',dict(utc=now(),pid=os.getpid(),selection_sha256=sha(P/'VISUAL_SELECTION.json'),stage='waiting_dependencies'));began=time.time()
 while not (P/'ORIGINAL_COMPLETE.json').exists() or any(not (P/'orientation'/r['key']/'COMPLETE.json').exists() for r in rows if r['group'].startswith('BTC')):
  assert time.time()-began<24*3600 and not (P/'ORIGINAL_ERROR.json').exists();assert not list(P.glob('ORIENTATION*_ERROR.json'));time.sleep(15)
 # This diagnostic yields to the complete spatial factorial, no duplicate GPU queue.
 while not (P/'SPATIAL_COMPLETE.json').exists():
  assert time.time()-began<24*3600 and not (P/'SPATIAL_ERROR.json').exists();time.sleep(15)
 while (datetime.now(timezone.utc)-datetime.fromisoformat(read(E/'A100_INVENTORY.json')['measured_utc'])).total_seconds()>3600:
  assert time.time()-began<24*3600;save(P/'VISUAL_STATUS.json',dict(utc=now(),state='waiting_fresh_cross_server_inventory'));time.sleep(15)
 gpu=lock('inference_gpu.lock');samples=setup();inv=inventory();save(P/'VISUAL_GPU_START.json',dict(pid=os.getpid(),utc=now(),free_MiB=samples))
 for i,r in enumerate(rows):
  save(P/'VISUAL_STATUS.json',dict(utc=now(),completed=i,total=100,current=r['visual_key']));run(r,inv);budget()
 save(P/'VISUAL_COMPLETE.json',dict(utc=now(),completed=100,no_training=True,POS_audit='Original signal_maps.py receives raw RGB before model zscore; no change needed',old_signal_code_sha256=sha(VIS/'signal_maps.py')))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'VISUAL_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
