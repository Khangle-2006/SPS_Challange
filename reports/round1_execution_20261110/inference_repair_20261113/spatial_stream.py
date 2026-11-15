"""One recording through stdin/stdout, full frames streamed, corrected crops in RAM only."""
from orientation_audit_v2 import *
from crop_io import cached,source_path
from oval_crop import OvalCrop,crop_from_points
import io
sys.path.insert(0,str(O));from preprocess import Video

def frames_cv(path,count):
 cap=cv2.VideoCapture(str(path));assert cap.isOpened()
 try:
  for _ in range(count):
   good,im=cap.read();assert good;yield cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 finally:cap.release()

def prepare(j,c,out,orientation_override=None):
 out.mkdir(parents=True,exist_ok=False);old,m,d=cached(j,c);orientation_path=orientation_override or P/'orientation'/j['key']/'COMPLETE.json';a=read(orientation_path);k=a['quarter_turn_ccw'];fit=a['fit'];path,h=source_path(j,c);assert sha(path)==h
 haspoints='oval_points' in m;original_t=m.get('original_t',m['t']);count=len(original_t)
 if haspoints:frames=frames_cv(path,count);det=None
 else:source=Video(path,out/'decode.log');assert len(source.t)==count;frames=iter(source);det=OvalCrop()
 names=['identity','rotate','mask','rotate_mask'];arrays={n:[] for n in names};fraction=[];maskvalid=[];tracked=[];boxes=[];last_image=None
 try:
  for i,im in enumerate(frames):
   if haspoints:points=m['oval_points'][i];ok,small,box=crop_from_points(im,points)
   else:ok,small,box,points=det.step(im)
   tracked.append(ok);boxes.append(box);arrays['identity'].append(small)
   # Integer bounding rectangle only: no scaling or interpolation. Work on this
   # rectangle before oval masking/resizing; translation is exactly reversible.
   x0,y0,x1,y1=box
   if ok:patch=im[y0:y1,x0:x1];local=points-np.array([x0,y0],dtype=points.dtype)
   else:patch=np.zeros((36,36,3),np.uint8);local=np.full((36,2),np.nan)
   oval=oval_mask(patch.shape[:2],local)
   if fit is not None and ok:masked,bad,frac,mv=color_mask(patch,oval,fit)
   else:masked=patch;frac=0.;mv=False
   fraction.append(frac);maskvalid.append(mv)
   pt=turn_points(local,k,patch.shape[1],patch.shape[0]);arrays['rotate'].append(crop_from_points(np.rot90(patch,k),pt)[1] if k else small)
   arrays['mask'].append(crop_from_points(masked,local)[1]);arrays['rotate_mask'].append(crop_from_points(np.rot90(masked,k),pt)[1] if k else arrays['mask'][-1])
   if i==0:last_image=np.hstack([arrays[n][-1] for n in names]);cv2.imwrite(str(out/'input_four_way.png'),cv2.cvtColor(cv2.resize(last_image,(576,144),interpolation=cv2.INTER_NEAREST),cv2.COLOR_RGB2BGR))
 finally:
  if det:det.close()
 assert len(tracked)==count
 unique,inverse,counts=np.unique(original_t,return_inverse=True,return_counts=True);np.testing.assert_array_equal(unique,m['t'])
 for name in names:
  x=np.asarray(arrays[name]);merged=np.zeros((len(unique),36,36,3),np.float32);np.add.at(merged,inverse,x);merged/=counts[:,None,None,None];arrays[name]=merged.astype(old.dtype)
 error=np.abs(arrays['identity'].astype(float)-old.astype(float));parity=dict(max_abs=float(error.max()),different_values=int(np.count_nonzero(error)),boxes_exact=bool(np.array_equal(boxes,m['boxes'])),frames=count)
 save(out/'IDENTITY_PARITY.json',parity)
 assert parity['different_values']==0,'RAW_REPRODUCTION_NOT_BIT_IDENTICAL'
 if not haspoints:assert np.array_equal(tracked,m['valid']),'TRACKING_REPRODUCTION_CHANGED'
 if k==0:assert np.array_equal(arrays['rotate'],old)
 receipt=dict(utc=now(),key=j['key'],condition=c,degrees_ccw=k*90,orientation_sha256=sha(orientation_path),source_sha256=h,identity_parity=parity,mask_fraction_mean=float(np.mean(fraction)),mask_fraction_p95=float(np.quantile(fraction,.95)),mask_valid_fraction=float(np.mean(maskvalid)),no_new_tracker=haspoints,tracking_reproduction_exact=not haspoints,no_PR_preprocessing=True)
 np.savez_compressed(out/'MASK_STATS.npz',fraction=fraction,mask_valid=maskvalid);save(out/'PREPARED.json',receipt)
 result={n:arrays[n] for n in names if n!='identity'};result.update(t=m['t'],valid=m['valid'],quarter_turn_ccw=np.array(k));return result

if __name__=='__main__':
 try:
  j=next(j for j in manifest() if j['key']==sys.argv[1]);c=sys.argv[2];out=Path(sys.argv[3]);payload=prepare(j,c,out);buf=io.BytesIO();np.savez(buf,**payload);sys.stdout.buffer.write(buf.getbuffer());sys.stdout.buffer.flush()
 except Exception as e:
  save(Path(sys.argv[3])/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
