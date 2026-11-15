import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',EGL_PLATFORM='surfaceless',MESA_LOADER_DRIVER_OVERRIDE='swrast',GALLIUM_DRIVER='llvmpipe',TF_CPP_MIN_LOG_LEVEL='3')
from common_repair import *
from spatial import *
from oval_frontend import OVAL,oval_mask
import mediapipe as mp,traceback,argparse,time
cv2.setNumThreads(1)
def source_info(j):
 d=Path(j['crop']);a=dict(np.load(d/'meta.npz'));t=a['t'] if j['dataset']=='BTC' else a['original_t']
 if j['dataset']=='BTC':s=read(j['source_info']);stream=s['stream'];vf=s['source_decode_command'][s['source_decode_command'].index('-vf')+1]
 else:stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',j['source']]))['streams'][0];vf='format=rgb24'
 w,h=int(stream['width']),int(stream['height']);rot=next((float(x['rotation']) for x in stream.get('side_data_list',[]) if 'rotation' in x),0)
 if abs(rot)%180==90:w,h=h,w
 return t,w,h,vf,rot
def one(j,out_override=None):
 out=out_override or P/'orientation'/j['key'];assert not out.exists();out.mkdir(parents=True)
 assert sha(j['source'])==j['source_sha256'];t,w,h,vf,metadata_rot=source_info(j)
 starts=[a['start'] for a in j['windows']];end=min(float(t[-1]),max(starts)+10);sample=np.unique([int(np.argmin(abs(t-q))) for q in np.linspace(max(t[0],min(starts)),end,9)])
 sw=round(w*min(1,640/max(w,h)));sh=round(h*min(1,640/max(w,h)))
 select='+'.join('eq(n\\,'+str(int(i))+')' for i in sample);filt='select='+select+','+vf+f',scale={sw}:{sh}:flags=area'
 cmd=['ffmpeg','-v','error','-threads','1','-i',j['source'],'-an','-sn','-filter_threads','1','-vf',filt,'-vsync','0','-frames:v',str(len(sample)),'-pix_fmt','rgb24','-f','rawvideo','pipe:1']
 raw=subprocess.check_output(cmd,stderr=(out/'decode.log').open('wb'));assert len(raw)==len(sample)*sw*sh*3
 images=np.frombuffer(raw,np.uint8).reshape(-1,sh,sw,3);points=[];turns=[];seeds=[];goodframes=[];mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=.5)
 for image in images:
  point=None
  for trial in range(4):
   rotated=np.rot90(image,trial).copy();det=mesh.process(rotated)
   if det.multi_face_landmarks:
    lm=det.multi_face_landmarks[0].landmark;rh,rw=rotated.shape[:2];q=np.array([(l.x*rw,l.y*rh) for l in lm]);point=turn_points(q,(-trial)%4,rw,rh);break
  if point is None:points.append(np.full((468,2),np.nan));turns.append(-1);continue
  points.append(point);turns.append(orientation(point)[0]);oval=oval_mask(image.shape[:2],point[OVAL]);pix=seed_pixels(image,point,oval)
  if len(pix):seeds.append(pix);goodframes.append((image,point))
 mesh.close();valid=[k for k in turns if k>=0];vote=np.bincount(valid,minlength=4);k=int(vote.argmax());agreement=float(vote[k]/max(len(valid),1));resolved=len(valid)>=3 and agreement>=.7
 k=k if resolved else 0
 fit=color_fit(np.concatenate(seeds)) if seeds else None;fractions=[];maskvalid=[]
 for image,point in goodframes:
  oval=oval_mask(image.shape[:2],point[OVAL]);masked,bad,frac,mv=color_mask(image,oval,fit);fractions.append(frac);maskvalid.append(mv)
 if goodframes:
  image,point=goodframes[0];oval=oval_mask(image.shape[:2],point[OVAL]);masked,bad,frac,mv=color_mask(image,oval,fit);red=image.copy();red[bad]=[255,0,255]
  panels=[image,np.rot90(image,k),red,np.rot90(masked,k)];panels=[cv2.resize(z,(240,240),interpolation=cv2.INTER_AREA) for z in panels]
  cv2.imwrite(str(out/'before_rotation_mask.png'),cv2.cvtColor(np.hstack(panels),cv2.COLOR_RGB2BGR))
 np.savez_compressed(out/'landmarks.npz',points=np.asarray(points),source_indices=sample,times=t[sample],turn_votes=turns)
 receipt=dict(utc=now(),dataset=j['dataset'],recording=j['recording'],group=j.get('metadata',{}).get('group'),key=j['key'],quarter_turn_ccw=k,degrees_ccw=k*90,resolved=resolved,agreement=agreement,valid_samples=len(valid),sample_count=len(sample),metadata_rotation=metadata_rot,rotation_after_metadata=True,source_sha256=j['source_sha256'],sample_command=cmd,scale=[sw,sh],fit=fit,non_skin_fraction_mean=float(np.mean(fractions)) if fractions else None,mask_valid_samples=sum(maskvalid),landmarks_sha256=sha(out/'landmarks.npz'))
 save(out/'COMPLETE.json',receipt)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('shard',type=int);ap.add_argument('--shards',type=int,default=2);args=ap.parse_args();lease=lock(f'orientation{args.shard}.lock');assert not (P/f'ORIENTATION{args.shard}_START.json').exists();budget(True)
 save(P/f'ORIENTATION{args.shard}_START.json',dict(utc=now(),pid=os.getpid(),code=sha(__file__)));jobs=manifest()[args.shard::args.shards];done=[]
 for j in jobs:
  save(P/f'ORIENTATION{args.shard}_STATUS.json',dict(utc=now(),completed=len(done),total=len(jobs),current=j['key']));one(j);done.append(j['key']);budget()
 save(P/f'ORIENTATION{args.shard}_COMPLETE.json',dict(utc=now(),completed=len(done),total=len(jobs)))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/f'ORIENTATION{sys.argv[1]}_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
