import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
from vis_common import *
import numpy as np,re,gzip,cv2,traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SOURCE='https://raw.githubusercontent.com/FFmpeg/FFmpeg/n6.1.1/libavcodec/mpegutils.c'
def decode(path,out):
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_streams','-show_frames','-show_entries','frame=best_effort_timestamp_time,pict_type','-of','json',str(path)]));s=probe['streams'][0];mw=(int(s['width'])+15)//16;mh=(int(s['height'])+15)//16
 cmd=['ffmpeg','-hide_banner','-nostats','-debug','qp+mb_type','-threads','1','-i',str(path),'-map','0:v:0','-an','-threads','1','-f','null','-'];frames=[];types=[];rows=[];kind=None
 old=P/'codec'/out.parent.name/out.name/'decoder_debug.log.gz';logpath=old if old.exists() else out/'decoder_debug.log.gz'
 if not old.exists():
  proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  with gzip.open(logpath,'wb') as log:
   for chunk in iter(lambda:proc.stderr.read(65536),b''):log.write(chunk)
  assert proc.wait()==0
 with gzip.open(logpath,'rt') as log:
  for line in log:
   m=re.match(r'^\[h264 @ 0x[0-9a-f]+\] (.*)\n?$',line)
   if not m:continue
   payload=m.group(1).rstrip('\r\n');h=re.match('New frame, type: ([IPB])',payload)
   if h:
    assert kind is None,'Incomplete macroblock frame before next header'
    kind=h.group(1);rows=[];continue
   if kind is not None and len(payload)==mw*5:
    cells=[payload[i:i+5] for i in range(0,len(payload),5)]
    if all(c[:2].strip().isdigit() and c[2] in 'PAiIdDgGS><X' for c in cells):
     rows.append([(int(c[:2]),c[2] in 'dgS') for c in cells])
     if len(rows)==mh:frames.append(rows);types.append(kind);kind=None;rows=[]
 assert kind is None,'Incomplete final macroblock frame'
 truth=[f['pict_type'] for f in probe['frames']];assert types==truth,('DISPLAY_ORDER_NOT_VERIFIED',len(types),len(truth))
 a=np.asarray(frames);pts=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']]);pts-=pts[0];np.savez_compressed(out/'bitstream_maps.npz',qp=a[:,:,:,0].astype(np.uint8),skip=a[:,:,:,1].astype(bool),t=pts,pict_type=types)
 save(out/'DECODE.json',dict(command=cmd,source_code=SOURCE,frame_order_exact_ffprobe=True,frames=len(frames),shape=[mh,mw],width=int(s['width']),height=int(s['height']),input_sha256=sha(path),maps_sha256=sha(out/'bitstream_maps.npz'),skip_symbols=['d','g','S'],qp_range=[int(a[:,:,:,0].min()),int(a[:,:,:,0].max())]))
 return a,pts,s
def run(r):
 dest=P/'codec_v2'/r['recording'];dest.mkdir(parents=True);fig,axs=plt.subplots(4,4,figsize=(17,14),layout='constrained');stats=[]
 for v,variant in enumerate(['default','bf0']):
  root=E/'c5prime/BTC'/r['recording']/variant;receipt=read(root/'COMPLETE.json');assert sha(root/'video.mp4')==receipt['encoded_sha256'];out=dest/variant;out.mkdir();a,t,s=decode(root/'video.mp4',out);meta=np.load(root/'crop/meta.npz');boxes=meta['boxes'];assert len(boxes)==len(t);width=int(s['width']);height=int(s['height']);indices=[int(np.argmin(abs(t-x))) for x in [1,10,20]];ww=640;hh=round(height/width*ww/2)*2
  command=['ffmpeg','-v','error','-threads','1','-i',str(root/'video.mp4'),'-vf','select='+ '+'.join(f'eq(n\\,{i})' for i in indices)+f',scale={ww}:{hh}:flags=area','-vsync','0','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'];frames=np.frombuffer(subprocess.check_output(command),np.uint8).reshape(-1,hh,ww,3);assert len(frames)==3
  skipmean=[];qpmean=[]
  for i,box in enumerate(boxes):
   x0,y0,x1,y1=box;mx0=max(0,int(np.floor(x0/16)));my0=max(0,int(np.floor(y0/16)));mx1=min(a.shape[2],int(np.ceil(x1/16)));my1=min(a.shape[1],int(np.ceil(y1/16)));roi=a[i,my0:my1,mx0:mx1]
   qpmean.append(float(roi[:,:,0].mean()) if roi.size else np.nan);skipmean.append(float(roi[:,:,1].mean()) if roi.size else np.nan)
  for j,(i,frame) in enumerate(zip(indices,frames)):
   for k in range(2):
    ax=axs[v*2+k,j];ax.imshow(frame,extent=[0,width,height,0]);h=a[i,:,:,1 if k==0 else 0];im=ax.imshow(h,extent=[0,a.shape[2]*16,a.shape[1]*16,0],cmap='Reds' if k==0 else 'viridis',alpha=.55,vmin=0,vmax=1 if k==0 else 51);x0,y0,x1,y1=boxes[i];ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'cyan',lw=1);ax.set_xlim(max(0,x0-60),min(width,x1+60));ax.set_ylim(min(height,y1+60),max(0,y0-60));ax.set_title(f'{variant} {"skip" if k==0 else "QP"} t={t[i]:.2f}s');fig.colorbar(im,ax=ax,shrink=.6)
  axs[2*v,3].plot(t,skipmean);axs[2*v,3].set_ylim(0,1);axs[2*v,3].set_title(variant+' face-bbox skip fraction');axs[2*v+1,3].plot(t,qpmean);axs[2*v+1,3].set_ylim(0,51);axs[2*v+1,3].set_title(variant+' face-bbox mean QP')
  stats.append(dict(recording=r['recording'],variant=variant,frames=len(t),face_bbox_skip_fraction=float(np.nanmean(skipmean)),face_bbox_QP=float(np.nanmean(qpmean)),B_frames=sum(x['pict_type']=='B' for x in json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=pict_type','-of','json',str(root/'video.mp4')]))['frames'])));save(out/'FACE_TIME_SERIES.json',dict(t=t.tolist(),skip=[x if np.isfinite(x) else None for x in skipmean],qp=[x if np.isfinite(x) else None for x in qpmean]))
 fig.suptitle(r['recording']+' — actual decoder MB flags/QP; cyan = cached face bbox');fig.savefig(dest/'CODEC_MAP.png',dpi=130);plt.close(fig);save(dest/'COMPLETE.json',dict(rows=stats,utc=now()));return stats
def main():
 lease=lock('codec.lock');assert not (P/'CODEC_V2_START.json').exists();rows=sorted([r for r in manifest() if r['group'].startswith('BTC') and all((E/'c5prime/BTC'/r['recording']/v/'COMPLETE.json').exists() for v in ['default','bf0'])],key=lambda r:r['recording'])[:5];assert len(rows)==5;save(P/'CODEC_V2_START.json',dict(utc=now(),pid=os.getpid(),recordings=[r['recording'] for r in rows],budget=budget(fresh=True),source_code=SOURCE));result=[];errors=[]
 for r in rows:
  budget()
  try:result+=run(r)
  except Exception as e:errors.append(r['recording']);save(P/'codec_v2'/r['recording']/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc()))
 save(P/'CODEC_V2_COMPLETE.json',dict(utc=now(),rows=result,errors=errors))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'CODEC_V2_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
