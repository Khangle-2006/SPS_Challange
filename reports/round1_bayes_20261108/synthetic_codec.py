"""D5 controlled codec probes. MPEG4 Part2 is experimental, NOT MATLAB VideoWriter.
No input/reference-dependent recipe choice. Keep encoded outputs and complete commands.
"""
from common import *
import numpy as np,fcntl,traceback
sys.path.insert(0,str(OLD));from preprocess import Video
from pipeline import pulse,psd,peak_info
from codec_audit import stream_probe,analyze_frames
def mean_roi(frame,box):
 x0,y0,x1,y1=box;patch=frame[y0:y1,x0:x1];yy,xx=np.ogrid[:len(patch),:patch.shape[1]];mask=((xx-(patch.shape[1]-1)/2)/max(patch.shape[1]/2,1))**2+((yy-(len(patch)-1)/2)/max(len(patch)/2,1))**2<=1
 return patch[mask].mean(0)
def frames_face(base,box,amp):
 x0,y0,x1,y1=box;yy,xx=np.ogrid[:y1-y0,:x1-x0];mask=((xx-(x1-x0-1)/2)/((x1-x0)/2))**2+((yy-(y1-y0-1)/2)/((y1-y0)/2))**2<=1
 for i in range(300):
  frame=base.copy();patch=frame[y0:y1,x0:x1].astype(float);mod=np.sin(2*np.pi*1.2*i/10)*amp*np.array([.25,1.,.1]);patch[mask]*=1+mod;frame[y0:y1,x0:x1]=np.clip(np.rint(patch),0,255).astype(np.uint8);yield frame
def encode(frames,w,h,path,recipe):
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','10','-i','pipe:0','-an','-threads','1','-filter_threads','1']
 if recipe=='old_x264':cmd+=['-c:v','libx264','-preset','veryfast','-g','20','-keyint_min','20','-sc_threshold','0']
 elif recipe=='noB_x264':cmd+=['-c:v','libx264','-preset','veryfast','-bf','0','-g','10','-keyint_min','10','-sc_threshold','0']
 else:cmd+=['-c:v','mpeg4','-bf','0','-g','10']
 cmd+=['-pix_fmt','yuv420p','-b:v','450k','-maxrate','450k','-bufsize','900k','-movflags','+faststart',str(path)]
 means=[];save(path.with_suffix('.command.json'),dict(command=cmd,sampling='10Hz samples of synthetic30Hz input, indices0,3,6...',modulation='green amplitude/DC; red=.25green,blue=.1green',official_MATLAB_parity=False))
 with path.with_suffix('.encode.log').open('wb') as log:
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
  try:
   for frame in frames:p.stdin.write(frame.tobytes())
   p.stdin.close();assert p.wait()==0,'encode_failed'
  finally:
   if p.poll() is None:p.kill();p.wait()
def main():
 lock=(OUT/'synthetic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prereg();save(OUT/'synthetic_PREFLIGHT.json',guard())
 d=OUT/'synthetic';d.mkdir(exist_ok=True);vs=dev_manifest();v=next(v for v in vs if v['recording']=='S_N1_Resting');source=Video(v['path'],d/'source_decode.log')
 gen=iter(source);base=next(gen).copy();gen.close();h,w=base.shape[:2]
 with np.load(OLD/'dev'/v['drive_id']/'native/meta.npz') as a:
  good=np.where(a['valid'])[0];box=np.rint(a['boxes'][good[0]]).astype(int)
 box[0::2]=np.clip(box[0::2],0,w);box[1::2]=np.clip(box[1::2],0,h);assert box[2]>box[0] and box[3]>box[1]
 np.savez_compressed(d/'fixed_face_input.npz',rgb=base,box=box)
 save(d/'INPUT.json',dict(source_recording=v['recording'],source_sha256=v['sha256'],frame=0,roi_from_frame=int(good[0]),frequency_hz=1.2,amps_DC=[.0025,.005,.01,.02],gray_resolution=[640,480],face_resolution=[w,h]))
 index=read(ROOT/'t8_oval_20261031/training_index.json');subjects=sorted([r['recording_id'] for r in index['records'] if r['dataset']=='UBFC2']);subject=subjects[0]
 ubpath=Path('/data_pool/khangle/facevitals/datasets/UBFC-rPPG/DATASET_2')/subject/'vid.avi';ub=Video(ubpath,d/'ubfc_decode.log')
 target=np.arange(300)/10;right=np.minimum(np.searchsorted(ub.t,target),len(ub.t)-1);left=np.maximum(right-1,0);selected=np.where(abs(ub.t[left]-target)<=abs(ub.t[right]-target),left,right)
 assert len(np.unique(selected))==300 and ub.t[selected[-1]]>=29.8
 def real_frames():
  wanted=set(selected.tolist());g=iter(ub)
  try:
   for i,fr in enumerate(g):
    if i in wanted:yield fr
    if i>=selected[-1]:break
  finally:g.close()
 save(d/'UBFC_INPUT.json',dict(subject=subject,path=str(ubpath),source_sha256=sha(ubpath),selected_indices=selected.tolist(),selected_PTS=ub.t[selected].tolist(),ROI='fixed central ellipse 25%-75%width,10%-80%height; controlled comparison, not full tracked evaluation'))
 inputs=[('gray',0.)]+[(f'face_{a}',a) for a in [.0025,.005,.01,.02]]+[('UBFC_real',0.)]
 result=[]
 for name,amp in inputs:
  for recipe in ['old_x264','noB_x264','part2_experimental']:
   folder=d/(name+'_'+recipe);folder.mkdir(exist_ok=True);path=folder/'video.mp4'
   if (folder/'RESULT.json').exists():result.append(read(folder/'RESULT.json'));continue
   if path.exists():raise RuntimeError('Partial synthetic output; inspect before retry')
   guard(False)
   if name=='gray':img=np.full((480,640,3),128,np.uint8);ww,hh=640,480;bb=np.array([0,0,640,480]);factory=lambda: (img for _ in range(300))
   elif name=='UBFC_real':ww,hh=ub.width,ub.height;bb=np.array([int(ww*.25),int(hh*.1),int(ww*.75),int(hh*.8)]);factory=real_frames
   else:ww,hh=w,h;bb=box;factory=lambda:frames_face(base,box,amp)
   before=np.array([mean_roi(f,bb) for f in factory()]);encode(factory(),ww,hh,path,recipe)
   video=Video(path,folder/'decode.log');after=np.array([mean_roi(f,bb) for f in video]);assert len(after)==300
   z=stream_probe(path,True);ff=z['frames'];gop=analyze_frames(ff);packet=np.array([float(x.get('pkt_size',0)) for x in ff]);wave=pulse(after,10.,'POS');f,p=psd(wave,10);freq,pkp=psd(packet,10);band=(f>=.7)&(f<=3);bin25=int(np.argmin(abs(f-2.5)));med=float(np.median(p[band]));power=float(p[bin25]);ratio=power/max(med,1e-30)
   t=np.arange(300)/10;design=np.stack([np.sin(2*np.pi*1.2*t),np.cos(2*np.pi*1.2*t),np.ones(len(t))],axis=1)
   ac=lambda x:float(np.linalg.norm(np.linalg.lstsq(design,x,rcond=None)[0][:2]))
   prewave=pulse(before,10.,'POS');a0=ac(prewave);a1=ac(wave);kbps=path.stat().st_size*8/30/1000
   r=dict(input=name,recipe=recipe,frames=300,whole_file_kbps=kbps,bitrate_pass=kbps<=500,POS_static=bool(np.std(wave)<1e-10),power_2_5hz=power,median_band_power=med,ratio_2_5_to_median=ratio,G4_trigger=bool(name=='gray' and recipe=='old_x264' and power>1e-20 and ratio>10),input_POS_amp1_2=a0,output_POS_amp1_2=a1,recovery_ratio=a1/a0 if a0>1e-12 else None,green_amp_DC=ac(after[:,1])/float(after[:,1].mean()),POS_peaks=peak_info(f,p),gop=gop,sha256=sha(path))
   csvout(folder/'frames.csv',[dict(frame=i,type=x.get('pict_type'),pkt_size=x.get('pkt_size'),pts=x.get('best_effort_timestamp_time')) for i,x in enumerate(ff)])
   csvout(folder/'PSD.csv',[dict(hz=float(a),POS_power=float(b),packet_power=float(c)) for a,b,c in zip(f,p,pkp)]);np.savez_compressed(folder/'traces.npz',input=before,decoded=after,POS=wave)
   save(folder/'RESULT.json',r);result.append(r);save(OUT/'synthetic_STATUS.json',dict(completed=len(result),total=18,pid=os.getpid(),utc=now()))
 save(OUT/'SYNTHETIC_RESULTS.json',result);save(OUT/'G4.json',dict(triggered=any(r['G4_trigger'] for r in result),synthetic_cases=15,flatfield_cases=3,real_UBFC_cases=3,scope='gray-flat only gate; absence does not exonerate natural-image codecs',official_MATLAB_parity=False,utc=now()))
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'synthetic_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
