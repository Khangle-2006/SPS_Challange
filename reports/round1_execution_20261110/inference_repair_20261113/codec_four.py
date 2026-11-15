"""Fixed native face boxes, matched source frame ordinals, no new disk video cache."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from common_repair import *
from crop_io import cached,source_path
from classical_repair import functions,waveform_decode
import numpy as np,threading,io,traceback
from scipy import signal
sys.path.insert(0,str(O));from preprocess import Video
sys.path.insert(0,str(VIS));import codec_maps_v3 as qp_decoder

def select():
 result=[];alljobs=manifest()
 for ds,n in [('BTC',4),('UBFC2',3),('MCD',3)]:
  candidates=[];people=set()
  for j in sorted([j for j in alljobs if j['dataset']==ds],key=lambda j:j['recording']):
   if j['person'] in people:continue
   receipts=[read(E/'c5prime'/ds/j['recording']/v/'COMPLETE.json') for v in ['default','bf0']]
   if not all(a['state']=='complete' and a['bitrate_pass'] and a['source_span_s']>=30 for a in receipts):continue
   candidates.append(j);people.add(j['person'])
   if len(candidates)==n:break
  assert len(candidates)==n;result+=candidates
 return result

class PipeFrames:
 def __init__(self,cmd,w,h,log,payload=None):
  self.log=open(log,'wb');self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=self.log);self.w=w;self.h=h;self.writer=None
  if payload is not None:
   def write():
    try:self.proc.stdin.write(payload);self.proc.stdin.close()
    except BrokenPipeError:pass
   self.writer=threading.Thread(target=write);self.writer.start()
 def frame(self):
  b=self.proc.stdout.read(self.w*self.h*3);assert len(b)==self.w*self.h*3,'DECODE_FRAME_SIZE';return np.frombuffer(b,np.uint8).reshape(self.h,self.w,3)
 def finish(self):
  assert not self.proc.stdout.read(1),'EXCESS_DECODE_FRAME';assert self.proc.wait()==0
  if self.writer:self.writer.join()
  self.log.close()
 def close(self):
  if self.proc.poll() is None:self.proc.terminate()
  self.proc.stdout.close()
  try:self.proc.wait(timeout=10)
  except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
  if self.writer:self.writer.join(timeout=10)
  self.log.close()

def one(j):
 out=P/'codec_four'/j['key'];assert not out.exists();out.mkdir(parents=True);assert sha(j['source'])==j['source_sha256'];native=Video(j['source'],out/'source_probe.log');old,m,d=cached(j,'native');root=E/'c5prime'/j['dataset']/j['recording'];clock=np.load(root/'default/clock.npz');clock2=np.load(root/'bf0/clock.npz');np.testing.assert_array_equal(clock['source_frame_indices'],clock2['source_frame_indices']);np.testing.assert_array_equal(clock['source_t'],clock2['source_t']);idx=clock['source_frame_indices'];ts=clock['source_t'];take=(ts<=30.05);idx=idx[take];ts=ts[take];count=len(idx);w,h=native.width,native.height;assert count>=250
 # Fixed frame skip is identical to both saved lossy conditions.
 select_filter='select='+ '+'.join(f'eq(n\\,{int(i)})' for i in idx)
 native_cmd=['ffmpeg','-v','error','-threads','1','-i',j['source'],'-an','-sn','-dn','-filter_threads','1','-vf',native.vf+','+select_filter,'-vsync','0','-frames:v',str(count),'-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
 fps=float(read(root/'default/COMPLETE.json')['fps']);lossless_cmd=['ffmpeg','-v','error','-threads','1','-i',j['source'],'-an','-sn','-dn','-filter_threads','1','-vf',native.vf+','+select_filter+f',setpts=N/({fps:.12f}*TB)','-frames:v',str(count),'-r',f'{fps:.12f}','-fps_mode','cfr','-threads','1','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-fflags','+bitexact','-flags:v','+bitexact','-f','matroska','pipe:1']
 with (out/'lossless_encode.log').open('wb') as log:lossless=subprocess.check_output(lossless_cmd,stderr=log)
 lossless_sha=hashlib.sha256(lossless).hexdigest();lossless_decode=['ffmpeg','-v','error','-threads','1','-i','pipe:0','-an','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
 commands={'native_selected':native_cmd,'ffv1_encode':lossless_cmd,'ffv1_decode':lossless_decode};streams={};traces={k:[] for k in ['native','lossless10','default','bf0']};errors={k:[] for k in ['lossless10','default','bf0']};max_ffv1=0;frame_hash=hashlib.sha256()
 try:
  streams['native']=PipeFrames(native_cmd,w,h,out/'native_decode.log');streams['lossless10']=PipeFrames(lossless_decode,w,h,out/'lossless_decode.log',lossless)
  for variant in ['default','bf0']:
   path=root/variant/'video.mp4';a=read(root/variant/'COMPLETE.json');assert sha(path)==a['encoded_sha256'];cmd=['ffmpeg','-v','error','-threads','1','-i',str(path),'-frames:v',str(count),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'];commands[variant]=cmd;streams[variant]=PipeFrames(cmd,w,h,out/(variant+'_decode.log'))
  boxes=m['boxes'];ot=m.get('original_t',m['t'])
  for n,index in enumerate(idx):
   images={k:v.frame() for k,v in streams.items()};ref=images['native'];frame_hash.update(ref.tobytes());x0,y0,x1,y1=boxes[index];assert x1>x0 and y1>y0,'INVALID_FIXED_NATIVE_BOX'
   for name,image in images.items():
    roi=image[y0:y1,x0:x1];traces[name].append(roi.astype(float).mean((0,1)))
    if name!='native':
     diff=image.astype(np.int16)-ref.astype(np.int16);rd=diff[y0:y1,x0:x1];errors[name].append([float(abs(diff).mean()),float(abs(rd).mean()),*rd.mean((0,1)).tolist()])
     if name=='lossless10':max_ffv1=max(max_ffv1,int(abs(diff).max()))
  for stream in streams.values():stream.finish()
 finally:
  for stream in streams.values():stream.close()
 assert max_ffv1==0,'FFV1_NOT_RGB_BIT_EXACT'
 # Native cadence control uses original cached box RGB only for source-rate trace.
 # Re-decode native face means at all source frames up to30s, same boxes, no tracking rerun.
 cmd=['ffmpeg','-v','error','-threads','1','-i',j['source'],'-an','-sn','-dn','-filter_threads','1','-vf',native.vf,'-frames:v',str(int(idx[-1])+1),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'];commands['native_all']=cmd;stream=PipeFrames(cmd,w,h,out/'native_all_decode.log');alltrace=[]
 try:
  for index in range(int(idx[-1])+1):
   image=stream.frame();x0,y0,x1,y1=boxes[index];alltrace.append(image[y0:y1,x0:x1].astype(float).mean((0,1)) if x1>x0 and y1>y0 else [np.nan]*3)
  stream.finish()
 finally:stream.close()
 native_t=ot[:len(alltrace)];u,iv,counts=np.unique(native_t,return_inverse=True,return_counts=True);merged=np.zeros((len(u),3));np.add.at(merged,iv,alltrace);merged/=counts[:,None];native_t=u;alltrace=merged;ns=functions();rows=[]
 for condition in ['native','lossless10','default','bf0']:
  t=native_t if condition=='native' else ts;rgb=np.asarray(alltrace if condition=='native' else traces[condition])
  for start in [0.,10.,20.]:
   if start<t[0] or start+10>t[-1]:continue
   q=start+np.arange(300)/30;x=np.stack([np.interp(q,t,rgb[:,k]) for k in range(3)],axis=1)
   for method in ['POS','CHROM','G']:
    y=x[:,1] if method=='G' else ns[method.lower()](x,30)
    try:hr=waveform_decode(y,.6);reason=None
    except ValueError as e:hr=None;reason=str(e)
    rows.append(dict(dataset=j['dataset'],recording=j['recording'],condition=condition,start=start,method=method,prediction=hr,near150=hr is not None and 148<=hr<=152,reason=reason))
 qp=[]
 for condition in ['native','default','bf0']:
  path=Path(j['source']) if condition=='native' else root/condition/'video.mp4';dest=out/condition;dest.mkdir();s=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(path)]))['streams'][0]
  if s['codec_name']!='h264':
   frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=pict_type,best_effort_timestamp_time','-of','json',str(path)]))['frames'];save(dest/'FRAME_TYPES.json',frames);qp.append(dict(condition=condition,codec=s['codec_name'],status='QP_not_exposed_by_verified_H264_parser',frame_types_available=True));continue
  arr,pts,s=qp_decoder.decode(path,dest);mean=arr[:,:,:,0].mean((1,2));sel=pts<=30.05;qt=pts[sel];qmean=mean[sel];fs_q=float(1/np.median(np.diff(np.unique(qt))));grid=np.arange(qt[0],qt[-1],1/fs_q);values=np.interp(grid,qt,qmean);freq,power=signal.periodogram(values,fs=fs_q,detrend='linear');peak=float(freq[1:][np.argmax(power[1:])]) if len(freq)>1 else None;np.savez_compressed(dest/'QP_TIME.npz',t=pts,qp_mean=mean,spectrum_f=freq,spectrum_power=power);qp.append(dict(condition=condition,codec=s['codec_name'],status='verified_display_order_QP',dominant_QP_hz=peak,mean_QP=float(qmean.mean()),spectrum_grid_fps=fs_q))
 qp.append(dict(condition='lossless10',codec='ffv1',status='QP_not_applicable_lossless'))
 np.savez_compressed(out/'COLOR_TIME.npz',source_indices=idx,source_t=ts,native_t=native_t,native_all_rgb=alltrace,**{'rgb_'+k:np.asarray(v) for k,v in traces.items()},**{'error_'+k:np.asarray(v) for k,v in errors.items()});save(out/'WINDOWS.json',rows);save(out/'COMPLETE.json',dict(utc=now(),source_sha256=j['source_sha256'],ffv1_sha256=lossless_sha,ffv1_bytes=len(lossless),FFV1_RGB_bit_exact=True,selected_native_RGB_sha256=frame_hash.hexdigest(),frames=count,actual_selected_fps=fps,commands=commands,QP=qp,fixed_native_boxes_sha256=hashlib.sha256(boxes.tobytes()).hexdigest(),trace_ROI='same native bounding box for all four conditions; no detector rerun; not identical to prior masked-oval POS',mean_errors={k:np.mean(v,axis=0).tolist() for k,v in errors.items()},no_disk_video_cache=True,official_MATLAB_parity=False))

def main():
 lease=lock('codec_four.lock');assert not (P/'CODEC_FOUR_START.json').exists();budget(True);rows=select();save(P/'CODEC_SELECTION.json',rows);save(P/'CODEC_FOUR_START.json',dict(utc=now(),pid=os.getpid(),selection_sha256=sha(P/'CODEC_SELECTION.json'),code=sha(__file__)))
 for i,j in enumerate(rows):
  save(P/'CODEC_FOUR_STATUS.json',dict(utc=now(),completed=i,total=10,current=j['key']))
  receipt=P/'codec_four'/j['key']/'COMPLETE.json'
  if receipt.exists():assert read(receipt)['source_sha256']==j['source_sha256'] and read(receipt)['FFV1_RGB_bit_exact']
  else:one(j)
  budget()
 save(P/'CODEC_FOUR_COMPLETE.json',dict(utc=now(),completed=10))
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'CODEC_FOUR_ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
