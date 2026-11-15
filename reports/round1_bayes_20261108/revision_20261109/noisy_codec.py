import os,sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from pipeline import pulse,psd
F=Path('C:/Users/Windows/AppData/Local/Programs/Python/Python311/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def frames():
 rng=np.random.default_rng(20261109);texture=gaussian_filter(rng.normal(size=(480,640)),12);texture=3*(texture-texture.mean())/texture.std()
 base=np.array([180.,130.,110.])+texture[:,:,None];pad=np.pad(base,((1,1),(1,1),(0,0)),mode='reflect')
 for i in range(900):
  dx,dy=rng.integers(-1,2,size=2);im=pad[1+dy:481+dy,1+dx:641+dx]+rng.normal(0,1.5,(480,640,3))
  if i%3==0:yield np.clip(np.rint(im),0,255).astype(np.uint8)
yy,xx=np.ogrid[:480,:640];mask=((xx-319.5)/160)**2+((yy-239.5)/160)**2<=1
def mean(frame):return frame[mask].mean(0)
def spectra(rgb):
 out={}
 for name,x in [('G',detrend(-rgb[:,1])),('POS',pulse(rgb,10,'POS'))]:
  f,p=psd(x,10);band=(f>=.7)&(f<=3);near=abs(f-2.5)<=.1;ac=f>0
  out[name]=dict(f=f,p=p,band_fraction=float(p[near].sum()/p[band].sum()) if p[band].sum()>1e-30 else None,ac_fraction=float(p[near].sum()/p[ac].sum()) if p[ac].sum()>1e-30 else None,peak_bpm=float(f[band][np.argmax(p[band])]*60) if p[band].max()>1e-20 else None)
 return out
def main():
 d=P/'noisy_control';d.mkdir(exist_ok=True);assert not (d/'RESULTS.json').exists(),'completed control must not rerun'
 assert not list(d.glob('*.mp4')),'preserve partial outputs, inspect before retry'
 (d/'PROTOCOL_HASH.json').write_text(json.dumps(dict(protocol_sha256=sha(P/'NOISY_CONTROL_PROTOCOL.md'),code_sha256=sha(Path(__file__)),pipeline_sha256=sha(P.parent/'pipeline.py'))))
 before=None;results=[]
 for recipe in ['old_x264','noB_x264','part2_experimental']:
  path=d/(recipe+'.mp4');cmd=[str(F),'-hide_banner','-loglevel','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','640x480','-r','10','-i','pipe:0','-an','-threads','1','-filter_threads','1']
  if recipe=='old_x264':cmd+=['-c:v','libx264','-preset','veryfast','-g','20','-keyint_min','20','-sc_threshold','0']
  elif recipe=='noB_x264':cmd+=['-c:v','libx264','-preset','veryfast','-bf','0','-g','10','-keyint_min','10','-sc_threshold','0']
  else:cmd+=['-c:v','mpeg4','-bf','0','-g','10']
  cmd+=['-pix_fmt','yuv420p','-b:v','450k','-maxrate','450k','-bufsize','900k','-movflags','+faststart',str(path)]
  (d/(recipe+'.command.json')).write_text(json.dumps(dict(command=cmd,official_MATLAB_parity=False)))
  pre=[]
  with (d/(recipe+'.encode.log')).open('wb') as log:
   proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
   try:
    for frame in frames():pre.append(mean(frame));proc.stdin.write(frame.tobytes())
    proc.stdin.close();assert proc.wait()==0
   finally:
    if proc.poll() is None:proc.kill();proc.wait()
  pre=np.array(pre)
  if before is None:before=pre
  else:assert np.array_equal(before,pre),'same synthetic input required across recipes'
  cmd=[str(F),'-v','error','-threads','1','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-threads','1','pipe:1']
  raw=subprocess.check_output(cmd);assert len(raw)==300*480*640*3
  after=np.array([mean(x) for x in np.frombuffer(raw,np.uint8).reshape(300,480,640,3)])
  a=spectra(before);b=spectra(after);np.savez_compressed(d/(recipe+'.npz'),before=before,after=after,f=a['G']['f'],G_before=a['G']['p'],G_after=b['G']['p'],POS_before=a['POS']['p'],POS_after=b['POS']['p'])
  row=dict(recipe=recipe,frames=300,duration_seconds=30,whole_file_kbps=path.stat().st_size*8/30/1000,bitrate_pass=path.stat().st_size*8/30<=500000,sha256=sha(path),official_MATLAB_parity=False)
  for stage,s in [('before',a),('after',b)]:
   for name,obj in s.items():
    for key in ['band_fraction','ac_fraction','peak_bpm']:row[stage+'_'+name+'_'+key]=obj[key]
  results.append(row);(d/'STATUS.json').write_text(json.dumps(dict(completed=len(results),total=3)))
 (d/'RESULTS.json').write_text(json.dumps(results,indent=2,allow_nan=False));print(json.dumps(results))
if __name__=='__main__':main()
