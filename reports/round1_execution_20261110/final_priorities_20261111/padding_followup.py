"""CPU-only OOF evaluation of final padding checkpoints; never launches training."""
import os,sys,time,json,traceback,gc
from pathlib import Path
P=Path(__file__).resolve().parent;E=P.parent
sys.path.insert(0,str(E))
import padding_queue as q
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from common import BTCData,infer_native,read,save,sha,now
from model_eval import c5_rows
from report_metrics import metric,belongs
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
from test_static_tsm import static_result
from t8_evaluate import decode_difference
import numpy as np,torch
OUT=P/'padding_evaluation';OUT.mkdir(exist_ok=True)
METHOD='padding_fixed_seed2027'

def checkpoint(f):
 return (q.ORIGINAL_OUT if f==0 else q.OUT)/f'fold{f}/seed2027/joint'

def report():
 windows=[]
 for f in range(5):
  d=OUT/f'fold{f}'
  if not (d/'COMPLETE.json').exists():continue
  for name in ['NATIVE.json','C5PRIME.json']:
   windows += [r for r in read(d/name) if r.get('dataset','BTC')=='BTC']
 video=[]
 for rid,c in sorted({(r['recording'],r['condition']) for r in windows}):
  rr=[r for r in windows if r['recording']==rid and r['condition']==c];vv=[r for r in rr if r['prediction'] is not None]
  if vv:video.append(dict(vv[0],prediction=float(np.median([r['prediction'] for r in vv])),valid_windows=len(vv),candidate_windows=len(rr),near175=float(np.mean([r['prediction']>=175 for r in vv]))))
 baseline=[r for r in read(E/'C5_BTC_VIDEO.json') if r['method'] in ['constant','calibrated']]
 tables=[];completed=[f for f in range(5) if (OUT/f'fold{f}/COMPLETE.json').exists()]
 for c in sorted({r['condition'] for r in video}):
  for f in completed+['pooled_complete' if len(completed)==5 else 'pooled_partial']:
   rr=[r for r in baseline+video if r['condition']==c and (isinstance(f,str) or r['fold']==f)]
   for group in ['all','Normal-Resting','Abnormal-Resting','Task','Walking']:
    for m in metric([r for r in rr if belongs(r,group)],['constant','calibrated',METHOD]):tables.append(dict(condition=c,fold=f,group=group,**m))
 save(OUT/'BTC_METRICS.json',dict(rows=tables,complete_folds=completed,holdout_evaluations=0,aggregation='video median -> person mean absolute error -> equal mean people',bootstrap='paired people 10000 seed0'))
 save(OUT/'BTC_VIDEO.json',video)
 save(OUT/'COVERAGE.json',[dict(condition=c,candidate_windows=len(rr),accepted_windows=sum(r['prediction'] is not None for r in rr),candidate_videos=len({r['recording'] for r in rr}),accepted_videos=len({r['recording'] for r in rr if r['prediction'] is not None})) for c in sorted({r['condition'] for r in windows}) if (rr:=[r for r in windows if r['condition']==c])])
 lines=['# Padding-fixed: BTC dev OOF','',f'Fold hoàn tất: {completed}. Chỉ khi đủ 5 fold mới là kết quả đầy đủ. Không chọn checkpoint hoặc mở holdout.','', '| Fold | Điều kiện | Nhóm | Phương pháp | Video/người | MAE [CI95] | Bias | Δ hiệu chỉnh [CI95] |','|---|---|---|---|---:|---:|---:|---:|']
 for r in tables:
  ci=r['CI95'];di=r['delta_calibrated_CI'];lines.append(f"| {r['fold']} | {r['condition']} | {r['group']} | {r['method']} | {r['videos']}/{r['people']} | {r['MAE']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {r['bias']:+.3f} | {r['delta_calibrated']:+.3f} [{di[0]:+.3f}, {di[1]:+.3f}] |")
 lines += ['', 'Mốc hiệu chỉnh lấy từ dự đoán đã lưu trên đúng giao video của mỗi điều kiện; không gán 11,37 cho mọi fold hoặc C5′. C5′ là proxy ffmpeg, chưa xác minh MATLAB. Độ phủ lưu riêng trong COVERAGE.json; cửa sổ không hợp lệ không được điền dự đoán.']
 (OUT/'BTC_PROGRESS.md').write_text('\n'.join(lines)+'\n')

def evaluate_fold(f):
 src=checkpoint(f);status=read(src/'status.json');assert status['state']=='complete' and status['completed_epochs']==30
 digest=sha(src/'final.pt');assert digest==status['checkpoint_sha256']
 if f==0:assert digest=='e0315095fda595be37e4dc3ce3b693016111cfe7da81ef924a839c5ddf32ab0c'
 dest=OUT/f'fold{f}';assert not dest.exists(),'Preserve partial evaluation, inspect before resuming';dest.mkdir()
 save(dest/'STARTED.json',dict(utc=now(),pid=os.getpid(),checkpoint_sha256=digest,code_sha256=sha(__file__),common_sha256=sha(E/'common.py'),c5_sha256=sha(E/'model_eval.py')))
 model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));state=torch.load(src/'final.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True);model.eval()
 static=static_result(model);assert static['hr'] is None,'STATIC_FAILURE';save(dest/'STATIC.json',static)
 data=BTCData(f);save(dest/'NATIVE.json',infer_native(model,data,f,METHOD,'cpu'))
 tests=[]
 for v in sorted([v for v in data.rows if v['fold']==f],key=lambda v:v['recording'])[:5]:
  starts=sorted({r['start'] for r in data.candidates(v,'native') if r['scheme']=='10s'})
  if not starts:tests.append(dict(recording=v['recording'],state='no_supported_window'));continue
  try:
   x=data.clip(v,starts[0]);x=np.repeat(x[:1],len(x),axis=0)
   with torch.inference_mode():wave=model(torch.from_numpy(x[None]))[0].numpy()
   hr=decode_difference(wave);tests.append(dict(recording=v['recording'],std=float(wave.std()),hr=hr,passed=hr is None));assert hr is None,'BTC_STATIC_FAILED'
  except ValueError as e:tests.append(dict(recording=v['recording'],state='unsupported',reason=str(e)))
 save(dest/'BTC_STATIC.json',tests)
 save(dest/'C5PRIME.json',c5_rows(model,f,'cpu',METHOD))
 save(dest/'COMPLETE.json',dict(utc=now(),checkpoint_sha256=digest,fold=f,holdout_evaluations=0));del data,model,state;gc.collect();report()

def main():
 lease=q.lock(OUT/'worker.lock');assert not (OUT/'LAUNCH.json').exists();q.budget(fresh=True)
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 save(OUT/'LAUNCH.json',dict(pid=os.getpid(),utc=now(),device='cpu',code_sha256=sha(__file__)))
 while True:
  for f in range(5):
   d=checkpoint(f)
   if not (OUT/f'fold{f}/COMPLETE.json').exists() and (d/'status.json').exists() and read(d/'status.json')['state']=='complete':
    q.budget();save(OUT/'STATUS.json',dict(state='evaluating',fold=f,pid=os.getpid(),utc=now()));evaluate_fold(f)
  completed=[f for f in range(5) if (OUT/f'fold{f}/COMPLETE.json').exists()]
  if len(completed)==5:save(OUT/'COMPLETE.json',dict(utc=now(),folds=completed));return
  if (P/'ERROR.json').exists():save(OUT/'STATUS.json',dict(state='stopped_training_error',completed=completed));return
  save(OUT/'STATUS.json',dict(state='waiting_final_checkpoints',completed=completed,pid=os.getpid(),utc=now()));time.sleep(30)
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),utc=now()));raise
