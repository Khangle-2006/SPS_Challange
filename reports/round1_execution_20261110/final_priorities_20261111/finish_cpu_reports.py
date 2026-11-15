"""Bounded final reporting stage for these two existing CPU workers only."""
import os,sys,time,json,traceback
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
P=Path(__file__).resolve().parent
import padding_queue as q
os.environ['CUDA_VISIBLE_DEVICES']=''
def coverage():
 root=P/'classical_screen';rows=[r for f in (root/'predictions').glob('*/*/*.json') for r in q.p.read(f)['rows']];out=[]
 for ds,c,m in sorted({(r['dataset'],r['condition'],r['method']) for r in rows}):
  rr=[r for r in rows if (r['dataset'],r['condition'],r['method'])==(ds,c,m)];valid=[r for r in rr if r['prediction'] is not None]
  out.append(dict(dataset=ds,condition=c,method=m,candidate_videos=len({r['recording'] for r in rr}),accepted_videos=len({r['recording'] for r in valid}),candidate_windows=len(rr),accepted_windows=len(valid),reasons=dict(Counter(r['reason'] for r in rr if r['prediction'] is None))))
 q.save(root/'COVERAGE.json',out)
 report=root/'REPORT.md';text=report.read_text()
 if '## Độ phủ theo phương pháp' not in text:
  text+='\n\n## Độ phủ theo phương pháp\n\n| Dataset | Điều kiện | Phương pháp | Video đạt/tổng | Cửa sổ đạt/tổng |\n|---|---|---|---:|---:|\n'
  for r in out:text+=f"|{r['dataset']}|{r['condition']}|{r['method']}|{r['accepted_videos']}/{r['candidate_videos']}|{r['accepted_windows']}/{r['candidate_windows']}|\n"
  text+='\nMẫu số trên là các cache/encode đã hoàn tất và đạt điều kiện được đưa vào worker, không phải toàn bộ video gốc: C5′ quá bitrate và thiếu cache vẫn nằm trong receipt nguồn. Bảng MAE dùng giao video; COVERAGE.json giữ lý do từng lỗi. Chỉ dùng trace oval36 có sẵn; chưa phải tất cả ROI khuôn mặt.\n'
  report.write_text(text)
 q.save(root/'COVERAGE_COMPLETE.json',dict(utc=q.now(),rows=len(rows),report_sha256=q.p.sha(report)))
def main():
 lease=q.lock(P/'cpu_finalizer.lock');assert not (P/'CPU_FINALIZER_LAUNCH.json').exists()
 q.save(P/'CPU_FINALIZER_LAUNCH.json',dict(utc=q.now(),pid=os.getpid(),code_sha256=q.p.sha(__file__)))
 end=datetime.now(timezone.utc)+timedelta(hours=12);done=set()
 while datetime.now(timezone.utc)<end:
  for name in ['classical_screen/ERROR_V2.json','codec_exact/ERROR_V2.json']:
   if (P/name).exists():raise RuntimeError('Existing CPU worker failed; inspect '+name)
  if 'classical' not in done and (P/'classical_screen/COMPLETE.json').exists() and (P/'classical_screen/REPORT.md').exists():coverage();done.add('classical')
  if 'codec' not in done and (P/'codec_exact/REAL_COMPLETE.json').exists():
   import report_codec_exact as report
   report.main();done.add('codec')
  q.save(P/'CPU_FINALIZER_STATUS.json',dict(utc=q.now(),completed=sorted(done),pid=os.getpid()))
  if len(done)==2:q.save(P/'CPU_REPORTS_COMPLETE.json',dict(utc=q.now(),holdout_evaluations=0));return
  time.sleep(30)
 raise RuntimeError('12hour reporting bound reached; inspect workers, do not restart blindly')
if __name__=='__main__':
 try:main()
 except Exception as e:q.save(P/'CPU_FINALIZER_ERROR.json',dict(utc=q.now(),error=str(e),traceback=traceback.format_exc()));raise
