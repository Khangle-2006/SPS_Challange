"""Narrow user-authorized cleanup: completed T9 native H264 intermediates only."""
import hashlib,json,subprocess,time
from pathlib import Path
BASE=Path('/data/khangle/facevitals');ROOT=BASE/'reports/round1_v4_20261105';T9=BASE/'t9_fps_codec_v2_20261031'
status=json.loads((T9/'status.json').read_text());assert status['state']=='complete' and status['completed']==35 and not status['errors']
source={r['path']:r for r in json.loads((ROOT/'SOURCE_RAW_UBFC_MANIFEST.json').read_text())}
spec=json.loads((ROOT/'code/folds_frozen.json').read_text());ids=sorted({s for f in spec['folds'] for s in f['ubfc_val_subjects']})
report=ROOT/'STORAGE_CLEANUP.json';assert not report.exists()
rows=[]
for sid in ids:
 p=T9/sid/'native_h264500.mp4';c=json.loads((p.parent/'complete.json').read_text());e=json.loads((p.parent/'encoding.json').read_text())
 assert p.resolve().parent==T9/sid and not p.is_symlink() and p.is_file()
 digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==c['h26430_video_sha256']
 src=source['DATASET_2/'+sid+'/vid.avi'];assert src['sha256']==c['source_sha256']==e['source_sha256']
 assert (BASE/'datasets/UBFC-rPPG'/src['path']).is_file()
 probe=json.loads(subprocess.check_output([str(BASE/'tools/t5_ffmpeg/ffmpeg-7.0.2-amd64-static/ffprobe'),'-v','error','-select_streams','v:0','-show_entries','stream=codec_name','-of','json',str(p)],text=True))
 assert probe['streams'][0]['codec_name']=='h264'
 rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest,source_path=str(BASE/'datasets/UBFC-rPPG'/src['path']),source_sha256=src['sha256'],regeneration_command=e['command'],retained_encoding_receipt=str(p.parent/'encoding.json'),retained_completion_receipt=str(p.parent/'complete.json')))
receipt=dict(state='validated_before_cleanup',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=rows,total_bytes=sum(r['bytes'] for r in rows),reason='Explicit V4 task0 authorized deletion of regenerable T9 H264 intermediates. Raw/FFV1/crops/checkpoints/reports/traces retained; diagnostic1.3 used existing traces.',E0_T7_remaining_video_candidates_on_P100=0,duplicate_waveforms_removed=0,duplicate_waveforms_reason='Retained for hybrid reevaluation; CSV summaries cannot reconstruct waveforms.')
report.write_text(json.dumps(receipt,indent=2))
for row in rows:Path(row['path']).unlink()
receipt.update(state='complete',all_listed_absent=all(not Path(r['path']).exists() for r in rows));report.write_text(json.dumps(receipt,indent=2))
print('FREED_BYTES',receipt['total_bytes'],'FILES',len(rows),flush=True)
