import json,sys
from pathlib import Path
from collections import Counter
import av
root,index_path=map(Path,sys.argv[1:]);idx=json.loads(index_path.read_text());rows=[]
for rec in idx['records']:
    if rec['dataset']!='MCD':continue
    manifest=json.loads((Path(idx['roots']['MCD'])/rec['recording_id']/'native/crop/manifest.json').read_text())
    meta=manifest['metadata'];assert meta['camera']=='FullHDwebcam'
    p=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')/meta['source_db_row']['video']
    with av.open(str(p)) as container:
        stream=container.streams.video[0]
        rows.append(dict(recording=rec['recording_id'],person=rec['subject_id'],camera=meta['camera'],source_codec=stream.codec_context.name,source_path=str(p),width=stream.width,height=stream.height))
out=dict(records=len(rows),people=len({r['person'] for r in rows}),codec_counts=dict(Counter(r['source_codec'] for r in rows)),all_source_mpeg4=all(r['source_codec']=='mpeg4' for r in rows),C5_codec='h264',double_compression_confirmed=all(r['source_codec']=='mpeg4' for r in rows),rows=rows)
(root/'SOURCE_CODEC_VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
