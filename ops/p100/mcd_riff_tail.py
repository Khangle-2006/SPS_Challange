"""Walk AVI chunks; distinguish sentinel RIFF length from incomplete payload."""
import json,struct
from pathlib import Path
def inspect(path):
    size=path.stat().st_size;chunks=[];truncated=[];sentinels=[]
    with path.open('rb') as f:
        def walk(start,end,depth=0):
            pos=start
            while pos+8<=end:
                f.seek(pos);tag,n=struct.unpack('<4sI',f.read(8));payload=pos+8;stop=payload+n
                item={'pos':pos,'tag':tag.decode('latin1'),'declared_payload':n,'available':end-payload,'depth':depth}
                if tag in (b'RIFF',b'LIST'):
                    f.seek(payload);item['kind']=f.read(4).decode('latin1')
                    if n==0xffffffff:sentinels.append(item)
                    if depth<5:walk(payload+4,min(stop,end),depth+1)
                else:
                    chunks.append(item)
                    if stop>end:truncated.append(item)
                if stop>end:break
                pos=stop+(n%2)
        walk(0,size)
    media=[c for c in chunks if c['tag'].endswith(('dc','db'))]
    return {'path':str(path),'file_bytes':size,'is_multiple_256KiB':size%(256*1024)==0,
        'media_chunk_count':len(media),'last_media_chunks':media[-3:], 'truncated_payloads':truncated,
        'sentinel_container_lengths':sentinels,'last_chunks':chunks[-3:]}
if __name__=='__main__':
    root=Path('/mnt/samsung_t5/khangle/facevitals/datasets/MCD-rPPG')
    out=root.parent.parent/'mapping_audit_20261030'
    rows=[inspect(p) for p in sorted((root/'video').glob('1020_*.avi'))]
    (out/'subject1020_riff.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows),flush=True)
