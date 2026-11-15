"""Boundary checks for destructive-to-cohort acceptance decisions and binary offsets."""
import struct,tempfile
from pathlib import Path
from datetime import datetime,timedelta
from mcd_header_gate import avi_header,metadata_gate

def chunk(tag,payload):
    return tag+struct.pack('<I',len(payload))+payload+(b'\0' if len(payload)%2 else b'')

with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);avih=bytearray(56);strh=bytearray(56)
    struct.pack_into('<I',avih,16,1234)
    strh[:4]=b'vids';struct.pack_into('<I',strh,20,1)
    struct.pack_into('<I',strh,24,30);struct.pack_into('<I',strh,32,1234)
    contents=b'AVI '+chunk(b'LIST',b'hdrl'+chunk(b'avih',avih)+chunk(b'LIST',b'strl'+chunk(b'strh',strh)))
    p=d/'example.avi';p.write_bytes(b'RIFF'+struct.pack('<I',len(contents))+contents)
    h=avi_header(p)
    assert h['avih'][0]['dwTotalFrames']==1234 and h['strh'][0]['dwLength']==1234
    assert h['riff_declared_total_bytes']==p.stat().st_size
    p.write_bytes(b'RIFF'+struct.pack('<I',0xffffffff)+contents)
    assert avi_header(p)['riff_declared_total_bytes'] is None
    meta=d/'meta.txt'
    def write(deltas):
        t=datetime(2023,1,1);lines=['','1 '+t.isoformat()]
        for i,dt in enumerate(deltas,2):
            t+=timedelta(microseconds=dt);lines.append(str(i)+' '+t.isoformat())
        meta.write_text('\n'.join(lines))
    write([32000]*100+[48000])
    g=metadata_gate(meta,12)
    assert g['excess_rows_before_final_drop']==90 and g['accepted_operational_prefix']
    assert not metadata_gate(meta,11)['accepted_operational_prefix']
    write([32000]*100+[48001])
    assert metadata_gate(meta,102)['gap_count']==1
    assert metadata_gate(meta,103)['rejection_reasons']==['metadata_gap_gt1.5_median','metadata_shorter_than_decoded_frames']
    write([32000]*100+[-1000])
    assert 'nonmonotonic_or_duplicate_timestamp' in metadata_gate(meta,102)['rejection_reasons']
print('PASS: AVI offsets/sentinel; exact1.5x boundary; 90-row boundary; metadata shortages; nonmonotonic clock')
