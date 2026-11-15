"""Lossless storage pilot; never removes or replaces source video."""
import sys, json, hashlib, time, argparse, os
from pathlib import Path
from fractions import Fraction
sys.path.insert(0, '/data/khangle/facevitals/tools/ubfc_audit_deps')
import av
import numpy as np
import cv2

ROOT = Path('/data/khangle/facevitals')
ap=argparse.ArgumentParser()
ap.add_argument('--subject',default='subject1')
ap.add_argument('--replace-verified',action='store_true')
args=ap.parse_args()
assert args.subject.startswith('subject') and args.subject[7:].isdigit()
SRC = ROOT / 'datasets/UBFC-rPPG/DATASET_2' / args.subject / 'vid.avi'
assert SRC.resolve()==SRC and SRC.is_file()
OUT = ROOT / ('reports/ubfc2_ffv1_20261030/'+args.subject if args.replace_verified else 'reports/storage_pilot_20261030')
OUT.mkdir(parents=True, exist_ok=True)
DST = OUT / (args.subject+'.ffv1.avi')
assert not DST.exists(), 'Preserving existing pilot output'
start = time.time()
def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

expected = []
with av.open(str(SRC)) as source, av.open(str(DST), 'w', format='avi') as dest:
    ss = source.streams.video[0]
    info = {'codec': ss.codec_context.name, 'pixel_format': ss.codec_context.format.name,
            'width': ss.width, 'height': ss.height, 'time_base': str(ss.time_base),
            'average_rate': str(ss.average_rate), 'declared_frames': ss.frames}
    ds = dest.add_stream('ffv1', rate=ss.average_rate)
    ds.width, ds.height, ds.pix_fmt = ss.width, ss.height, 'bgr0'
    ds.time_base = ss.time_base
    ds.codec_context.time_base = ss.time_base
    ds.codec_context.thread_count = 2
    ds.options = {'level': '3', 'coder': '1', 'context': '1', 'slicecrc': '1'}
    for i, frame in enumerate(source.decode(video=0)):
        rgb = frame.to_ndarray(format='rgb24')
        assert frame.pts is not None
        expected.append({'index': i, 'sha256_rgb24': hashlib.sha256(rgb.tobytes()).hexdigest(),
                         'time': str(frame.pts * frame.time_base)})
        converted = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        converted.pts, converted.time_base = frame.pts, frame.time_base
        for packet in ds.encode(converted): dest.mux(packet)
    for packet in ds.encode(): dest.mux(packet)
assert expected
with av.open(str(DST)) as encoded:
    actual_count = 0
    for i, frame in enumerate(encoded.decode(video=0)):
        assert i < len(expected), 'Extra frame'
        assert hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest() == expected[i]['sha256_rgb24'], f'Pixel mismatch {i}'
        assert frame.pts * frame.time_base == Fraction(expected[i]['time']), f'PTS mismatch {i}'
        actual_count += 1
assert actual_count == len(expected)
# The training frontend uses OpenCV, so also verify that decoder's RGB output.
for path in (SRC, DST):
    cap = cv2.VideoCapture(str(path)); n = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        assert n < len(expected)
        assert hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == expected[n]['sha256_rgb24'], f'OpenCV mismatch {path.name} {n}'
        n += 1
    cap.release()
    assert n == len(expected)
(OUT/'frame_checksums.json').write_text(json.dumps(expected))
result = {'state': 'verified', 'source': str(SRC), 'ffv1': str(DST), 'source_info': info,
          'frames': len(expected), 'pixel_exact_all_frames': True, 'pts_exact_all_frames': True,
          'opencv_pixel_exact_all_frames': True, 'source_bytes': SRC.stat().st_size,
          'ffv1_bytes': DST.stat().st_size, 'source_sha256': sha_file(SRC), 'ffv1_sha256': sha_file(DST),
          'source_preserved': True, 'wall_seconds': time.time()-start}
result['saved_fraction'] = 1-result['ffv1_bytes']/result['source_bytes']
(OUT/'ffv1_pilot.json').write_text(json.dumps(result, indent=2))
if args.replace_verified:
    assert result['saved_fraction']>0, 'No saving: preserve source'
    # Audit record is durable before the atomic replacement on this same filesystem.
    with (OUT/'ffv1_pilot.json').open('rb') as f: os.fsync(f.fileno())
    with DST.open('rb') as f: os.fsync(f.fileno())
    assert sha_file(SRC)==result['source_sha256'], 'Source changed during verification'
    os.replace(DST,SRC)
    assert sha_file(SRC)==result['ffv1_sha256']
    result.update(state='replaced_verified',source_preserved=False,active_path=str(SRC),
                  note='Original byte stream replaced only after RGB, PTS and OpenCV equality. Original SHA retained; physiological labels unchanged.')
    (SRC.parent/'vid.avi.storage.json').write_text(json.dumps(result,indent=2))
    (OUT/'ffv1_pilot.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result), flush=True)
