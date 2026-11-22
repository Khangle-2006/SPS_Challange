"""Container/codec metadata of the 218 BTC dev videos (headers only, via ffprobe).
Output: out/dev_video_meta.csv keyed by the export pseudonym (video###). Holdout files are never opened."""
import csv, json, subprocess, sys
from pathlib import Path
ROOT = Path('/mnt/samsung_t5/khangle/facevitals')
W = ROOT / 'reports/claude_research_20261118'
mp = json.loads((ROOT / 'reports/w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video']
files = {p.stem: p for p in (ROOT / 'datasets/FaceVitals_BTC_20261107').rglob('*') if p.is_file() and p.suffix.lower() in ('.mov', '.mp4')}
rows = []
for name, vid in sorted(mp.items(), key=lambda kv: kv[1]):
    p = files.get(name)
    if p is None:
        rows.append(dict(recording=vid, found=0)); continue
    j = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_format', '-of', 'json', str(p)],
                                  capture_output=True, text=True, timeout=60).stdout or '{}')
    s = (j.get('streams') or [{}])[0]; f = j.get('format', {})
    rot = ''
    for sd in s.get('side_data_list', []) or []:
        if 'rotation' in sd: rot = sd['rotation']
    def fr(x):
        try:
            a, b = x.split('/'); return round(float(a) / float(b), 4)
        except Exception: return ''
    rows.append(dict(recording=vid, found=1, ext=p.suffix.lower(), group=p.parts[-3], codec=s.get('codec_name'), profile=s.get('profile'),
                     pix_fmt=s.get('pix_fmt'), width=s.get('width'), height=s.get('height'), rotation=rot, r_fps=fr(s.get('r_frame_rate', '')),
                     avg_fps=fr(s.get('avg_frame_rate', '')), nb_frames=s.get('nb_frames'), duration=f.get('duration'), size=f.get('size'),
                     bit_rate=s.get('bit_rate') or f.get('bit_rate'), color_transfer=s.get('color_transfer'), color_primaries=s.get('color_primaries'),
                     color_space=s.get('color_space'), color_range=s.get('color_range'), has_b_frames=s.get('has_b_frames'), level=s.get('level'),
                     encoder=(f.get('tags') or {}).get('encoder') or (f.get('tags') or {}).get('com.apple.quicktime.software') or '',
                     make=(f.get('tags') or {}).get('com.apple.quicktime.make') or (f.get('tags') or {}).get('com.android.manufacturer') or '',
                     model=(f.get('tags') or {}).get('com.apple.quicktime.model') or (f.get('tags') or {}).get('com.android.model') or ''))
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with (W / 'out/dev_video_meta.csv').open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(len(rows), 'rows;', sum(r['found'] for r in rows), 'found')
