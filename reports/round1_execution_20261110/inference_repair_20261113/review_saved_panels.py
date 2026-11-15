"""Local contact sheets for inspecting saved figures; no inference or new data."""
import json
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
destination = P / 'review_20261115'
destination.mkdir(exist_ok=True)
selection = json.loads((P / 'VISUAL_SELECTION.json').read_text())
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
created = []
for start in range(0, 100, 10):
    rows = selection[start:start+10]
    files = [P / 'rendered/figures' / (r['visual_key'] + '_sensitivity.png') for r in rows]
    if not all(f.exists() for f in files):
        continue
    try:
        for f in files:
            with Image.open(f) as im:
                im.verify()
    except (OSError, SyntaxError):
        continue  # A download may still be in progress.
    target = destination / f'panels_{start:03d}_{start+9:03d}.png'
    if not target.exists():
        sheet = Image.new('RGB', (1690, 1110), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (row, f) in enumerate(zip(rows, files)):
            x, y = (i % 2)*845, (i // 2)*222
            draw.text((x+8, y+3), row['visual_key'], fill='black', font=font)
            with Image.open(f) as im:
                sheet.paste(im.convert('RGB').resize((845, 195), Image.Resampling.LANCZOS), (x, y+25))
        sheet.save(target)
    created.append(str(target))
print(json.dumps({'sheets': created, 'count': len(created)}))
