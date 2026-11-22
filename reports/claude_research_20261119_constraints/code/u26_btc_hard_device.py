"""u26 (22 Nov, Claude): is the hard half of the resting camera originals a capture-device effect? Descriptive cross-tab of the
u18 groups (easy = >=4/5 readers agree, hard = the rest) by stream type (codec, size, pixel format, rotation, HDR), and the
u21/u21b codec measurements by group. Dev only, no model fitting."""
import sys, json, subprocess, collections
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); sys.path.insert(0, str(W / 'code')); sys.argv = ['x']
import y06_reader_quality as Y
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
jobs = [J[INV.get(r, r)] for r in Y.rec]; keys = [j['key'] for j in jobs]
U = np.load(W / 'out/u18_btc_beats.npz', allow_pickle=True); grp, sub, v4 = U['grp'], U['sub'], U['v4']
rest = sub & ~Y.task & (Y.low == 0); err = np.abs(v4 - Y.lab)
def st(i):
    s = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', jobs[i]['source']]))['streams'][0]
    rot = next((float(d['rotation']) for d in s.get('side_data_list', []) if 'rotation' in d), 0)
    return f"{s['codec_name']} {s['width']}x{s['height']} {s.get('pix_fmt')} rot{int(rot)} {s.get('color_transfer')} {round(int(s.get('bit_rate', 0)) / 1e6, 1)}Mb/s"
T = collections.defaultdict(lambda: collections.Counter()); E = collections.defaultdict(list)
ab = np.array([k.split('_')[2].startswith('AB') for k in keys])
for i in np.flatnonzero(rest):
    s = st(i); typ = ' '.join(s.split()[:5]); T[typ][grp[i]] += 1; E[typ].append(err[i])
print('resting camera originals in the reader subset:', int(rest.sum()))
for typ, c in sorted(T.items(), key=lambda x: -sum(x[1].values())):
    n = sum(c.values()); print(f'{typ:55s} n {n:3d} | easy {c["easy"]:3d} hard {c["hard"]:3d} | hard share {c["hard"] / n:.2f} | v4 MAE {np.mean(E[typ]):5.2f}')
print('patients among resting originals: easy', int((rest & (grp == 'easy') & ab).sum()), 'of', int((rest & (grp == 'easy')).sum()), '| hard', int((rest & (grp == 'hard') & ab).sum()), 'of', int((rest & (grp == 'hard')).sum()))
R = {}
for l in (W / 'logs/u21.jsonl').open():
    r = json.loads(l)
    if r.get('state') == 'done' and r['mode'] == 'native' and not r['low']: R[r['i']] = r['native']
Q = {}
for l in (W / 'logs/u21b.jsonl').open():
    r = json.loads(l)
    if r.get('tag') == 'orig_h264' and 'P_qp' in r: Q[r['i']] = r
for g in ('easy', 'hard'):
    ii = [i for i in R if grp[i] == g and rest[i]]; qq = [i for i in Q if grp[i] == g and rest[i]]
    f = lambda k: np.median([R[i][k] for i in ii]) if ii else float('nan')
    print(f'{g}: u21 sample n {len(ii)} | sigma_t {f("sigma_t"):.2f} | band RMS {f("band_rms"):.3f} grey | changed px {f("chg"):.3f} || H.264 originals n {len(qq)} | QP {np.median([Q[i]["P_qp"] for i in qq]) if qq else float("nan"):.1f}')
