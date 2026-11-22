"""u21b (22 Nov, Claude): macroblock type and QP inside the face, read from the H.264 decoder itself (ffmpeg -debug qp+mb_type),
for every H.264 file of u21: the real low class, the H.264 camera originals of dev (not 4K), and the u21 sim chains (kept mp4).
Face = central 60% of the per-frame x01 box (display coords) mapped to coded coords (rotation side data), MBs fully inside.
Zero-coefficient threshold for a uniform luma change d inside a 4x4 inter block (H.264 forward transform, MF(0,0), inter deadzone 1/6):
|d| < (5/6) * 2^(15+floor(QP/6)) / (16 * MF)  ~ 0.13 * 2^(QP/6) grey levels. Descriptive; dev only."""
import os, sys, json, re, subprocess, time, traceback
ARGV = sys.argv[:]; sys.argv = ['x']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import u21_codec_static as U
import y06_reader_quality as Y
sys.argv = ARGV
W = U.W; OUT = W / 'out/u21b'; MF = [13107, 11916, 10082, 9362, 8192, 7282]
TOK = re.compile(r'^(\d{1,2})(\S)')
def thr(qp): qp = int(qp); return (5 / 6) * 2 ** (15 + qp // 6) / (16 * MF[qp % 6])
def maps(src, mbw, mbh):
    p = subprocess.Popen(['ffmpeg', '-hide_banner', '-nostats', '-threads', '1', '-debug', 'qp+mb_type', '-i', str(src), '-map', '0:v:0', '-an', '-f', 'null', '-'], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors='replace')
    F = []; cur = None
    for line in p.stderr:
        if 'New frame, type:' in line:
            if cur is not None and len(cur[1]) == mbh: F.append(cur)
            cur = (line.strip()[-1], [])
        elif cur is not None and len(cur[1]) < mbh:
            tk = line.split()[3:]
            if len(tk) == mbw:
                m = [TOK.match(x) for x in tk]
                if all(m): cur[1].append([(int(g.group(1)), g.group(2)) for g in m])
    if cur is not None and len(cur[1]) == mbh: F.append(cur)
    p.wait(); return F
def to_coded(b, rot, Wc, Hc):
    x0, y0, x1, y1 = b
    if rot == 0: return x0, y0, x1, y1
    if abs(rot + 90) < 1 or abs(rot - 270) < 1: return y0, Hc - x1, y1, Hc - x0
    if abs(abs(rot) - 180) < 1: return Wc - x1, Hc - y1, Wc - x0, Hc - y0
    if abs(rot - 90) < 1 or abs(rot + 270) < 1: return Wc - y1, x0, Wc - y0, x1
    raise ValueError(rot)
def face_mbs(b):
    x0, y0, x1, y1 = b; cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    return int(np.ceil((cx - .3 * w) / 16)), int(np.floor((cx + .3 * w) / 16)), int(np.ceil((cy - .3 * h) / 16)), int(np.floor((cy + .3 * h) / 16))
def analyse(src, boxes, valid, rot, tag, key, i):
    st = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,profile', '-of', 'json', str(src)]))['streams'][0]
    Wc, Hc = int(st['width']), int(st['height']); mbw, mbh = (Wc + 15) // 16, (Hc + 15) // 16; F = maps(src, mbw, mbh)
    n = min(len(F), len(boxes)); acc = {k: [] for k in ('P_skip', 'P_intra', 'P_inter', 'P_qp', 'I_qp', 'all_skip')}
    for k in range(n):
        if not valid[k]: continue
        ty, rows = F[k]; i0, i1, j0, j1 = face_mbs(to_coded(boxes[k], rot, Wc, Hc)); i0, j0 = max(i0, 0), max(j0, 0); i1, j1 = min(i1, mbw), min(j1, mbh)
        if i1 - i0 < 2 or j1 - j0 < 2: continue
        cells = [rows[r][c] for r in range(j0, j1) for c in range(i0, i1)]; t = np.array([c[1] for c in cells]); q = np.array([c[0] for c in cells])
        if ty == 'P':
            acc['P_skip'].append(np.mean(t == 'S')); acc['P_intra'].append(np.mean(np.isin(t, ['I', 'i', 'A']))); acc['P_inter'].append(np.mean(~np.isin(t, ['S', 'I', 'i', 'A'])))
            coded = t != 'S'; acc['P_qp'].append(q[coded].mean() if coded.any() else np.nan); acc['all_skip'].append(np.mean([c[1] == 'S' for r in rows for c in r]))
        elif ty == 'I': acc['I_qp'].append(q.mean())
    s = {k: float(np.nanmean(v)) if len(v) else float('nan') for k, v in acc.items()}
    s.update(key=key, i=int(i), tag=tag, profile=st.get('profile'), size=[Wc, Hc], rot=rot, frames_parsed=len(F), frames_box=len(boxes), P_frames=len(acc['P_skip']), thr_P=thr(round(s['P_qp'])) if np.isfinite(s['P_qp']) else float('nan'))
    return s
def job(arg):
    tag, i = arg; key = U.keys[i]; src = U.jobs[i]['source']
    try:
        with np.load(W / 'out/x01/native' / (key + '.npz')) as z: box = z['box'].astype(float); valid = z['valid'].astype(bool)
        if tag in ('sim_l1', 'sim_l2'):
            with np.load(W / 'out/u21' / tag / (key + '.npz')) as z: sel, sx, sy = z['sel'], float(z['sx']), float(z['sy'])
            return analyse(U.TMP / f'{key}__{tag}.mp4', box[sel] * np.array([sx, sy, sx, sy]), valid[sel], 0, tag, key, i)
        st = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', src]))['streams'][0]
        rot = next((float(s['rotation']) for s in st.get('side_data_list', []) if 'rotation' in s), 0)
        return analyse(src, box, valid, rot, tag, key, i)
    except Exception as e:
        return dict(key=key, i=int(i), tag=tag, state='error', error=repr(e), tb=traceback.format_exc()[-800:])
def codec(i):
    st = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', U.jobs[i]['source']]))['streams'][0]
    return st['codec_name'], int(st['width']) * int(st['height'])
if __name__ == '__main__':
    if sys.argv[1] == 'check':
        # rotation mapping check: for each rotation seen in H.264 files, the display frame must equal the coded frame turned the way
        # to_coded assumes, and a display rectangle mapped by to_coded must hold the same pixels
        seen = set()
        for i in range(Y.N):
            src = U.jobs[i]['source']; st = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', src]))['streams'][0]
            rot = next((float(s['rotation']) for s in st.get('side_data_list', []) if 'rotation' in s), 0)
            if st['codec_name'] != 'h264' or rot == 0 or rot in seen or int(st['width']) > 3000: continue
            seen.add(rot); Wc, Hc = int(st['width']), int(st['height'])
            g = lambda extra: np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error'] + extra + ['-i', src, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'gray', '-']), np.uint8)
            a = g(['-noautorotate']).reshape(Hc, Wc); b = g([]).reshape(-1, Hc if abs(rot) % 180 == 90 else Wc)
            k = {-90.0: -1, 270.0: -1, 90.0: 1, -270.0: 1, 180.0: 2, -180.0: 2}[rot]
            x0, y0, x1, y1 = 100, 200, 300, 400; c = [int(v) for v in to_coded((x0, y0, x1, y1), rot, Wc, Hc)]
            print('rot', rot, 'coded', a.shape, 'display', b.shape, '| frame equal:', np.array_equal(np.rot90(a, k), b), '| box equal:', np.array_equal(np.rot90(a[c[1]:c[3], c[0]:c[2]], k), b[y0:y1, x0:x1]))
    elif sys.argv[1] == 'run':
        from multiprocessing import Pool
        work = [('low', int(i)) for i in np.flatnonzero(Y.low == 1)]
        cod = {int(i): codec(int(i)) for i in np.flatnonzero(Y.low == 0)}
        work += [('orig_h264', i) for i, (c, px) in cod.items() if c == 'h264' and px < 3e6]
        work += [(c, int(i)) for c in ('sim_l1', 'sim_l2') for i in range(Y.N) if (W / 'out/u21' / c / (U.keys[i] + '.npz')).exists() and (U.TMP / f'{U.keys[i]}__{c}.mp4').exists()]
        print('jobs', len(work), flush=True); OUT.mkdir(parents=True, exist_ok=True); log = (W / 'logs/u21b.jsonl').open('w')
        with Pool(int(sys.argv[2]), maxtasksperchild=4) as pool:
            for r in pool.imap_unordered(job, work): log.write(json.dumps(r) + '\n'); log.flush()
        print('done', flush=True)
    else:
        R = [json.loads(l) for l in (W / 'logs/u21b.jsonl').open()]; E = [r for r in R if r.get('state') == 'error']
        print('rows', len(R), 'errors', len(E), [e['error'][:120] for e in E[:3]])
        q = lambda a: f'{np.nanmedian(a):6.3f} [{np.nanpercentile(a, 25):6.3f}; {np.nanpercentile(a, 75):6.3f}]'
        groups = [('low', 'High'), ('low', 'Baseline'), ('orig_h264', None), ('sim_l1', None), ('sim_l2', None)]
        print('face macroblocks in P frames (median [IQR] over videos)')
        for tag, prof in groups:
            G = [r for r in R if r.get('tag') == tag and 'P_skip' in r and (prof is None or str(r['profile']).startswith(prof)) and r['P_frames'] > 10]
            if not G: continue
            print(f'{tag + (" " + prof if prof else ""):16s} n {len(G):3d} | skip {q([r["P_skip"] for r in G])} | inter-coded {q([r["P_inter"] for r in G])} | intra {q([r["P_intra"] for r in G])} | QP coded {q([r["P_qp"] for r in G])} | QP I {q([r["I_qp"] for r in G])} | zero-coef threshold (grey) {q([r["thr_P"] for r in G])} | whole-frame skip {q([r["all_skip"] for r in G])}')
