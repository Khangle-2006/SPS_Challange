"""u21 (22 Nov, Claude): does the codec remove the pulse information deterministically in the BTC low-resolution class?
Descriptive diagnostic proposed by the 22/11 literature review (Q1c). For every frame, the fraction of face macroblocks whose decoded
pixels are bit-identical to the previous frame (P_Skip with zero motion: the block carries no new information in that frame).
Classes: real low files (848x480, all 67 in dev), camera originals (resting, not 4K/HDR, random 30, seed 7, as u04) and the same
originals after the two measured re-encode chains of x12 (sim_l1 Baseline 30 fps GOP 1 s, sim_l2 High 20 fps one key frame).
Blocks: 16x16 grid in display coordinates (all coded sizes here are multiples of 16, so a 90/180 deg rotation maps the macroblock grid
onto itself); only blocks fully inside the central 60% of the per-frame face box (x01 native boxes; scaled for the 848 px chains);
comparison on the interior 8x8 pixels (4 px margin, so chroma upsampling cannot leak in from neighbouring blocks). Key frames excluded.
Score also checks whether the wrong video-level peaks of the low class sit on key-frame harmonics (u09 did this for originals only).
Dev videos only; holdout untouched. Usage: $AV code/u21_codec_static.py run <workers>  |  $AV code/u21_codec_static.py score"""
import os, sys, json, time, traceback, subprocess, pickle
ARGV = sys.argv[:]; sys.argv = ['x']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np
import x01_extract as X
import x06_c5 as C5
import x12_btcsim as SIM
import y06_reader_quality as Y
sys.argv = ARGV
W = X.W; OUT = W / 'out/u21'; TMP = W / 'out/u21_tmp'; co = X.co
INV = {v: k for k, v in json.loads((W.parent / 'w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video'].items()}
J = {j['recording']: j for j in json.loads((W.parent / 'round1_execution_20261110/inference_repair_20261113/MANIFEST.json').read_text()) if j['dataset'] == 'BTC'}
jobs = [J[INV.get(r, r)] for r in Y.rec]; keys = [j['key'] for j in jobs]
def media(j):
    import ast
    md = j['metadata'] if isinstance(j['metadata'], dict) else ast.literal_eval(j['metadata']); return md.get('media', {})
def probe(src):
    st = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,profile,width,height,pix_fmt,bit_rate', '-of', 'json', str(src)]))['streams'][0]
    o = subprocess.run(['ffprobe', '-v', 'error', '-skip_frame', 'nokey', '-select_streams', 'v:0', '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', str(src)], capture_output=True, text=True).stdout.split()
    kt = np.array([float(x) for x in o if x.replace('.', '', 1).isdigit()]); return st, kt
def key_index(t, kt):
    if not len(kt): return set([0])
    kt = kt - kt[0]; return set(np.abs(t[None, :] - kt[:, None]).argmin(1).tolist())
def stats(cur, prev, box):
    Hf, Wf = cur.shape[:2]; x0, y0, x1, y1 = [float(v) for v in box]; cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    a0, a1 = max(0., cx - .3 * w), min(float(Wf), cx + .3 * w); b0, b1 = max(0., cy - .3 * h), min(float(Hf), cy + .3 * h)
    i0, i1, j0, j1 = int(np.ceil(a0 / 16)), int(np.floor(a1 / 16)), int(np.ceil(b0 / 16)), int(np.floor(b1 / 16))
    if i1 - i0 < 1 or j1 - j0 < 1: return 0, 0, np.nan, False
    A = cur[j0 * 16:j1 * 16, i0 * 16:i1 * 16].reshape(j1 - j0, 16, i1 - i0, 16, -1)[:, 4:12, :, 4:12]
    B = prev[j0 * 16:j1 * 16, i0 * 16:i1 * 16].reshape(j1 - j0, 16, i1 - i0, 16, -1)[:, 4:12, :, 4:12]
    same = (A == B).all(axis=(1, 3, 4)); d = (cur[int(b0):int(b1), int(a0):int(a1)] != prev[int(b0):int(b1), int(a0):int(a1)]).any(2)
    return same.size, int(same.sum()), float(d.mean()), bool(not d.any())
def inner(box, Wf, Hf):
    x0, y0, x1, y1 = [float(v) for v in box]; cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    return int(max(0, cx - .3 * w)), int(min(Wf, cx + .3 * w)), int(max(0, cy - .3 * h)), int(min(Hf, cy + .3 * h))
def collect(frames, box, valid, keyset):
    prev = prev2 = None; R = []; RGB = []; SIG = []
    for t, fr in enumerate(frames):
        fr = np.array(fr, copy=True); ok = t < len(valid) and bool(valid[t])
        r = stats(fr, prev, box[t]) if (prev is not None and ok and valid[t - 1]) else (0, 0, np.nan, False)
        R.append(r + (t in keyset,)); sg = np.nan; m = np.full(3, np.nan)
        if ok:
            a0, a1, b0, b1 = inner(box[t], fr.shape[1], fr.shape[0])
            if a1 - a0 >= 8 and b1 - b0 >= 8:
                m = fr[b0:b1, a0:a1].reshape(-1, fr.shape[2]).mean(0)
                if prev2 is not None and t % 5 == 0 and valid[t - 1] and valid[t - 2]:
                    d2 = fr[b0:b1:2, a0:a1:2, 1].astype(np.float32) - 2 * prev[b0:b1:2, a0:a1:2, 1] + prev2[b0:b1:2, a0:a1:2, 1]
                    sg = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
        RGB.append(m); SIG.append(sg); prev2 = prev; prev = fr
    a = np.array(R, dtype=float); return dict(nb=a[:, 0], ni=a[:, 1], chg=a[:, 2], frz=a[:, 3], key=a[:, 4], rgb=np.array(RGB, np.float32), sig=np.array(SIG, np.float32))
def pulse_rms(rgb, t):
    from scipy import signal
    g = rgb[:, 1].astype(float); ok = np.isfinite(g)
    if ok.sum() < 100: return np.nan, np.nan
    g = np.interp(np.arange(len(g)), np.flatnonzero(ok), g[ok]); fs = (len(t) - 1) / (t[-1] - t[0])
    sos = signal.butter(3, [.7, 3.], btype='bandpass', fs=fs, output='sos'); b = signal.sosfiltfilt(sos, g - g.mean())[int(fs * 2):-int(fs * 2)]
    return float(np.sqrt(np.mean(b ** 2))), float(np.sqrt(np.mean(b ** 2)) / g.mean())
def summ(d, t):
    m = (d['nb'] >= 4) & (d['key'] == 0); r = d['ni'][m] / d['nb'][m]
    if not m.any(): return dict(frames=0)
    pr, prel = pulse_rms(d['rgb'], t)
    return dict(frames=int(m.sum()), ident=float(r.mean()), f90=float((r >= .9).mean()), fall=float((r == 1).mean()), chg=float(np.nanmean(d['chg'][m])), freeze=float(d['frz'][m].mean()), nb=float(np.median(d['nb'][m])),
                sigma_t=float(np.nanmedian(d['sig'])), band_rms=pr, band_rel=prel, green=float(np.nanmean(d['rgb'][:, 1])))
def run(arg):
    i, mode = arg; j = jobs[i]; key = keys[i]; t0 = time.time()
    names = ['native'] if mode == 'native' else ['sim_l1', 'sim_l2']
    if all((OUT / c / (key + '.npz')).exists() for c in names): return dict(key=key, mode=mode, state='skip')
    try:
        with np.load(W / 'out/x01/native' / (key + '.npz')) as z: box = z['box'].astype(float); valid = z['valid'].astype(bool); tref = z['t']
        st, kt = probe(j['source']); vid = X.BTCVideo(j['source'], tref); n0 = len(vid.t)
        info = dict(key=key, i=int(i), mode=mode, low=int(Y.low[i]), codec=st.get('codec_name'), profile=st.get('profile'), size=[vid.w, vid.h], pix_fmt=st.get('pix_fmt'), bit_rate=st.get('bit_rate'), n=n0, key_frames=len(kt), gop_s=float(np.median(np.diff(kt))) if len(kt) > 1 else float('nan'))
        if mode == 'native':
            d = collect(iter(vid), box, valid, key_index(vid.t, kt)); assert len(d['nb']) == n0, f'decoded {len(d["nb"])} of {n0}'
            (OUT / 'native').mkdir(parents=True, exist_ok=True); np.savez_compressed(OUT / 'native' / (key + '.npz'), **d); info['native'] = summ(d, vid.t)
        else:
            fps = (n0 - 1) / (vid.t[-1] - vid.t[0]); sc = 848. / max(vid.w, vid.h); Wd, H = C5.even(vid.w * sc), C5.even(vid.h * sc); sel = {}; fo = {}
            for c, cf in SIM.CH.items():
                if fps <= cf['fps'] * 1.02: sel[c] = np.arange(n0); fo[c] = fps
                else:
                    q = vid.t[0] + np.arange(int((vid.t[-1] - vid.t[0]) * cf['fps']) + 1) / cf['fps']; ii = np.searchsorted(vid.t, q).clip(1, n0 - 1); ii = np.where(q - vid.t[ii - 1] <= vid.t[ii] - q, ii - 1, ii); sel[c] = ii; fo[c] = cf['fps']
            TMP.mkdir(parents=True, exist_ok=True); path = {c: TMP / f'{key}__{c}.mp4' for c in SIM.CH}; kb = {c: SIM.CH[c]['bps'] * (Wd * H / SIM.PX) * (fo[c] / SIM.CH[c]['fps']) / 1000 for c in SIM.CH}
            enc = {c: SIM.encoder(path[c], Wd, H, fo[c], kb[c], SIM.CH[c]['args']) for c in SIM.CH}; ptr = {c: 0 for c in SIM.CH}; n = 0
            for k, rgb in enumerate(vid):
                raw = np.ascontiguousarray(co.c4_resize(np.ascontiguousarray(rgb), (Wd, H))).tobytes(); n += 1
                for c in SIM.CH:
                    while ptr[c] < len(sel[c]) and sel[c][ptr[c]] == k: enc[c].stdin.write(raw); ptr[c] += 1
            assert n == n0, f'decoded {n} of {n0}'
            sx, sy = Wd / vid.w, H / vid.h
            for c in SIM.CH:
                enc[c].stdin.close(); assert enc[c].wait() == 0, 'encoder failed'
                stc, ktc = probe(path[c]); tc = np.arange(len(sel[c])) / fo[c]; b = box[sel[c]] * np.array([sx, sy, sx, sy]); v = valid[sel[c]]
                d = collect(C5.decode(path[c], Wd, H), b, v, key_index(tc, ktc)); assert len(d['nb']) == len(sel[c]), f'{c}: decoded {len(d["nb"])} of {len(sel[c])}'
                (OUT / c).mkdir(parents=True, exist_ok=True); np.savez_compressed(OUT / c / (key + '.npz'), sel=sel[c], sx=sx, sy=sy, **d)
                info[c] = summ(d, tc); info[c].update(kbps=round(path[c].stat().st_size * 8 / (len(sel[c]) / fo[c]) / 1000, 1), gop_s=float(np.median(np.diff(ktc))) if len(ktc) > 1 else float('nan'), profile=stc.get('profile'))  # mp4 kept for u21b
        info.update(seconds=round(time.time() - t0, 1), state='done'); return info
    except Exception as e:
        return dict(key=key, mode=mode, state='error', error=repr(e), traceback=traceback.format_exc()[-1200:])
if __name__ == '__main__':
    if sys.argv[1] == 'run':
        from multiprocessing import Pool
        lowi = [int(i) for i in np.flatnonzero(Y.low == 1)]
        cand = [int(i) for i in np.flatnonzero((Y.low == 0) & ~Y.task) if not X.is_heavy(dict(jobs[i], metadata=dict(media=media(jobs[i]))))]
        orig = [cand[k] for k in np.random.default_rng(7).permutation(len(cand))[:30]]
        work = [(i, 'native') for i in lowi] + [(i, 'native') for i in orig] + [(i, 'sim') for i in orig]
        print('low', len(lowi), 'originals', len(orig), 'jobs', len(work), flush=True); log = (W / 'logs/u21.jsonl').open('a')
        with Pool(int(sys.argv[2]), maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run, work): log.write(json.dumps(r) + '\n'); log.flush(); print(r.get('key'), r.get('mode'), r.get('state'), r.get('seconds'), r.get('error', ''), flush=True)
    else:
        R = {}
        for l in (W / 'logs/u21.jsonl').open():
            r = json.loads(l)
            if r.get('state') == 'done': R[(r['key'], r['mode'])] = r
        rows = []
        for (k, m), r in R.items():
            for c in (['native'] if m == 'native' else ['sim_l1', 'sim_l2']):
                if c in r and r[c].get('frames'):
                    g = ('low ' + str(r['profile'])) if (m == 'native' and r['low']) else ('original ' + str(r['codec'])) if m == 'native' else c
                    rows.append(dict(g=g, i=r['i'], **r[c], gop=r[c].get('gop_s', r['gop_s'])))
        q = lambda a: f'{np.median(a):.3f} [{np.percentile(a, 25):.3f}; {np.percentile(a, 75):.3f}]'
        print('fraction of face macroblocks bit-identical to the previous frame (key frames excluded); median [IQR] over videos')
        for g in sorted(set(r['g'] for r in rows)):
            G = [r for r in rows if r['g'] == g]
            print(f'{g:28s} n {len(G):3d} | identical {q([r["ident"] for r in G])} | frames >=90% identical {q([r["f90"] for r in G])} | changed px {q([r["chg"] for r in G])} | blocks {np.median([r["nb"] for r in G]):.0f} | GOP s {np.nanmedian([r["gop"] for r in G]):.2f}')
            print(f'{"":28s}        temporal noise sigma_t (grey) {q([r["sigma_t"] for r in G])} | face green 0.7-3 Hz RMS (grey) {q([r["band_rms"] for r in G])} | relative {q([1e3 * r["band_rel"] for r in G])} x1e-3')
        # does skipping go with losing the pulse? (descriptive, low class only)
        Sp = pickle.load((W / 'out/y01_spectra.pkl').open('rb')); pk = np.full((5, Y.N), np.nan)
        for i, s in enumerate(Sp):
            if s['chrom']: pk[0, i] = Y.GRID[np.mean([w[0] for w in s['chrom']], 0).argmax()]; pk[1, i] = Y.GRID[np.mean([w[0] for w in s['pos']], 0).argmax()]
        for jx, n in enumerate(['FacePhys.rlap', 'ME-chunk.pure', 'ME-flow.pure']):
            with np.load(W / 'out/y05' / (n + '.npz')) as z: S = z['S']
            h = np.isfinite(S).all(1); pk[2 + jx, h] = Y.GRID[S[h].argmax(1)]
        hit = np.nanmean(np.abs(pk - Y.lab) <= 5, 0)
        L = [r for r in rows if r['g'].startswith('low')]; idv = np.array([r['ident'] for r in L]); hv = np.array([hit[r['i']] for r in L])
        from scipy.stats import spearmanr
        print(f'low class: Spearman(identical fraction, share of the 5 readers within 5 bpm of label) = {spearmanr(idv, hv)[0]:+.3f} (n {len(L)})')
        # wrong peaks on key-frame harmonics, low class
        out = []
        for r in [R[k] for k in R if k[1] == 'native' and R[k]['low']]:
            rate = 1. / r['gop_s'] if np.isfinite(r['gop_s']) and r['gop_s'] > 0 else np.nan
            lines = np.array([m * rate * 60 for m in range(1, 40) if 42 <= m * rate * 60 <= 180]) if np.isfinite(rate) else np.array([])
            wrong = [f for f in pk[:, r['i']] if np.isfinite(f) and abs(f - Y.lab[r['i']]) > 5]
            if len(lines) and wrong: out.append((np.mean([np.min(np.abs(lines - f)) <= 3 for f in wrong]), min(1, len(lines) * 6 / 138), r['gop_s']))
        if out:
            o = np.array(out); print(f'low class: wrong reader peaks within 3 bpm of a key-frame harmonic {o[:, 0].mean():.2f} (chance ~{o[:, 1].mean():.2f}); videos {len(o)}; GOP s values {np.unique(np.round(o[:, 2], 2), return_counts=True)}')
        else: print('low class: no video with a key-frame harmonic inside 42-180 bpm')
