"""z08 - C5 split stream with LOSSLESS coding of the 36x36 crops (21 Nov). The experts of reader_v4 resize every face crop to 36x36
(INTER_AREA) inside open-rppg, so sending the 36x36 INTER_AREA reduction of the 72x72 box crop is exactly their input; lossless WebP
(OpenCV, quality > 100) keeps it bit-exact, so the y11 / y13 'split10' scores (raw crops) apply unchanged. Measures, per recording, the
stream at 10 fps (every k-th frame): time stamp + flag 9 B, plus
   BTC    : 6x6 block means of the masked oval crop (float32, 432 B) + box crop 36x36 (lossless WebP)
   public : masked oval crop 36x36 (lossless WebP; gives the team CNN input and the POS/CHROM trace) + box crop 36x36 (lossless WebP)
and checks that decoding returns the input bit for bit. -> logs/z08_split_lossless.json and a summary on stdout.
Usage: nice -n 19 $AV code/z08_split_lossless.py <workers>"""
import sys, json, glob
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); R = W.parent / 'round1_execution_20261110/inference_repair_20261113'
P = [cv2.IMWRITE_WEBP_QUALITY, 101]
def enc(x):
    ok, b = cv2.imencode('.webp', cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_RGB2BGR), P); d = cv2.cvtColor(cv2.imdecode(b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB); return len(b), bool(np.array_equal(d, x))
def one(a):
    f, masked = a; key = Path(f).stem
    with np.load(f) as z: t, valid, crops = z['t'].astype(float), z['valid'].astype(bool), z['crop72']
    fps = (len(t) - 1) / (t[-1] - t[0]); k = max(1, int(np.ceil(fps / 10 - 1e-3))); t, valid, crops = t[::k], valid[::k], crops[::k]
    mk = None
    if masked is not None:
        m = np.asarray(np.load(Path(masked) / 'rgb.npy', mmap_mode='r'))[::k] if Path(masked, 'rgb.npy').exists() else None
        if m is not None and len(m) >= len(t): mk = np.clip(np.rint(m[:len(t)].astype(np.float32)), 0, 255).astype(np.uint8)
    nb = np.zeros(len(t)); exact = True; nm = np.zeros(len(t))
    for i in range(len(t)):
        if not valid[i]: continue
        b36 = cv2.resize(crops[i], (36, 36), interpolation=cv2.INTER_AREA); n, e = enc(b36); nb[i] = n; exact &= e
        if mk is not None: n2, e2 = enc(mk[i]); nm[i] = n2; exact &= e2
    fr = (len(t) - 1) / (t[-1] - t[0]); sec = np.floor(t - t[0]).astype(int)
    def kbps(per): s = np.bincount(sec, weights=per) * 8 / 1000; return float(per.mean() * fr * 8 / 1000), float(s[:-1].max() if len(s) > 1 else s.max())
    out = dict(fps_sent=round(fr, 3), k=k, box36_bytes=float(nb[valid].mean()) if valid.any() else 0., lossless=exact)
    out['btc_stream_kbps'] = kbps(9 + 432 + nb)
    if mk is not None: out['masked36_bytes'] = float(nm[valid].mean()); out['public_stream_kbps'] = kbps(9 + nm + nb)
    return key, out
if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = {j['key']: j for j in json.loads((R / 'MANIFEST.json').read_text())}
    fs = [f for f in sorted(glob.glob(str(W / 'out/x10/*.npz'))) if not f.endswith('.tmp.npz')]
    args = [(f, None if Path(f).stem.startswith('BTC') else jobs[Path(f).stem]['crop']) for f in fs]
    with Pool(int(sys.argv[1])) as p: RES = dict(p.map(one, args, chunksize=2))
    (W / 'logs/z08_split_lossless.json').write_text(json.dumps(RES, indent=0))
    print('all crops decoded bit-exactly:', all(r['lossless'] for r in RES.values()))
    for g, var in (('BTC', 'btc_stream_kbps'), ('UBFC2', 'public_stream_kbps'), ('MCD', 'public_stream_kbps')):
        v = np.array([r[var] for k, r in RES.items() if k.startswith(g) and var in r])
        if len(v): print(g, len(v), var, 'mean %.0f kbps (max over recordings %.0f), highest 1-s total %.0f kbps' % (v[:, 0].mean(), v[:, 0].max(), v[:, 1].max()), flush=True)
    for g in ('UBFC2', 'MCD'):
        v = [r.get('masked36_bytes') for k, r in RES.items() if k.startswith(g)]; print(g, 'masked 36x36 crop available for', sum(x is not None for x in v), 'recordings; mean bytes', np.mean([x for x in v if x is not None]) if any(x is not None for x in v) else None)
