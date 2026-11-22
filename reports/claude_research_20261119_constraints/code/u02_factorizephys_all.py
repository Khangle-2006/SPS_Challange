"""z09 - PILOT for Codex (30 BTC originals at rest): FactorizePhys (Joshi et al., NeurIPS 2024; rPPG-Toolbox release weights trained on
iBVP, PURE, SCAMPS, UBFC-rPPG; 'Raw' 72x72 input, 160-frame chunks, the model does diff + instance norm itself) zero-shot on our crops,
read with our frozen window readout. Question: is any of them at FacePhys level (peak within 5 bpm of the label) on these videos?
Third-party code is imported from thirdparty/rPPG-Toolbox (Responsible AI License; untrusted); weights loaded on CPU, no training.
Usage: $PY code/z09_factorizephys_pilot.py run <weights-name> <crop-set: x10 | x15/<variant>> <keys-file>   -> out/u02/<crop>/<weights>.json
       $PY code/z09_factorizephys_pilot.py score <keys-file>       (all z09 outputs vs FacePhys.rlap on x10 crops, same videos)"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
TB = W / 'thirdparty/rPPG-Toolbox'; sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
if sys.argv[1] == 'run':
    import torch
    torch.set_num_threads(2); sys.path.append(str(TB))
    from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
    from lib_btc import supported
    import y01_btc_improve as Y
    name, cropset, kf = sys.argv[2], sys.argv[3], sys.argv[4]; keys = (W / kf).read_text().strip().split(',')
    md = dict(FRAME_NUM=160, MD_TYPE='NMF', MD_FSAM=False, MD_TRANSFORM='T_KAB', MD_R=1, MD_S=1, MD_STEPS=4, MD_INFERENCE=True, MD_RESIDUAL=True)
    m = FactorizePhys(frames=160, md_config=md, in_channels=3, dropout=.1, device=torch.device('cpu'))
    sd = torch.load(TB / 'final_model_release' / (name + '.pth'), map_location='cpu'); sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    r = m.load_state_dict(sd, strict=False); print('missing', len(r.missing_keys), 'unexpected', len(r.unexpected_keys), flush=True); m.eval()
    out = W / 'out/u02' / cropset.replace('/', '_') / (name + '.json'); out.parent.mkdir(parents=True, exist_ok=True); R = json.loads(out.read_text()) if out.exists() else {}
    for k in keys:
        if k in R: continue
        p = W / 'out' / cropset / (k + '.npz'); t0 = time.time()
        with np.load(p) as z: t, valid, crops = z['t'].astype(float), z['valid'].astype(bool), z['crop72']
        n = (len(t) // 160) * 160; bvp = np.full(len(t), np.nan)
        with torch.no_grad():
            for s in range(0, n, 160):
                x = torch.from_numpy(crops[s:s + 160].astype(np.float32)).permute(3, 0, 1, 2)[None]; x = torch.cat([x, x[:, :, -1:]], 2)
                y = m(x)[0].reshape(-1).numpy(); bvp[s:s + 160] = (y - y.mean()) / (y.std() + 1e-8)
        P = []
        for s in np.arange(0, max(0., t[n - 1] - 10 + 1e-8) if n else 0, 5.):
            if not supported(t, valid, s, s + 10) or s + 10 > t[n - 1]: continue
            ok = np.isfinite(bvp[:n]); w = np.interp(s + np.arange(300) / 30., t[:n][ok], bvp[:n][ok])
            if np.isfinite(w).all() and w.std() > 1e-12: P.append(Y.spec2(w)[0])
        R[k] = np.mean(P, 0).tolist() if P else None; out.write_text(json.dumps(R)); print(name, cropset, k, len(R), '/', len(keys), '%.1f s' % (time.time() - t0), flush=True)
else:
    A = sys.argv[:]; sys.argv = ['x']
    import y06_reader_quality as Q
    import y05_openrppg as Y5
    from lib_btc import GRID
    keys = (W / A[2]).read_text().strip().split(','); _, allk = Y5.btc_keys(); idx = {k: i for i, k in enumerate(allk)}
    with np.load(W / 'out/y05/FacePhys.rlap.npz') as z: S0 = z['S']
    rows = {'FacePhys.rlap (x10)': {k: S0[idx[k]] for k in keys}}
    for p in sorted((W / 'out/u02').glob('*/*.json')): rows[f'{p.stem} ({p.parent.name})'] = {k: np.array(v) for k, v in json.loads(p.read_text()).items() if v is not None}
    common = sorted(set.intersection(*[set(r) for r in rows.values()])); print('videos in common', len(common))
    for nm, r in rows.items():
        e = np.array([abs(GRID[r[k].argmax()] - Q.lab[idx[k]]) for k in common]); print(f'   {nm:45s} peak within 5 bpm {np.mean(e <= 5):.3f} | median |peak - label| {np.median(e):5.1f} | mean {e.mean():5.1f}')
