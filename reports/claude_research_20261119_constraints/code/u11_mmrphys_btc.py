"""u11 (22 Nov, Claude; to be RUN BY CODEX - the Claude session may not execute this third-party code):
MMRPhys (Joshi & Cho, arXiv 2505.07013; repo PhysiologicAILab/MMRPhys, commit 2cedee23 cloned to thirdparty/MMRPhys, RESEARCH-ONLY
RAIL-AMS licence, untrusted third-party code) zero-shot on the BTC dev face-box crops (x10, 72x72), read with our frozen window readout,
exactly as z09 does for FactorizePhys. Only weights trained on data that is NOT BTC (iBVP, PURE, UBFC-rPPG, BP4D fold 1). CPU, no training.
Usage: $PY code/u11_mmrphys_btc.py run <name> <keys-file>  -> out/u11/x10/<name>.json ;  $PY code/u11_mmrphys_btc.py npz"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
TB = W / 'thirdparty/MMRPhys'; sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
MODELS = {'MMRPhys.ibvp': ('iBVP/iBVP_MMRPhys_SFSAM_Label_Epoch9.pth', 160), 'MMRPhys.pure': ('PURE/PURE_MMRPhys_SFSAM_Label_Epoch9.pth', 160),
          'MMRPhys.ubfc': ('UBFC-rPPG/UBFC-rPPG_MMRPhys_SFSAM_Label_Epoch9.pth', 160), 'MMRPhys.bp4d': ('BP4D/BP4D_MMRPhysLEF_BVP_RGBx180x72_SFSAM_Label_Fold1_Epoch4.pth', 180)}
if sys.argv[1] == 'run':
    import torch
    torch.set_num_threads(2); sys.path.append(str(TB))
    from neural_methods.model.MMRPhys.MMRPhysLEF import MMRPhysLEF
    from lib_btc import supported
    import y01_btc_improve as Y
    name, kf = sys.argv[2], sys.argv[3]; keys = (W / kf).read_text().strip().split(','); path, F = MODELS[name]
    md = dict(FRAME_NUM=F, MD_TYPE='SNMF_Label', MD_FSAM=True, MD_S=1, MD_R=1, MD_STEPS=5, MD_INFERENCE=False, MD_RESIDUAL=True, TASKS=['BVP'], BP_USE_RSP=False, FS=30)
    m = MMRPhysLEF(frames=F, md_config=md, in_channels=3, dropout=.1, device=torch.device('cpu'))
    sd = torch.load(TB / 'final_model_release' / path, map_location='cpu'); sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    r = m.load_state_dict(sd, strict=False); print('missing', len(r.missing_keys), r.missing_keys[:5], 'unexpected', len(r.unexpected_keys), r.unexpected_keys[:5], flush=True); m.eval()
    out = W / 'out/u11/x10' / (name + '.json'); out.parent.mkdir(parents=True, exist_ok=True); R = json.loads(out.read_text()) if out.exists() else {}
    for k in keys:
        if k in R: continue
        t0 = time.time()
        with np.load(W / 'out/x10' / (k + '.npz')) as z: t, valid, crops = z['t'].astype(float), z['valid'].astype(bool), z['crop72']
        n = (len(t) // F) * F; bvp = np.full(len(t), np.nan)
        with torch.no_grad():
            for s in range(0, n, F):
                x = torch.from_numpy(crops[s:s + F].astype(np.float32)).permute(3, 0, 1, 2)[None]; x = torch.cat([x, x[:, :, -1:]], 2)
                y = m(x)[0].reshape(-1).numpy(); bvp[s:s + F] = (y - y.mean()) / (y.std() + 1e-8)
        P = []
        for s in np.arange(0, max(0., t[n - 1] - 10 + 1e-8) if n else 0, 5.):
            if not supported(t, valid, s, s + 10) or s + 10 > t[n - 1]: continue
            ok = np.isfinite(bvp[:n]); w = np.interp(s + np.arange(300) / 30., t[:n][ok], bvp[:n][ok])
            if np.isfinite(w).all() and w.std() > 1e-12: P.append(Y.spec2(w)[0])
        R[k] = np.mean(P, 0).tolist() if P else None; out.write_text(json.dumps(R)); print(name, k, len(R), '/', len(keys), '%.1f s' % (time.time() - t0), flush=True)
else:
    (W / 'out/u11/npz').mkdir(parents=True, exist_ok=True)
    for p in sorted((W / 'out/u11/x10').glob('*.json')):
        R = json.loads(p.read_text()); keys = sorted(R); S = np.array([np.array(R[k]) if R[k] is not None else np.full(277, np.nan) for k in keys])
        np.savez_compressed(W / 'out/u11/npz' / (p.stem + '.npz'), keys=np.array(keys), S=S); print(p.stem, S.shape, int(np.isfinite(S).all(1).sum()))
