"""Small shared pieces for the 19 Nov constraint run (no heavy imports)."""
import numpy as np
TAU_C, RHO = .08, 2.0; LUM = np.array([.299, .587, .114])
def detect_rows(Bm, Fm):
    """Label-free occlusion detector on the 6x6 grid of the 36x36 crop (rows 0..5 = forehead..chin).
    Bm[6,6,3] time-mean block means (zeros outside the oval included), Fm[6,6] time-mean in-oval fraction.
    A row is flagged when >=50% of its usable blocks differ from the forehead/brow reference in chromaticity (>TAU_C) or luminance
    (factor RHO); an occlusion is declared only for a contiguous run of flagged rows reaching the chin row, never above row 2.
    Thresholds were set on 19 Nov from unmasked BTC statistics (c07_mask.py), before any masked video existed."""
    col = Bm / np.maximum(Fm[..., None], 1e-6); use = Fm >= .5
    ch = col[..., :2] / np.maximum(col.sum(-1, keepdims=True), 1e-6); lum = col @ LUM
    refm = use[:2] & (Fm[:2] >= .9)
    if refm.sum() < 2: return ()
    ref_c = np.median(ch[:2][refm], 0); ref_l = np.median(lum[:2][refm])
    bad = (np.linalg.norm(ch - ref_c, axis=-1) > TAU_C) | (lum > RHO * ref_l) | (lum < ref_l / RHO)
    rowbad = [bool(bad[i][use[i]].mean() >= .5) if use[i].any() else False for i in range(6)]; rows = []
    for i in range(5, 1, -1):
        if rowbad[i]: rows.append(i)
        else: break
    return tuple(sorted(rows))
def upper_trace(blk, rows):
    keep = [i for i in range(6) if i not in rows]; return blk[:, keep].reshape(len(blk), -1, 3).mean(1)
