"""Original cross-fitted residual regression; no changed penalties or grids."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from btc_spectral_math import GRID, zlog
from reader_models_fixed import pw
from reader_v4 import MODELS

FEATURES = ("log_bpp", "log_pixels", "log_fps", "is_hevc", "relative_box_speed",
            "relative_green_diff", "valid_fraction", "classical_shift", "neural_shift")

def capture_features(bit_rate, width, height, fps, codec):
    pixels = float(width) * float(height)
    if min(float(bit_rate), pixels, float(fps)) <= 0:
        raise ValueError("Expected positive original bitrate, image dimensions and fps")
    return np.array([np.log(bit_rate / (pixels * fps)), np.log(pixels), np.log(fps),
                     float(str(codec).lower() in {"hevc", "h265", "h.265", "hev1", "hvc1"})])

def trace_features(t, valid, rgb_mean, boxes):
    t = np.asarray(t, float)
    valid = np.asarray(valid, bool)
    boxes = np.asarray(boxes, float)
    green = np.asarray(rgb_mean, float)[:, 1]
    dt = np.diff(t)
    pairs = valid[:-1] & valid[1:] & (dt > 0) & (dt <= .2)
    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
    diagonal = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)
    face_ok = pairs & (diagonal[:-1] > 0) & (diagonal[1:] > 0)
    speed = np.linalg.norm(np.diff(centre, axis=0)[face_ok], axis=1)
    scale = (diagonal[:-1][face_ok] + diagonal[1:][face_ok]) / 2
    median_speed = np.nanmedian(speed / (dt[face_ok] * scale)) if face_ok.any() else np.nan
    green_ok = pairs & (green[:-1] > 0) & (green[1:] > 0)
    green_diff = abs(np.log(green[1:][green_ok] / green[:-1][green_ok]))
    return np.array([median_speed, np.nanmedian(green_diff) if len(green_diff) else np.nan,
                     np.mean(valid)])

def local_pulse_features(anchor_pr, chrom, pos, experts):
    """Accept a batch of shifted classical spectra and the frozen neural spectra."""
    anchor_pr = np.atleast_1d(np.asarray(anchor_pr, float))
    classical = (zlog(np.atleast_2d(chrom)) + zlog(np.atleast_2d(pos))) / 2
    neural = np.mean([zlog(np.atleast_2d(experts[m])) for m in MODELS], axis=0)
    kernel = -.5 * ((GRID - anchor_pr[:, None]) / 20.) ** 2
    pulse = [softmax(kernel + source, axis=1) @ GRID - anchor_pr for source in (classical, neural)]
    return np.column_stack(pulse)

def fit(method, capture, trace, pulse, residual, person):
    """Train with cross-fitted anchor errors from low-bitrate train people."""
    if method not in {"capture", "signal"}:
        raise ValueError(method)
    matrix = np.asarray(capture, float) if method == "capture" else np.column_stack((capture, trace, pulse))
    weights = pw(person)
    weights *= len(np.unique(person)) / weights.sum()
    valid = np.isfinite(matrix)
    sw = np.sum(weights[:, None] * valid, axis=0)
    mu = (weights @ np.where(valid, matrix, 0)) / np.maximum(sw, 1e-12)
    centered = np.where(valid, matrix - mu, 0)
    sd = np.sqrt((weights @ (centered ** 2)) / np.maximum(sw, 1e-12))
    enabled = (sw > 0) & (sd > 1e-6)
    scale = np.where(enabled, sd, 1.)
    x = np.column_stack([np.ones(len(matrix)), np.clip(centered / scale, -3, 3) * enabled])

    def objective(theta):
        e = x @ theta - residual
        small = np.minimum(np.abs(e), 8.)
        loss = weights @ (.5 * small ** 2 + 8 * (abs(e) - small)) + 10 * theta @ theta
        grad = x.T @ (weights * np.clip(e, -8, 8)) + 20 * theta
        return loss, grad

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    return dict(method=method, mean=mu.tolist(), scale=scale.tolist(), enabled=enabled.tolist(),
                theta=result.x.tolist(), features=list(FEATURES[:matrix.shape[1]]))

def predict(model, anchor_pr, low, beta_low, capture, trace=None, pulse=None):
    """Apply to N accepted recordings; missing trace values default to training means."""
    p = np.atleast_1d(np.asarray(anchor_pr, float)).copy()
    use = np.asarray(low, bool) & (np.asarray(beta_low) == 0)
    if not np.any(use):
        return p
    x = np.atleast_2d(capture)
    if model["method"] == "signal":
        x = np.column_stack([x, np.atleast_2d(trace), np.atleast_2d(pulse)])
    xx = np.where(np.isfinite(x[use]), x[use] - model["mean"], 0)
    standardized = np.clip(xx / model["scale"], -3, 3) * model["enabled"]
    theta = np.asarray(model["theta"])
    p[use] += np.clip(theta[0] + standardized @ theta[1:], -8, 8)
    return p
