"""Pixel-level constraint simulations with the semantics of the MATLAB functions named in the official document
(2027 SP Cup FaceVitals, section 2.2). Every function takes/returns uint8 RGB frames [H,W,3] and is applied
frame by frame in memory, BEFORE the landmark front end - no derivative video has to be written (except C5).

C1  im2double * gain ; imnoise(I,'poisson') ; imnoise(I,'gaussian',0,var)
C4  imresize(I,[rows cols],'bicubic')  (MATLAB default: antialiasing ON when shrinking)
C6  imfilter(I, edge-emphasising kernel) restricted to the face region
C7  insertShape(I,'FilledPolygon', lower-face polygon aligned to landmarks)
C3  two recordings composited into one frame (target + bystander)
Notes on fidelity are in each docstring. Requires numpy, Pillow, scipy (cv2 only for polygon fill)."""
import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------------------------- C1
def c1_attenuate(rgb, gain):
    """J = gain * im2double(I), written back as uint8 (round half away from zero like im2uint8)."""
    return np.clip(np.floor(rgb.astype(np.float64) * gain + .5), 0, 255).astype(np.uint8)

def c1_poisson(rgb, rng):
    """imnoise(I,'poisson') for uint8 input: each pixel value is used directly as the mean of a Poisson variable
    (no scaling). On a double image MATLAB scales by 1e12 and the noise is negligible - so apply this on uint8."""
    return np.clip(rng.poisson(rgb.astype(np.float64)), 0, 255).astype(np.uint8)

def c1_gaussian(rgb, var, rng, mean=0.):
    """imnoise(I,'gaussian',mean,var): mean/variance are specified for intensities on [0,1]; result clipped to [0,1]."""
    x = rgb.astype(np.float64) / 255. + mean + rng.normal(0., np.sqrt(var), rgb.shape)
    return np.clip(np.floor(np.clip(x, 0, 1) * 255. + .5), 0, 255).astype(np.uint8)

def c1(rgb, rng, gain=1., poisson=False, gauss_var=0.):
    out = c1_attenuate(rgb, gain) if gain != 1. else rgb
    if poisson: out = c1_poisson(out, rng)
    if gauss_var > 0: out = c1_gaussian(out, gauss_var, rng)
    return out

# ---------------------------------------------------------------------------------------------- C4
def c4_resize(rgb, size):
    """imresize(I,[size[1] size[0]],'bicubic'). Pillow's BICUBIC (a=-0.5, kernel support widened when shrinking)
    matches MATLAB's default antialiased bicubic closely; cv2.INTER_CUBIC does NOT antialias and must not be used here.
    size = (width, height)."""
    return np.asarray(Image.fromarray(rgb).resize(tuple(int(v) for v in size), Image.BICUBIC))

def c4_size_keep_aspect(w, h, target=(640, 480)):
    """largest size with the source aspect ratio that fits inside target (either orientation), even dimensions."""
    tw, th = target if w >= h else target[::-1]
    s = min(tw / w, th / h, 1.)
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)

# ---------------------------------------------------------------------------------------------- C6
def unsharp_kernel(alpha=.2):
    """fspecial('unsharp',alpha): 3x3 edge-emphasising (unsharp contrast enhancement) kernel; sums to 1."""
    a = float(alpha)
    return np.array([[-a, a - 1, -a], [a - 1, a + 5, a - 1], [-a, a - 1, -a]]) / (a + 1)

def c6_wrinkle(rgb, face_mask=None, alpha=.2, passes=1):
    """Supplementary synthetic stress test of the document: imfilter(I, edge-emphasising kernel,'replicate') applied
    `passes` times inside the face region only (face_mask bool [H,W]; None = whole frame). The kernel has unit DC
    gain, so it exaggerates micro-texture/shadow edges without changing the local mean, except where it saturates."""
    k = unsharp_kernel(alpha); x = rgb.astype(np.float64)
    for _ in range(passes):
        x = np.clip(np.stack([ndimage.correlate(x[..., c], k, mode='nearest') for c in range(3)], -1), 0, 255)
    out = np.floor(x + .5).astype(np.uint8)
    if face_mask is not None: out = np.where(face_mask[..., None], out, rgb)
    return out

# ---------------------------------------------------------------------------------------------- C7
OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
MASK_TOP = {'standard': [346, 195, 117],      # over the mid nose bridge, below the eyes: ~56% of the oval on the canonical face
            'high': [345, 6, 116],            # up to the bridge between the eyes: ~60%
            'low': [4]}                       # worn low, nose tip exposed: ~51%
MASK_RGB = {'surgical_blue': (167, 199, 231), 'white': (235, 235, 235), 'black': (25, 25, 25)}

def c7_polygon(landmarks_px, fit='standard'):
    """landmarks_px [468,2] in pixels (MediaPipe FaceMesh order) -> lower-face polygon [K,2]: jaw contour + upper edge."""
    return np.asarray(landmarks_px, float)[JAW + MASK_TOP[fit]]

def polygon_mask(shape_hw, poly):
    import cv2
    m = np.zeros(shape_hw, np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1); return m.astype(bool)

def c7_mask(rgb, landmarks_px, colour='surgical_blue', fit='standard'):
    """insertShape(I,'FilledPolygon',poly,'Color',c,'Opacity',1). Landmarks come from the CLEAN frame, as in the
    official recipe; the pipeline under test must then run its own detector on the returned (masked) frame.
    Returns (frame, occluded_fraction_of_oval)."""
    lm = np.asarray(landmarks_px, float); pm = polygon_mask(rgb.shape[:2], c7_polygon(lm, fit)); om = polygon_mask(rgb.shape[:2], lm[OVAL])
    out = rgb.copy(); out[pm] = np.array(MASK_RGB[colour] if isinstance(colour, str) else colour, np.uint8)
    return out, float((pm & om).sum() / max(om.sum(), 1))

# ---------------------------------------------------------------------------------------------- C3
def c3_composite(target, bystander, layout='side', scale=.6, offset=(0, 0)):
    """Put a second recording into the target's frame.
    layout 'side': frames side by side (bystander resized to the target's height) -> returns (frame, x_offset_of_target).
    layout 'pip' : bystander shrunk by `scale` and pasted at `offset` (x,y; clipped to the frame), i.e. a face that can
                   be moved across the scene frame by frame to cross the target -> returns (frame, 0)."""
    H, W = target.shape[:2]
    if layout == 'side':
        h2, w2 = bystander.shape[:2]; b = c4_resize(bystander, (max(2, int(round(w2 * H / h2))), H))
        return np.concatenate([b, target], 1), b.shape[1]
    h2, w2 = bystander.shape[:2]; b = c4_resize(bystander, (max(2, int(w2 * scale)), max(2, int(h2 * scale))))
    out = target.copy(); x, y = int(offset[0]), int(offset[1]); bh, bw = b.shape[:2]
    xs, ys = max(0, x), max(0, y); xe, ye = min(W, x + bw), min(H, y + bh)
    if xe > xs and ye > ys: out[ys:ye, xs:xe] = b[ys - y:ye - y, xs - x:xe - x]
    return out, 0
