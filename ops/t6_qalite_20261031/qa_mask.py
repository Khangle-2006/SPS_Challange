"""Image-only16x16 decoded-block proxy. No labels, pulse model or tuning."""
import numpy as np


class QALite:
    def __init__(self):
        self.previous = None

    def step(self, image, masks):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected uint8 RGB')
        h, w = image.shape[:2]
        masks = [np.asarray(m, dtype=bool) for m in masks]
        if not masks or any(m.shape != (h, w) for m in masks):
            raise ValueError('ROI dimensions')
        previous, self.previous = self.previous, image.copy()
        if previous is not None and previous.shape != image.shape:
            raise ValueError('Video dimensions changed')
        union = np.logical_or.reduce(masks)
        baseline = np.array([image[m].mean(0) if m.sum() >= 16 else [np.nan]*3 for m in masks])
        stats = dict(gate=False, unchanged_fraction=0., noise_proxy_DN=None,
                     mean_roi_weight=1., full_block_roi_pixels=0, roi_pixels=int(union.sum()))
        if previous is None or union.sum() < 16:
            return baseline, baseline.copy(), stats
        yy, xx = np.where(union)
        # Include full gridblocks touching ROI, never resize/blend or align by motion.
        x0, y0 = (int(xx.min())//16)*16, (int(yy.min())//16)*16
        x1, y1 = min(w//16*16, (int(xx.max())//16+1)*16), min(h//16*16, (int(yy.max())//16+1)*16)
        pixel_delta = image[union].astype(np.float32)-previous[union].astype(np.float32)
        centered = pixel_delta-np.median(pixel_delta, axis=0)
        sigma = max(1., float(1.4826*np.median(np.abs(centered))))
        stats['noise_proxy_DN'] = sigma
        weights = np.ones((h, w), np.float32)
        if x1 > x0 and y1 > y0:
            a, b = image[y0:y1, x0:x1], previous[y0:y1, x0:x1]
            ny, nx = a.shape[0]//16, a.shape[1]//16
            delta = a.astype(np.float32)-b.astype(np.float32)
            blocks = delta.reshape(ny, 16, nx, 16, 3).transpose(0, 2, 1, 3, 4)
            unchanged = (blocks == 0).all(axis=(2, 3, 4))
            area = union[y0:y1, x0:x1].reshape(ny, 16, nx, 16).sum(axis=(1, 3))
            fraction = float((unchanged*area).sum()/union.sum())
            stats.update(unchanged_fraction=fraction, full_block_roi_pixels=int(area.sum()), gate=fraction > .25)
            if stats['gate']:
                q = np.sqrt(np.mean(blocks*blocks, axis=(2, 3, 4)))/sigma
                block_weights = .25+.75*np.clip(q/3, 0, 1)
                weights[y0:y1, x0:x1] = block_weights.repeat(16, axis=0).repeat(16, axis=1)
        stats['mean_roi_weight'] = float(weights[union].mean())
        weighted = []
        for mask in masks:
            if mask.sum() < 16:
                weighted.append([np.nan]*3)
            else:
                ww = weights[mask].astype(np.float64)
                weighted.append(np.sum(image[mask].astype(np.float64)*ww[:, None], axis=0)/ww.sum())
        return baseline, np.asarray(weighted), stats
