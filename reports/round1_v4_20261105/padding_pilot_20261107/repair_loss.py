"""Signed, bounded-lag correlation on the cardiac band with difference-MSE anchor."""
import torch

def cardiac_wave(difference):
    # Center before integration to remove a constant slope without roundoff.
    wave = (difference - difference.mean(-1, keepdim=True)).cumsum(-1)
    n = wave.shape[-1]
    freq = torch.fft.rfftfreq(n, 1/30., device=wave.device)
    keep = (freq >= .7) & (freq <= 3.)
    return torch.fft.irfft(torch.fft.rfft(wave, dim=-1) * keep, n=n, dim=-1)

def phase_loss(difference, target):
    prediction = cardiac_wave(difference)
    reference = cardiac_wave(target)
    correlations = []
    for lag in range(-6, 7):
        start = max(lag, 0)
        stop = prediction.shape[-1] + min(lag, 0)
        p = prediction[:, start:stop]
        y = reference[:, start-lag:stop-lag]
        p = p - p.mean(-1, keepdim=True)
        y = y - y.mean(-1, keepdim=True)
        y_power = y.square().mean(-1).clamp_min(1e-8)
        # Bound gradients for near-flat predictions on the normalized label scale.
        p_power = torch.maximum(p.square().mean(-1), .01 * y_power)
        r = (p*y).mean(-1) / (p_power*y_power).sqrt()
        correlations.append(r.clamp(-1, 1))
    r = torch.stack(correlations, dim=-1)
    shape = .5*(1-r[:,6]) + .5*(1-r.amax(-1))
    # Exact signed V3 target constrains DC, amplitude, and out-of-band residuals.
    anchor = (difference-target).square().mean(-1)
    return anchor + shape
