"""Original probability normalization and median helper used by calibration."""
import numpy as np
from scipy.special import logsumexp
from reader_models_fixed import GRID

def median_prob(q):
    return GRID[np.minimum(np.sum(q.cumsum(axis=1)<.5,axis=1),len(GRID)-1)]

def posterior(logits):
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
