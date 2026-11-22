"""Per-video CNN evidence on BTC dev from the SAVED waveforms of inference_repair_20261113 (no new inference): for each 10 s window the
frozen readout spectrum of cumsum(difference) is computed per checkpoint, averaged over the checkpoints of a family (padding-fixed, V3)
and over the windows of the video. Output: out/cnn_evidence.npz with arrays [video, grid] keyed by the export pseudonym order of VIDEOS.json."""
import json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261118/code')
from lib_btc import *
R = ROOT / 'reports/round1_execution_20261110/inference_repair_20261113'
mp = json.loads((ROOT / 'reports/w2_btc_20261116/PRIVATE_EXPORT_MAPPING.json').read_text())['video']; inv = {v: k for k, v in mp.items()}
V = videos(); out = {'pad': np.full((len(V), len(GRID)), np.nan), 'v3': np.full((len(V), len(GRID)), np.nan)}; nw = np.zeros((len(V), 2), int); fam = defaultdict(set)
for j, v in enumerate(V):
    p = R / 'original' / ('BTC_' + inv[v['recording']]) / 'native/waveforms.npz'
    if not p.exists(): continue
    z = np.load(p); acc = {'pad': defaultdict(list), 'v3': defaultdict(list)}
    for (model, st), d in zip(z['keys'].tolist(), z['difference']):
        f = 'pad' if model.startswith('padding') else 'v3'; fam[f].add(model)
        w = np.cumsum(d.astype(float))
        if np.isfinite(w).all() and w.std() > 1e-12: acc[f][st].append(window_spectrum(w))
    for k, f in enumerate(['pad', 'v3']):
        if acc[f]: out[f][j] = np.mean([np.mean(s, 0) for s in acc[f].values()], 0); nw[j, k] = len(acc[f])
np.savez_compressed(OUT / 'cnn_evidence.npz', rec=np.array([v['recording'] for v in V]), pad=out['pad'], v3=out['v3'], nwin=nw)
print('videos with CNN evidence: pad', int(np.isfinite(out['pad']).all(1).sum()), 'v3', int(np.isfinite(out['v3']).all(1).sum()), '| checkpoints per family:', {k: len(v) for k, v in fam.items()}, '| windows per video (median):', np.median(nw, 0))
