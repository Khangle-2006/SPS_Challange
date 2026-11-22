import pickle, numpy as np
from pathlib import Path
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints/out/u28')
rng = np.random.default_rng(0)
for fn, ms in (('sim.pkl', ['CHROM', 'PBV', 'PBVfix', 'SC', 'SC_h', 'SCH_g1', 'SCH_g2']), ('sim2.pkl', ['CHROM', 'PBVfix', 'SC', 'POS_T1', 'SC_T1', 'SCh_T1', 'SC_T2'])):
    R = pickle.load((W / fn).open('rb')); print(f'== {fn}: paired difference vs POS, pooled over amplitudes, bootstrap over videos (2000)')
    for grp, f in (('easy noise', lambda r: r['conf']), ('hard noise', lambda r: not r['conf'])):
        Rs = [r for r in R if f(r)]; vids = sorted(set(r['i'] for r in Rs)); by = {v: [r for r in Rs if r['i'] == v] for v in vids}
        line = []
        for m in ms:
            d = np.array([np.mean([r[m] - r['POS'] for r in by[v]]) for v in vids]); e = np.array([np.mean([min(r['err_' + m], 40) - min(r['err_POS'], 40) for r in by[v]]) for v in vids])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]; be = [e[rng.integers(0, len(e), len(e))].mean() for _ in range(2000)]
            line.append(f'{m} hit {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f};{np.percentile(bs, 97.5):+.3f}] |err| {e.mean():+.2f} [{np.percentile(be, 2.5):+.2f};{np.percentile(be, 97.5):+.2f}]')
        print(f'{grp} ({len(vids)} videos):'); [print('   ', l) for l in line]
