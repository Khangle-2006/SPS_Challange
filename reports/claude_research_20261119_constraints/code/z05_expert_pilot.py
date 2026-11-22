"""z05 - SMALL pilot of an expert model on alternative BTC crop sets (e.g. x19 stabilised boxes), saved after every video so that a
time-limited foreground call loses nothing. Same per-video readout as y05/y11 (spectrum_of). Compared with the x10 crops of the same
videos (out/y05/<model>.npz): share of videos whose spectral peak is within 5 bpm of the label, and |peak - label|.
Usage: venv_orppg/bin/python code/z05_expert_pilot.py run <model> <variant> <keys-file>   |   ... score <model> <variant,...> <keys-file>"""
import os, sys, json
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(k, '2')
os.environ.setdefault('KERAS_BACKEND', 'jax'); os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')
from pathlib import Path
import numpy as np
W = Path('/mnt/samsung_t5/khangle/facevitals/reports/claude_research_20261119_constraints'); OLD = W.parent / 'claude_research_20261118'
sys.path.insert(0, str(OLD / 'code')); sys.path.insert(0, str(W / 'code'))
if __name__ == '__main__':
    mode, model = sys.argv[1], sys.argv[2]; keys = (W / sys.argv[4]).read_text().strip().split(',')
    import y05_openrppg as Y5
    from lib_btc import GRID
    Y, allkeys = Y5.btc_keys(); idx = {k: i for i, k in enumerate(allkeys)}
    if mode == 'run':
        import rppg, y11_openrppg_conditions as Y11
        var = sys.argv[3]; out = W / 'out/z05' / var / (model + '.json'); out.parent.mkdir(parents=True, exist_ok=True); R = json.loads(out.read_text()) if out.exists() else {}
        m = rppg.Model(model)
        for k in keys:
            if k in R: continue
            s, sec = Y11.spectrum_of(m, W / 'out/x15' / var / (k + '.npz')); R[k] = None if s is None else s.tolist(); out.write_text(json.dumps(R)); print(model, var, k, 'done', len(R), '/', len(keys), '%.1f s' % sec, flush=True)
    else:
        A_ = sys.argv[:]; sys.argv = ['x']
        import y06_reader_quality as Q
        sys.argv = A_
        L = {k: Q.lab[idx[k]] for k in keys}
        with np.load(W / 'out/y05' / (model + '.npz')) as z: S0 = z['S']
        rows = {'x10 (per-frame tight box)': {k: S0[idx[k]] for k in keys if np.isfinite(S0[idx[k]]).all()}}
        for var in sys.argv[3].split(','):
            p = W / 'out/z05' / var / (model + '.json')
            if p.exists(): rows[var] = {k: np.array(v) for k, v in json.loads(p.read_text()).items() if v is not None}
        common = sorted(set.intersection(*[set(r) for r in rows.values()]))
        print(model, 'videos in common', len(common))
        for nm, r in rows.items():
            e = np.array([abs(GRID[r[k].argmax()] - L[k]) for k in common]); print(f'   {nm:28s} peak within 5 bpm {np.mean(e <= 5):.3f} | median |peak - label| {np.median(e):5.1f} | mean {e.mean():5.1f}')
