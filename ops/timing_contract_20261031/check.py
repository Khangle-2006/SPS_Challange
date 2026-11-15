"""Read-only runtime instrumentation of frozen T8/T9 temporal contracts. No GPU."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    sys.path.insert(0, str(a.t8))
    import t8_run
    import t8_evaluate
    import t8_t1_loader
    assert t8_run.load_clip is t8_evaluate.load_clip is t8_t1_loader.load_clip
    root = a.cache / 'subject1' / 'f10_b500' / 'crop'
    with np.load(root / 'meta.npz') as m:
        t = m['t'].copy()
        start = 10.0
        q = start + np.arange(301) / 30
        images = np.load(root / 'rgb.npy', mmap_mode='r')
        x, y, ref = t8_evaluate.load_clip(root, start)
        batch, targets = t8_run.batch_arrays(
            {'roots': {'UBFC2': str(a.cache)}},
            [('UBFC2', 'subject1', start, 'f10_b500')])
        assert np.array_equal(batch[0], x) and np.array_equal(targets[0], y)
        # Independent scalar interpolation at every spatial/channel location.
        flat = images.reshape(len(images), -1)
        expected = np.stack([np.interp(q, t, flat[:, j]) for j in range(flat.shape[1])], axis=1)
        expected = expected.reshape(301, 36, 36, 3)
        expected = ((expected - expected.mean()) / expected.std()).transpose(0, 3, 1, 2)
        max_error = float(np.max(np.abs(expected - x)))
        assert max_error < 1e-5, max_error
        t8 = dict(train_eval_same_function=True, input_tensors_bit_equal=True,
                  source_effective_fps=float((len(t)-1)/(t[-1]-t[0])),
                  grid_fs_hz=30, image_samples=301, output_difference_samples=300,
                  independent_interpolation_max_abs_error=max_error,
                  independent_ppg_clock=True, test_record='subject1', start_s=start,
                  decoder_grid_fs_hz=30,
                  source_hashes={p: digest(a.t8/p) for p in
                                 ['t8_t1_loader.py', 't8_run.py', 't8_evaluate.py']})
    sys.path.insert(0, str(a.t9))
    import run_t9
    source_pos, source_decoder = run_t9.pos, run_t9.estimate_hr
    calls = []
    def logged_pos(rgb, fs):
        calls.append(dict(stage='POS', fs_hz=fs, samples=len(rgb),
                          window_samples=int(np.ceil(1.6*fs))))
        assert fs == 30 and len(rgb) == 300
        return source_pos(rgb, fs)
    def logged_decoder(wave, fs, **kwargs):
        calls.append(dict(stage='decoder', fs_hz=fs, samples=len(wave), band=kwargs['band']))
        assert fs == 30 and kwargs['band'] == (.7, 3.)
        return source_decoder(wave, fs, **kwargs)
    run_t9.pos, run_t9.estimate_hr = logged_pos, logged_decoder
    trace = a.t9out / 'subject1' / 'raw10_trace.npz'
    assert (trace.parent / 'complete.json').exists()
    with np.load(trace) as z:
        run_t9.predict(z['t'], z['rgb'], z['valid'], 10.0)
        t9 = dict(source_effective_fps=float((len(z['t'])-1)/(z['t'][-1]-z['t'][0])),
                  input_samples=len(z['t']), calls=calls,
                  source_hashes={p: digest(a.t9/p) for p in ['run_t9.py', 'hr_baseline.py']},
                  trace_sha256=digest(trace))
    result = dict(passed=True, scope='one completed real 10fps cache and trace; no model training',
                  t8=t8, t9=t9,
                  limitation='T8 deployment packager not yet implemented/validated; must share this contract.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['t8', 'cache', 't9', 't9out', 'output']:
        p.add_argument('--'+key, required=True, type=Path)
    main(p.parse_args())
