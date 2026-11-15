"""Launch-time data contracts and explicit complete-GOP path binding, before CUDA."""
from common import *
for file,h in read(P/'EXECUTION_FROZEN_V2.json')['files'].items():assert sha(file)==h,('Execution dependency changed',file)
plan=read(P/'PLAN_ELIGIBLE.json')
for shard in range(4):
 assert (P/f'btc_c5ensemble/SHARD{shard}_COMPLETE.json').exists(),'A3 all-checkpoint BTC comparison pending'
assert sha(R/'t8_oval_20261031/training_index.json')==plan['index_sha256']
assert sha(V/'btc_zeroshot/dev_manifest.json')==plan['btc_manifest_sha256']
assert sha(V/'btc_zeroshot/btc_split.json')==plan['split_sha256']==SPLIT
g=read(P/'GOP_COMPLETE.json');assert len(g['rows'])==221
meta={v['recording']:v for v in read(V/'btc_zeroshot/dev_manifest.json')['videos']}
for row in g['rows']:
 file=P/'native_gop'/(row['recording']+'.json')
 assert sha(file)==row['gop_sha256'] and read(file)['source_sha256']==meta[row['recording']]['sha256']
import btc_data
# The unchanged BTCData class uses P.parent/native_gop only for this lookup.
# BTC manifest/cache root is its separate frozen BTC variable and stays unchanged.
btc_data.P=P/'compatibility_path_anchor'
assert btc_data.BTC==V/'btc_zeroshot'
save(P/'SWEEP_INPUT_GUARDS.json',dict(utc=now(),plan_sha256=sha(P/'PLAN_ELIGIBLE.json'),GOP_manifest_sha256=sha(P/'GOP_COMPLETE.json'),GOP_root=str(P/'native_gop'),path_binding='btc_data.P.parent/native_gop; no changes to sampling, waveform, labels or folds',wrapper_sha256=sha(__file__),holdout_evaluations=0))
from train_sweep_eligible_v2 import main
main()
