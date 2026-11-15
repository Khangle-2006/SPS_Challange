"""Freeze only the declared amended protocol; never read outcome metrics."""
from common import *
plan=read(S/'PLAN.json')
validation=read(P/'SWEEP_CPU_VALIDATION.json');assert validation['passed']
for name in ['train_sweep.py','training_provider.py','aipex_data.py','model_eval.py','common.py']:
 assert sha(P/name)==validation['code'][name],('Changed since CPU validation',name)
assert not (P/'PLAN.json').exists(),'Plan already frozen'
plan.update(source_plan_sha256=sha(S/'PLAN.json'),source_plan=str(S/'PLAN.json'),root=str(P),btc_aux_batch_size=4,aipex_aux_batch_size=4,aipex_roi='forehead',aipex_fs_assumed=15,weak_sigma_bpm=5,static_every_epoch=True,C5='x264 CRF23 maxrate500k bufsize1000k defaultGOP; unverified VideoWriter proxy',dependencies=['AdaBN native and C5prime','A3 complete including inference'],output_reserve_bytes=400000000,codec_cache_bytes=300000000)
for run in plan['runs']:
 run['recipe_name']=['augmentation','augmentation','weak_CE_BTC_AiPEX','SiNC_BTC_AiPEX','SiNC_BTC_AiPEX'][run['fold']]
save(P/'PLAN.json',plan)
deps=[P/n for n in ['train_sweep.py','model_eval.py','training_provider.py','aipex_data.py','common.py','PLAN.json','GATES.md','APPROVAL.json']]
deps += [S/n for n in ['PLAN.json','sweep.py','btc_data.py','domain_loss.py','codec_provider.py']]
deps += [V/'phase_repair/repair_loss.py',V/'tsm_padding/replicate_tsm.py',V/'tsm_padding/test_static_tsm.py']
frozen=read(S/'FROZEN.json')
for f,h in frozen.items():assert sha(f)==h,('Inherited frozen dependency changed',f)
frozen.update({str(f):sha(f) for f in deps});frozen[str(P/'SWEEP_CPU_VALIDATION.json')]=sha(P/'SWEEP_CPU_VALIDATION.json')
save(P/'FROZEN.json',frozen)
print('FROZEN',sha(P/'PLAN.json'))
