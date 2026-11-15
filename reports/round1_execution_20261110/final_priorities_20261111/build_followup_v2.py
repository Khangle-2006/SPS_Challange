from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'padding_followup.py').read_text(encoding='utf-8')
start=s.index(' tests=[]\n');end=s.index(" save(dest/'C5PRIME.json'",start)
block=s[start:end]
block=block.replace('   hr=decode_difference(wave);tests.append',"""   try:hr=decode_difference(wave)
   except ValueError as e:
    assert str(e)=='invalid_prediction_waveform' and len(wave)==300 and np.isfinite(wave).all() and wave.std()<1e-10,'Unexpected waveform rejection'
    hr=None
   tests.append""")
block=block.replace("'BTC_STATIC.json'","'BTC_STATIC_V2.json'")
function='def btc_static(model,data,f,dest):\n'+block+'\n'
s=s[:start]+' btc_static(model,data,f,dest)\n'+s[end:]
s=s.replace('def evaluate_fold(f):',function+'def evaluate_fold(f):')
s=s.replace("assert not (OUT/'LAUNCH.json').exists()","assert not (OUT/'LAUNCH_V2.json').exists();assert not Path('/proc/2631023').exists()")
s=s.replace("OUT/'LAUNCH.json'","OUT/'LAUNCH_V2.json'").replace("OUT/'ERROR.json'","OUT/'ERROR_V2.json'")
needle=" while True:\n  for f in range(5):"
insert=""" # Reclassify only the previously mislabelled static abstentions; preserve all predictions.
 for f in range(5):
  dest=OUT/f'fold{f}'
  if (dest/'COMPLETE.json').exists():
   src=checkpoint(f);assert sha(src/'final.pt')==read(dest/'COMPLETE.json')['checkpoint_sha256']
   hashes={name:sha(dest/name) for name in ['NATIVE.json','C5PRIME.json','COMPLETE.json']}
   if not (dest/'BTC_STATIC_V2.json').exists():
    model=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));state=torch.load(src/'final.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True);model.eval()
    btc_static(model,BTCData(f),f,dest);del model,state;gc.collect()
   assert hashes=={name:sha(dest/name) for name in hashes}
   save(dest/'STATIC_CLASSIFICATION_REPAIR.json',dict(utc=now(),preserved_predictions=hashes,reason='Flat finite output correctly triggers decoder abstention, not unsupported input; original static receipt preserved.'))
 while True:
  for f in range(5):"""
assert needle in s;s=s.replace(needle,insert)
(P/'padding_followup_v2.py').write_text(s,encoding='utf-8')
