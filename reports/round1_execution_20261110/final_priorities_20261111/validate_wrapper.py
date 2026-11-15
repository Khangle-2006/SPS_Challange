import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import padding_queue as q
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from EfficientPhys_fixed import EfficientPhys
from replicate_tsm import use_replicate_padding
from test_static_tsm import static_result
torch.set_num_threads(2)
out=[]
for fold in [1,2,3,4]:
    q.p.baseline.seed_all(2027,fold)
    m=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36))
    initial=q.p.baseline.model_hash(m)
    old=q.p.read(q.p.V3/f'fold{fold}/seed2027/joint/provenance.json')
    assert initial==old['initial_model_sha256']
    rng=torch.get_rng_state().clone();bn={k:v.clone() for k,v in m.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        probe=use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36));probe.load_state_dict(m.state_dict())
        result=static_result(probe)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in bn.items())
    assert result['hr'] is None
    out.append(dict(fold=fold,initial_model_sha256=initial,static=result,RNG_unchanged=True,live_state_unchanged=True))
q.save(q.P/'WRAPPER_TEST.json',dict(passed=True,utc=q.now(),folds=out,wrapper_sha256=q.p.sha(q.P/'padding_queue.py')))
print('WRAPPER_TEST_PASS',flush=True)
