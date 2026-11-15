import ast,json,hashlib,inspect
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;S=P/'sources/LQ-rPPG'
m=S/'setup/mamba/mamba_ssm/modules/mamba_simple.py'
tree=ast.parse(m.read_text(encoding='utf-8'));cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Mamba');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
args=[n.arg for n in fn.args.args];assert 'bimamba' not in args and fn.args.kwarg is None
model=S/'neural_methods/model/LQC2F.py';calls=[n for n in ast.walk(ast.parse(model.read_text(encoding='utf-8'))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Mamba'];assert any(k.arg=='bimamba' for n in calls for k in n.keywords)
weight=S/'PreTrainedModels/MMPD_MMPD_LQC2F_rPPG_Epoch19.pth';state=torch.load(weight,map_location='cpu',weights_only=True)['C2F_model'];back=[k for k in state if 'A_b_log' in k or 'conv1d_b' in k or 'dt_proj_b' in k]
assert len(back)>0
result=dict(commit='dc0143a6e67e12c30aa722d902e078cb24b1f17b',weights_downloaded=True,weights_only_load_passed=True,mamba_constructor_arguments=args,accepts_bimamba=False,model_requires_bimamba=True,checkpoint_backward_keys=back,checkpoint_parameters=len(state),faithful_zero_shot_ready=False,reason='Published model requests bimamba=True, but bundled Mamba does not accept it and lacks backward parameters required by checkpoint.',model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),mamba_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),weights_modified=False,model_modified=False)
(P/'LQ_COMPATIBILITY.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:result[k] for k in ['weights_downloaded','accepts_bimamba','model_requires_bimamba','faithful_zero_shot_ready']}))
