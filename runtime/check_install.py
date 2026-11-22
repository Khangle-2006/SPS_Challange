"""Check shipped parameters and the two package weight files used by three experts."""
from common import verify_weights,load_model
import json

if __name__=='__main__':
    weights=verify_weights();_,model=load_model()
    for f in range(5):load_model(f)
    print(json.dumps({'expert_weights':weights,'deployment_reader':model,'ready':True},indent=2))
