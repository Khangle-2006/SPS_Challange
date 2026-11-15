"""Run the permanent static-image acceptance contract on a supplied checkpoint."""
import argparse
from pathlib import Path
import torch
from test_static_tsm import EfficientPhys, use_replicate_padding, static_result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--padding",choices=["original","replicate"],required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    model=EfficientPhys(frame_depth=10,img_size=36)
    model.load_state_dict(torch.load(args.checkpoint,weights_only=False,map_location="cpu")["model"])
    if args.padding=="replicate":
        use_replicate_padding(model)
    result=static_result(model)
    print(result)
    raise SystemExit(0 if result["hr"] is None else 1)
