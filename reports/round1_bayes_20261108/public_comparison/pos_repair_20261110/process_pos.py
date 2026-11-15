"""Canonical POS entry point for existing frozen oval spatial-RGB caches.

Example: python process_pos.py --spatial-npz native.npz --start 0
Input: t [T], rgb [T,3], valid [T], exported from frozen oval36 crops.
Do not pass per-skin-ROI means or container FPS in place of the source clock.
"""
import argparse,json
import numpy as np
from pos_corrected import estimate_window,FRONTEND

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spatial-npz',required=True)
    p.add_argument('--start',type=float,required=True)
    a=p.parse_args()
    with np.load(a.spatial_npz) as z:
        result=estimate_window(z['t'],z['rgb'],z['valid'],a.start,frontend=FRONTEND)
    print(json.dumps(dict(frontend=FRONTEND,method='canonical_POS_restored_v1',**result),allow_nan=False))

if __name__=='__main__':main()
