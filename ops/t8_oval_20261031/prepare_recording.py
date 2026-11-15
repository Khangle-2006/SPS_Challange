"""Adapter -> existing Haar crop cache, without loading a full raw video in RAM."""
import argparse
import json
from pathlib import Path
import numpy as np
from data_adapters import BACKENDS
from window_support import recording_windows


def prepare(recording, out, duration_s=10., stride_s=10., max_gap_s=.2):
    import cv2
    from oval_crop import OvalCrop
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    ppg = recording.require_synchronized_ppg()
    t = recording.timestamps
    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError("Recording too short")
    # Use measured time for cadence on newly adapted sources.
    frontend = OvalCrop()
    crops, boxes, valid, oval_points = [], [], [], []
    source_valid=recording.metadata.get('frame_valid',[True]*len(t))
    try:
        for i, rgb in enumerate(recording.frames):
            ok,crop,box,points=frontend.step(rgb)
            crops.append(crop);boxes.append(box);oval_points.append(points)
            valid.append(bool(ok and source_valid[i]))
        if len(crops)!=len(t):raise ValueError('Frame/time mismatch')
    finally:frontend.close()
    unique, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    merged = np.zeros((len(unique), 36, 36, 3), np.float32)
    np.add.at(merged, inverse, crops)
    merged /= counts[:, None, None, None]
    usable = np.ones(len(unique), bool)
    np.logical_and.at(usable, inverse, valid)
    # Keep the reference's own sampling grid; never stretch to match video length.
    lt, li, lc = np.unique(ppg.timestamps, return_inverse=True, return_counts=True)
    lv = np.zeros(len(lt))
    np.add.at(lv, li, ppg.values)
    lv /= lc
    ppg_valid=np.ones(len(lt),bool)
    np.logical_and.at(ppg_valid,li,ppg.valid_mask)
    windows=recording_windows(recording,video_valid=np.asarray(valid,bool),duration_s=duration_s,
                              stride_s=stride_s,max_gap_s=max_gap_s)
    out.mkdir(parents=True)
    np.save(out/"rgb.npy", merged.astype(np.float16))
    np.savez(out/"meta.npz", t=unique, original_t=t, boxes=boxes, valid=usable, oval_points=np.asarray(oval_points,np.float32),
             ppg_t=lt, ppg=lv, ppg_valid=ppg_valid,
             window_start_s=np.asarray([w['start_s'] for w in windows['valid_windows']]))
    (out/'windows.json').write_text(json.dumps(windows,indent=2))
    manifest = {"subject_id": recording.subject_id, "recording_id": recording.recording_id,
                "frames": len(t), "metadata": recording.metadata, "cache_schema": 3,
                "window_policy":windows['parameters'],"valid_windows":len(windows['valid_windows']),
                "frontend": "T8 unified large oval v2; tight masked bbox to36; identical datasets/arms/inference"}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=BACKENDS, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--recording", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", help="AiPEX mapping manifest")
    p.add_argument('--audit-dir',help='MCD: directory with completed V3 receipts')
    p.add_argument("--sensor-offset-s", type=float,
                   help="UBFC1 only: independently verified sensor-to-video clock offset")
    a = p.parse_args()
    if a.dataset=='mcd' and not a.audit_dir:p.error('--audit-dir required for MCD')
    backend = BACKENDS[a.dataset](a.root, a.manifest) if a.dataset == "aipex" else (BACKENDS[a.dataset](a.root,a.audit_dir) if a.dataset=='mcd' else BACKENDS[a.dataset](a.root))
    if a.sensor_offset_s is not None and a.dataset != "ubfc1":
        p.error("--sensor-offset-s is only supported for UBFC1")
    record = backend.load(a.recording, sensor_offset_s=a.sensor_offset_s) if a.dataset == "ubfc1" else backend.load(a.recording)
    print(json.dumps(prepare(record, a.out), indent=2))
