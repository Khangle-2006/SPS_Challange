"""Run the unchanged H264 trainer on causal-EMA ROI caches."""
import hashlib,json
from pathlib import Path
import train_h264_ablation as base
base.CLEAN=base.DATA/'roi_ema010_clean_v1'
base.COMP_TRAIN=base.DATA/'roi_ema010_train500_v1'
base.COMP_VAL=base.DATA/'roi_ema010_val500_v1'
base.OUT=base.ROOT/'results/roi_ema010_ablation_v1'
if __name__=='__main__':
    base.OUT.mkdir(parents=True,exist_ok=True)
    p=base.OUT/'roi_protocol.json'
    protocol={'tau_s':.10,'reset_if_iou_below':.50,'filter':'causal EMA of center and log-size; recorded elapsed timestamps; no lookahead; reset to detector when IoU falls below .50',
        'seeds':[2027,2028,2029],'reference':'h264_ablation_v1 augmentation',
        'selection':'unchanged equal mean clean/H264500 validation MAE; earliest tie',
        'keep_gate':{'relative_selection_score_improvement_min':.10,'clean_MAE_increase_max':.25,'h264_improved_seeds_min':2,'h264_P95_increase_max':1.,'h264_over20_increase_max':0},
        'test_opened':False,'extra_rates':'250/1000 already seen in prior experiments, secondary development conditions only',
        'wrapper_sha256':base.sha(Path(__file__))}
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2))
    base.main()
