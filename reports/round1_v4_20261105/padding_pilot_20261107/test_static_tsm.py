"""Regression contract at the model-output / canonical HR-decoder boundary."""
import sys,unittest
from pathlib import Path
import torch
REMOTE=Path("/data_pool/khangle/facevitals/code/t8_polarity_v3_20261102")
LOCAL=Path(__file__).resolve().parents[2]/"round1_final_20261105/revision_02/code/vendor/v3"
sys.path.insert(0,str(REMOTE if REMOTE.is_dir() else LOCAL))
from EfficientPhys_fixed import EfficientPhys
from t8_evaluate import decode_difference
from replicate_tsm import ReplicateTSM,use_replicate_padding

def static_result(model):
    # A spatially textured, time-constant 10s clip with its real terminal frame.
    frame=torch.linspace(-1,1,3*36*36).reshape(1,1,3,36,36)
    x=frame.expand(1,301,3,36,36).contiguous()
    with torch.inference_mode():
        wave=model.eval()(x)[0].cpu().double().numpy()
    if not torch.isfinite(torch.from_numpy(wave)).all():
        raise AssertionError("Nonfinite outputs do not qualify as a static abstention")
    result={"std":float(wave.std()),"hr":None}
    try:
        result["hr"]=decode_difference(wave)
    except ValueError as error:
        if str(error)!="invalid_prediction_waveform":
            raise
    return result

class StaticContract(unittest.TestCase):
    def test_real_decoder_abstains(self):
        torch.manual_seed(2027)
        self.assertIsNone(static_result(use_replicate_padding(EfficientPhys(frame_depth=10,img_size=36)))["hr"])

    def test_shifts_preserve_block_edges_and_interior(self):
        x=torch.arange(60.).reshape(20,3,1,1).requires_grad_()
        y=ReplicateTSM(10,3)(x)
        for t in range(20):
            begin=t//10*10; end=begin+9
            self.assertEqual(y[t,:,0,0].tolist(),[x[min(t+1,end),0].item(),x[max(t-1,begin),1].item(),x[t,2].item()])
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
