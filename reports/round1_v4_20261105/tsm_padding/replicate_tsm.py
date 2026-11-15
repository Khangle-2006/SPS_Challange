"""Versioned TSM boundary replacement for the frozen EfficientPhys architecture."""
from EfficientPhys_fixed import TSM

class ReplicateTSM(TSM):
    def forward(self, features):
        # Frozen TSM already performs the two shifts correctly in the interior.
        shifted = super().forward(features)
        n,c,h,w = features.shape
        original = features.reshape(-1,self.n_segment,c,h,w)
        output = shifted.reshape_as(original)
        fold = c // self.fold_div
        output[:, -1, :fold] = original[:, -1, :fold]
        output[:, 0, fold:2*fold] = original[:, 0, fold:2*fold]
        return output.reshape(n,c,h,w)

def use_replicate_padding(model):
    for name in ("TSM_1","TSM_2","TSM_3","TSM_4"):
        old = getattr(model,name)
        if type(old) is not TSM:
            raise TypeError(f"Unexpected layer {name}: {type(old)}")
        setattr(model,name,ReplicateTSM(old.n_segment,old.fold_div))
    return model
