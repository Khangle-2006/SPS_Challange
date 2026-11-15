from fasttrack_gate_v5 import below_twelve, stop_after_priority, ORDER
assert ORDER==[3,2,0,1,4]
assert below_twelve(11.999) and not below_twelve(12.) and not below_twelve(None)
assert not below_twelve(8.,False)
assert not stop_after_priority({3:dict(passed_native_gate=False)})
assert stop_after_priority({3:dict(passed_native_gate=False),2:dict(passed_native_gate=False)})
assert not stop_after_priority({3:dict(passed_native_gate=True),2:dict(passed_native_gate=False)})
assert not stop_after_priority({3:dict(passed_native_gate=False),2:dict(passed_native_gate=True)})
print('PASS strict12boundary, both priority runs required, either passing continues, invalid result cannot pass')
