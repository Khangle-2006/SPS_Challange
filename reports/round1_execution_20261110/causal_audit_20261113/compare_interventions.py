"""Explicit contract extension; original comparator remains untouched.
common_input is the real SOURCE video before intervention, not transformed pixels.
Differences are expected; neither equality nor first difference establishes damage.
"""
from common_causal import *
sys.path.insert(0,str(C/'supplied/FaceVitals_BTC_Causal_Audit_V1'))
import compare_stage_exports as supplied
def compare(a,b):
 supplied.KINDS.add('controlled_degradation')
 out=supplied.compare(a,b)
 out['numeric_comparator_status']=out.pop('status');out['status']='CONTROLLED_INTERVENTION_DESCRIPTIVE_DIFFERENCES'
 out['not_an_equivalence_test']=True;out['not_a_defect_or_HR_failure_test']=True
 out['extension']='comparison_kind controlled_degradation; identical original source hash and fixed checkpoint; intentionally different transforms. Original numeric comparison unchanged.'
 out['original_comparator_sha256']=sha(Path(supplied.__file__))
 return out
