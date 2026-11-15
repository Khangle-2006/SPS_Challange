"""CPU-only post-checkpoint C5/public evaluation; never blocks training."""
import os,time,traceback,gc
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
from runtime_shared_v4 import install
install()
from common import *
from model_eval import c5_rows
from fasttrack_gate_v5 import ORDER

def main():
    import torch
    from sweep import make_model,deadline
    from phase_evaluate import evaluate
    lease=lock('deferred_eval_v5.lock');torch.set_num_threads(2)
    save(P/'DEFERRED_EVAL_V5_STARTED.json',dict(utc=now(),pid=os.getpid()))
    plan=read(P/'PLAN_ELIGIBLE.json')
    while True:
        worked=False
        for fold in ORDER:
            d=P/'sweep'/f'fold{fold}'
            if (d/'POSTCHECKPOINT_COMPLETE.json').exists() or not (d/'NATIVE_STOP_GATE.json').exists():continue
            status=read(d/'status.json')
            if status['state']=='rejected_static':
                save(d/'POSTCHECKPOINT_COMPLETE.json',dict(utc=now(),state='not_evaluated_rejected_static'));continue
            if not (P/'C5_COMPLETE.json').exists():continue
            guard(True)
            for file,h in read(P/'FROZEN_FASTTRACK_V5.json')['files'].items():assert sha(file)==h
            run=next(x for x in plan['runs'] if x['fold']==fold)
            assert sha(d/'final.pt')==status['checkpoint_sha256']
            model=make_model(run,'cpu');model.load_state_dict(torch.load(d/'final.pt',map_location='cpu',weights_only=False)['model']);model.eval()
            save(P/'DEFERRED_EVAL_V5_STATUS.json',dict(utc=now(),pid=os.getpid(),fold=fold,stage='C5prime'))
            assert not (d/'C5PRIME.json').exists(),'Inspect any prior output; no silent rerun'
            save(d/'C5PRIME.json',c5_rows(model,fold,'cpu','sweep_'+run['recipe_name']))
            subprocess.run([sys.executable,str(P/'report_progress.py')],check=True)
            save(P/'DEFERRED_EVAL_V5_STATUS.json',dict(utc=now(),pid=os.getpid(),fold=fold,stage='public'))
            evaluate(model,read(R/'t8_oval_20261031/training_index.json'),fold,2027,d/'public_evaluation',R/'t8_polarity_v3_20261102','cpu')
            save(d/'POSTCHECKPOINT_COMPLETE.json',dict(utc=now(),checkpoint_sha256=sha(d/'final.pt'),device='cpu',holdout_evaluations=0))
            del model;gc.collect();worked=True
        if (P/'SWEEP_STATUS.json').exists():
            s=read(P/'SWEEP_STATUS.json')
            if s['state'] in ['complete','stopped_user_native_gate']:
                folds=ORDER if s['state']=='complete' else s['completed_folds']
                if all((P/f'sweep/fold{f}/POSTCHECKPOINT_COMPLETE.json').exists() for f in folds):
                    save(P/'DEFERRED_EVAL_V5_COMPLETE.json',dict(utc=now(),folds=folds,holdout_evaluations=0));break
        if not worked:time.sleep(20)

if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'DEFERRED_EVAL_V5_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
