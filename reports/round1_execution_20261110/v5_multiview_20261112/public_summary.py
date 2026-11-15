import os,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
P=Path(__file__).resolve().parent;N=P.parent/'final_priorities_20261111';V=P.parent.parent/'round1_v4_20261105'
sys.path.insert(0,str(V/'aggregation_20261107'));import aggregate as a
def main():
    frozen=a.read(a.FROZEN);rows=[];receipts=[]
    for f in range(5):
        paths={'V5':P/f'runs/fold{f}/seed2027/joint','padding_fixed':(V/'v4_padding_pilot' if f==0 else N/'padding_fixed')/f'fold{f}/seed2027/joint','old_V4_static_fail':V/f'v4_repaired/fold{f}/seed2027/joint'}
        for branch,d in paths.items():
            rr=a.load_run(d,frozen)
            for r in rr:r['branch']=branch
            rows+=rr;receipts.append(dict(fold=f,branch=branch,checkpoint_sha256=a.sha(d/'final.pt')))
        rows += [r for r in frozen if r['branch'] in ['v3_joint','POS','CHROM'] and r['seed']==2027 and r['fold']==f]
    branches=['V5','padding_fixed','old_V4_static_fail','v3_joint','POS','CHROM'];persons=a.person_means(a.person_seed_rows(rows),required_seeds=(2027,));result=a.summary(persons,branches=branches)
    result['paired_V5_minus']=result.pop('paired_V4_minus');result.update(checkpoints=receipts,strict_reference_failure_hash_parity=True,holdout_evaluations=0,primary_public='padding_fixed',seeds=[2027]);a.save(P/'PUBLIC_FINAL.json',result)
if __name__=='__main__':main()
