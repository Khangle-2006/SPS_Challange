"""Lock people using metadata only; export PR only for development people."""
import csv, hashlib, json, math, os, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from openpyxl import load_workbook

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def key(x): return re.sub(r'\d+',lambda m:f'{int(m[0]):06d}',x)

def assign(people):
    rng=np.random.Generator(np.random.PCG64(0))
    strata=defaultdict(list)
    for pid,p in sorted(people.items(),key=lambda x:key(x[0])):
        skins=sorted(p['skins'])
        p['skin_stratum']=skins[0] if len(skins)==1 else 'Mixed/unresolved'
        strata[p['group']+'|'+p['skin_stratum']].append(pid)
    # Largest remainder allocates exactly ceil(.2*N), including rare strata.
    n_hold=math.ceil(.2*len(people))
    allocation={s:math.floor(.2*len(v)) for s,v in strata.items()}
    order=sorted(strata,key=lambda s:(-(.2*len(strata[s])-allocation[s]),s))
    for s in order[:n_hold-sum(allocation.values())]:allocation[s]+=1
    fold_counts=[0]*5
    for s in sorted(strata):
        ids=list(strata[s]);rng.shuffle(ids)
        hold=set(ids[:allocation[s]])
        counts=[0]*5
        for pid in ids:
            if pid in hold:people[pid].update(partition='holdout',fold=None)
            else:
                f=min(range(5),key=lambda f:(counts[f],fold_counts[f],f))
                people[pid].update(partition='dev',fold=f)
                counts[f]+=1;fold_counts[f]+=1
    assert sum(p['partition']=='holdout' for p in people.values())==n_hold
    return allocation

def main():
    sheet=Path(os.environ['TEMP'])/'FaceVitals_EDA_source.xlsx'
    assert sha(sheet)=='e52a9b190bf91f8a5d6d476b8374b8904f47328a668bfbe7cd60f3c14fafc49a'
    recpath=ROOT/'organiser_download/BTC_rows_reconciled.csv'
    rec={(r['group'],int(r['sheet_row'])):r for r in csv.DictReader(recpath.open(encoding='utf-8-sig'))}
    wb=load_workbook(sheet,read_only=True,data_only=True)
    people={};metadata=[]
    for group in ['Normal','Abnormal']:
        for row,values in enumerate(wb[group].iter_rows(min_row=2,max_col=9,values_only=True),2):
            if not(values[0] and values[1]):continue
            recording=str(values[0]).strip();pid=re.match(r'^(S_(?:AB|N)\d+)',recording)[0]
            skin=' '.join(str(values[5]).split()).casefold()
            p=people.setdefault(pid,dict(group=group,skins=set(),recordings=[]))
            assert p['group']==group
            p['skins'].add(skin);p['recordings'].append(recording)
            metadata.append(dict(group=group,sheet_row=row,person=pid,recording=recording,skin=skin,
                task=''.join(str(values[8]).split()),lighting=next((s for s in ['Bright','Dim','Med'] if '_'+s+'_' in recording),'Unspecified')))
    assert len(people)==202 and len(metadata)==295
    allocation=assign(people)
    for p in people.values():p['skins']=sorted(p['skins']);p['recordings']=sorted(p['recordings'])
    lock=dict(version=1,seed=0,rng='numpy PCG64',person_count=202,holdout_people=41,
        allocation='ceil(20% total); largest remainder per group x normalized skin; conflicting skin=Mixed/unresolved; ties lexical',
        fold_assignment='Within-stratum random order; greedy minimum stratum count, then global count, then fold number',
        source_sha256=sha(sheet),reconciliation_sha256=sha(recpath),people=people,stratum_holdout=allocation,
        disclosure='All-cohort descriptive EDA preceded this split. No BTC model predictions or PR values entered assignment. IDs are organizer codes; unresolved real-person aliases cannot be inferred.',
        holdout_access='No post-lock frames, C5, QC, labels, predictions or calibration until final chosen configuration and one evaluation.')
    payload=json.dumps(lock,indent=2,ensure_ascii=False,allow_nan=False).encode('utf-8')
    split=HERE/'btc_split.json'
    if split.exists():assert split.read_bytes()==payload,'Refuse different existing split'
    else:split.write_bytes(payload)
    (HERE/'btc_split.sha256').write_text(sha(split)+'  btc_split.json\n')
    # Only dev row references are accessed for PR after the split bytes are locked.
    dev=[];missing=[]
    media={r['id']:r for r in csv.DictReader((ROOT/'eda_three_datasets/media.csv').open(encoding='utf-8-sig')) if r['dataset']=='BTC'}
    for m in metadata:
        p=people[m['person']]
        if p['partition']!='dev':continue
        r=rec[m['group'],m['sheet_row']]
        if not r['verified_drive_id']:missing.append(dict(m,reason='source_video_missing_deferred_by_user'));continue
        pr=float(wb[m['group']].cell(m['sheet_row'],14).value)
        v=media[r['verified_drive_id']]
        dev.append(dict(m,fold=p['fold'],pr_bpm=pr,path=r['path'],sha256=r['sha256'],drive_id=r['verified_drive_id'],media=v))
    dump(HERE/'dev_manifest.json',dict(split_sha256=sha(split),videos=dev,missing=missing))
    dump(HERE/'SPLIT_SUMMARY.json',dict(split_sha256=sha(split),people=202,holdout_people=41,dev_people=161,
        dev_videos=len(dev),dev_missing=len(missing),dev_folds=dict(Counter(p['fold'] for p in people.values() if p['partition']=='dev')),
        groups={g:{part:sum(p['group']==g and p['partition']==part for p in people.values()) for part in ['dev','holdout']} for g in ['Normal','Abnormal']},
        conflicting_skin_people=[pid for pid,p in people.items() if len(p['skins'])>1]))
    print((HERE/'SPLIT_SUMMARY.json').read_text())
if __name__=='__main__':main()
