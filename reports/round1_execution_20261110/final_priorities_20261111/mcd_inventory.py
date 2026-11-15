"""Read-only ffprobe of the previously frozen 357 DEV60 recordings, no D3 scan."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import csv,json,subprocess,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent;R=P.parents[2];root=R/'datasets/MCD-rPPG'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(name,x):(P/name).write_text(json.dumps(x,indent=2))
def main():
    prior=json.loads((P/'MCD_FROZEN_DEV.json').read_text(encoding='utf-8-sig'));assert len(prior)==357
    index=json.loads((R/'t8_oval_20261031/training_index.json').read_text())
    # Inspect known index structure only, retained with report for fold-source verification.
    save('MCD_INDEX_SCHEMA.json',dict(keys=list(index),folds=index.get('folds')))
    allowed={Path(r['video']).stem:r for r in prior};subjects={r['subject'] for r in prior};assert len(subjects)==60
    with (root/'db.csv').open() as f: db={Path(r['video']).stem:r for r in csv.DictReader(f) if Path(r['video']).stem in allowed}
    assert set(db)==set(allowed)
    rows=[]
    for rid,old in allowed.items():
        row=db[rid];file=root/row['video']
        raw=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-show_format','-of','json',str(file)]))
        s=raw['streams'][0];duration=old.get('duration_s');meta=root/row['meta'];ppg=root/row['ppg'];sync=root/row['ppg_sync']
        mt=[line.split(maxsplit=1)[1] for line in meta.read_text().splitlines() if line.strip()]
        clock=[datetime.fromisoformat(t) for t in mt];span=(clock[-1]-clock[0]).total_seconds()
        ppgrows=[line.split(maxsplit=1) for line in ppg.read_text().splitlines() if line.strip()]
        pt=[datetime.fromisoformat(v[1]) for v in ppgrows]
        overlap=max(0,(min(clock[-1],pt[-1])-max(clock[0],pt[0])).total_seconds())
        r=dict(recording=rid,person=old['subject'],session=old['step'],camera=old['camera'],official_fold=old['official_fold'],width=s['width'],height=s['height'],codec=s['codec_name'],pixel_format=s.get('pix_fmt'),mux_avg_rate=s.get('avg_frame_rate'),mux_nominal_rate=s.get('r_frame_rate'),mux_duration=raw['format'].get('duration'),source_bytes=file.stat().st_size,metadata_rows=len(mt),metadata_span_seconds=span,metadata_effective_fps=(len(mt)-1)/span,retained_clock_duration_seconds=duration,wholefile_sourceclock_kbps=file.stat().st_size*8/span/1000,ppg_exists=ppg.exists(),ppg_sync_exists=sync.exists(),ppg_sha256=sha(ppg),sync_sha256=sha(sync),meta_sha256=sha(meta),wallclock_overlap_seconds=overlap,wallclock_overlap_fraction=overlap/span,prior_processing_error=old.get('error'),synchronization_verified=False,synchronization_basis='Published synchronized acquisition; shared wallclock and PPG verified here; exposure/transport latency not independently verified')
        rows.append(r)
        save('MCD_INVENTORY_STATUS.json',dict(completed=len(rows),total=len(prior),last=rid,pid=os.getpid()))
    with (P/'MCD_VIEW_INVENTORY.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    persons={s:{r['official_fold'] for r in rows if r['person']==s} for s in subjects};assert all(len(v)==1 for v in persons.values())
    groups={}
    for r in rows:groups.setdefault((r['person'],r['session']),[]).append(r)
    shared=sum(len(rs)==3 and len({r['ppg_sha256'] for r in rs})==1 for rs in groups.values())
    save('MCD_INVENTORY_COMPLETE.json',dict(utc=datetime.now(timezone.utc).isoformat(),records=len(rows),people=len(subjects),sessions=len(groups),same_person_same_official_fold=True,shared_ppg_triplets=shared,all_ppg_present=all(r['ppg_exists'] and r['ppg_sync_exists'] for r in rows),exposure_synchronization_verified=False,V5_release=False,reason='Shared acquisition clock is verified; view-dependent exposure/transport latency remains unresolved. No new offsets fitted.',holdout_access=False,D3_access=False,script_sha256=sha(__file__)))
if __name__=='__main__':
    try:main()
    except Exception as e:save('MCD_INVENTORY_ERROR.json',dict(error=str(e),traceback=traceback.format_exc()));raise
