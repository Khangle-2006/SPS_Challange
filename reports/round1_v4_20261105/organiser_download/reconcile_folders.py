"""Copy the public folder-only files using rclone, then audit folders, sheet rows and P100 bytes."""
import csv, fcntl, hashlib, json, os, subprocess, time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode
from mirror import HERE,DEST,read,save,probe,fetch,retry,guard,now,url

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()

def write_csv(name,rows):
    path=HERE/name; tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def main():
    os.umask(0o077)
    index=read(HERE/"manifest.json")
    sheets=index["rows"]
    linked={r["drive_id"] for r in sheets}
    folder={}
    for group in ["Normal","Abnormal"]:
        for item in read(HERE/(group.lower()+"_folder.json")):
            k=item["drive_id"]
            assert k not in folder
            folder[k]=dict(**item,group=group)
    assert len(folder)==296
    extra=sorted(folder.keys()-linked)
    assert len(extra)==4
    status=read(HERE/"status.json")
    budget=read(HERE/"budget.json")
    plans=[]
    for k in extra:
        row=dict(group=folder[k]["group"],drive_id=k,url="https://drive.google.com/file/d/"+k+"/view")
        meta=retry(lambda:probe(row))
        assert meta["filename"]==folder[k]["filename"]
        plans.append((k,row,meta))
    guard(max(0,budget["source_bytes"]-status["bytes"])+sum(m["size"] for _,_,m in plans),budget["a100_bytes"])
    completed=[]
    for k,row,meta in plans:
        d=DEST/row["group"]/k
        d.mkdir(parents=True,exist_ok=True)
        dest=d/meta["filename"]; partial=dest.with_name(dest.name+".part")
        if not dest.exists() and (not partial.exists() or partial.stat().st_size!=meta["size"]):
            subprocess.run(["rclone","copyurl","--retries","3","--low-level-retries","5","--contimeout","20s","--timeout","90s",
                            "https://drive.usercontent.google.com/download?"+urlencode(url(row)),str(partial)],check=True,timeout=1200)
        if not dest.exists() and partial.stat().st_size!=meta["size"]:
            raise RuntimeError("rclone size differs from source for "+k)
        rec=fetch(k,row,meta)
        completed.append(dict(drive_id=k,source_name=rec["filename"],sha256=rec["sha256"],bytes=rec["bytes"]))
        save(HERE/"rclone_supplement.json",dict(files=completed,utc=now()))
    while not (HERE/"result.json").exists():
        if (HERE/"worker_error.json").exists():raise RuntimeError("Original download stopped; see worker_error.json")
        time.sleep(10)
    source_meta=read(HERE/"source_metadata.json")
    expected={k:m["size"] for k,m in source_meta.items()}
    expected.update({k:m["size"] for k,_,m in plans})
    by_name=defaultdict(list);files=[];verified={}
    for receipt in sorted((HERE/"receipts").glob("*.json")):
        rec=read(receipt); k=rec["drive_id"]; p=Path(rec["path"])
        assert p.is_file() and not p.is_symlink() and p.parent.name==k and p.parent.parent.parent==DEST
        assert expected[k]==p.stat().st_size==rec["bytes"] and digest(p)==rec["sha256"]
        group=p.parent.parent.name; name_key=(group,p.stem.casefold())
        by_name[name_key].append(k); verified[k]=rec
        files.append(dict(group=group,source_filename=p.name,drive_id=k,bytes=rec["bytes"],sha256=rec["sha256"],path=str(p)))
    matched=set(); audit=[]
    for r in sheets:
        group=r["group"]; key=r["drive_id"]; slots=by_name[(group,r["recording"].casefold())]
        assert len(slots)<=1
        chosen=slots[0] if slots else None
        if chosen: matched.add(chosen)
        audit.append(dict(group=group,sheet_row=r["row"],recording=r["recording"],sheet_drive_id=key,
                          verified_drive_id=chosen or "",status=("sheet_link_exact" if key==chosen else "folder_filename_recovery" if chosen else "missing"),
                          path=verified[chosen]["path"] if chosen else "",sha256=verified[chosen]["sha256"] if chosen else ""))
    folder_ok=lambda k:k in verified and folder[k]["filename"]==verified[k]["filename"] and folder[k]["group"]==Path(verified[k]["path"]).parent.parent.name
    groups={}
    for group in ("Normal","Abnormal"):
        rows=[r for r in audit if r["group"]==group]
        groups[group]=dict(sheet_rows=len(rows),rows_matched=sum(bool(r["verified_drive_id"]) for r in rows),
                           folder_listed=sum(v["group"]==group for v in folder.values()),
                           folder_verified=sum(v["group"]==group and folder_ok(k) for k,v in folder.items()),
                           extra=[r["source_filename"] for r in files if r["group"]==group and r["drive_id"] not in matched])
    write_csv("P100_verified_files.csv",files);write_csv("BTC_rows_reconciled.csv",audit)
    save(HERE/"FINAL.json",dict(utc=now(),root=str(DEST),workbook_sha256=index["xlsx_sha256"],groups=groups,verified_files=len(files),
            bytes=sum(r["bytes"] for r in files),folder_missing=[v for k,v in folder.items() if not folder_ok(k)],
            rows_missing=[r for r in audit if not r["verified_drive_id"]],recovered=[r for r in audit if r["status"]=="folder_filename_recovery"]))
if __name__=="__main__":
    with (HERE/"reconcile.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
