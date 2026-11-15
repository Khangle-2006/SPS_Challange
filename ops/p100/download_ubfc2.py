"""Download only official UBFC Dataset 2; resumable, atomic completion, SHA256 inventory."""
import concurrent.futures, hashlib, json, os, re, time
from pathlib import Path
import gdown
ROOT=Path("/data_pool/khangle/facevitals")
OUT=ROOT/"datasets/UBFC-rPPG/DATASET_2"
items=[dict(id=a,path=b) for a,b in re.findall(r"GoogleDriveFileToDownload\(id='([^']+)', path='([^']+)'", (ROOT/"logs/drive_listing.txt").read_text())]
assert len(items)==84, len(items)
assert len({x["path"].split("/")[0] for x in items})==42
for x in items:
    assert re.fullmatch(r"subject\d+/(vid\.avi|ground_truth\.txt)",x["path"])
(ROOT/"datasets/UBFC-rPPG/drive_manifest.json").parent.mkdir(parents=True,exist_ok=True)
(ROOT/"datasets/UBFC-rPPG/drive_manifest.json").write_text(json.dumps(items,indent=2))
def run(item):
    dest=OUT/item["path"]
    dest.parent.mkdir(parents=True,exist_ok=True)
    marker=dest.with_name(dest.name+".complete.json")
    if marker.exists() and dest.exists() and dest.stat().st_size==json.loads(marker.read_text())["bytes"]:
        return {"path":item["path"],"status":"already_complete"}
    last=None
    for attempt in range(3):
        try:
            if not dest.exists():
                result=gdown.download(id=item["id"],output=str(dest)+".part",quiet=True,resume=True,timeout=(20,120),retries=2,use_cookies=False)
                if result is None: raise RuntimeError("Download returned None")
                os.replace(str(dest)+".part",dest)
            if dest.name=="vid.avi":
                with dest.open("rb") as f: head=f.read(12)
                assert head[:4]==b"RIFF" and head[8:12]==b"AVI ",repr(head)
            else:
                lines=dest.read_text().strip().splitlines()
                assert len(lines)==3, len(lines)
            sha=hashlib.sha256()
            with dest.open("rb") as f:
                for block in iter(lambda:f.read(8*1024*1024),b""): sha.update(block)
            record={**item,"bytes":dest.stat().st_size,"sha256":sha.hexdigest(),"completed_unix":time.time()}
            marker.write_text(json.dumps(record,indent=2))
            print(json.dumps({"status":"complete",**record}),flush=True)
            return record
        except Exception as exc:
            last=str(exc)
            print(json.dumps({"status":"retry","path":item["path"],"attempt":attempt+1,"error":last}),flush=True)
            if "Too many users" in last: break
            time.sleep(5*(attempt+1))
    return {**item,"error":last}
if __name__=="__main__":
    print(json.dumps({"status":"start","files":len(items),"subjects":42}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run,sorted(items,key=lambda x:(int(re.search(r"\d+",x["path"])[0]),x["path"]))))
    (ROOT/"logs/download_result.json").write_text(json.dumps(results,indent=2))
    failures=[x for x in results if "error" in x]
    print(json.dumps({"status":"finished","failed":len(failures),"total":len(results)}),flush=True)
    raise SystemExit(bool(failures))

