import concurrent.futures, hashlib, json, os, time
from pathlib import Path
import requests
ROOT=Path("/data_pool/khangle/facevitals")
manifest=json.loads((ROOT/"logs/aipex_tree.json").read_text())
commit=manifest["resolved_commit"]
files=[x for x in manifest["tree"] if x["type"]=="blob" and (x["path"].startswith(("Data/","Ground_Truth/")) or x["path"] in ("README.md","LICENSE"))]
destroot=ROOT/"datasets/AiPEX"
destroot.mkdir(parents=True,exist_ok=True)
(destroot/"source_manifest.json").write_text(json.dumps({"commit":commit,"tree_sha":manifest["resolved_tree_sha"],"files":files},indent=2))
def download(x):
    path=destroot/x["path"]; path.parent.mkdir(parents=True,exist_ok=True)
    mark=path.with_name(path.name+".complete.json")
    if path.exists() and mark.exists() and path.stat().st_size==x["size"]: return
    url="https://raw.githubusercontent.com/AiPEX-Lab/rppg_biases/"+commit+"/"+x["path"]
    for attempt in range(3):
        try:
            gitsha=hashlib.sha1(("blob "+str(x["size"])+"\0").encode())
            sha=hashlib.sha256()
            with requests.get(url,stream=True,timeout=(20,120)) as r:
                r.raise_for_status()
                with open(str(path)+".part","wb") as out:
                    for chunk in r.iter_content(1024*1024):
                        out.write(chunk); gitsha.update(chunk); sha.update(chunk)
            assert gitsha.hexdigest()==x["sha"],"Git blob checksum mismatch"
            os.replace(str(path)+".part",path)
            mark.write_text(json.dumps({"sha256":sha.hexdigest(),"git_blob_sha1":x["sha"],"bytes":x["size"]}))
            print("OK",x["path"],x["size"],flush=True)
            return
        except Exception as exc:
            print("RETRY",x["path"],repr(exc),flush=True); time.sleep(5)
    raise RuntimeError(x["path"])
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(download,sorted(files,key=lambda x:(not x["path"].startswith("Ground_Truth/"),x["size"]))))
print("COMPLETE",len(files),flush=True)

