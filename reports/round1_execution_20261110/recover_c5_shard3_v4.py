"""Inspected shard-only continuation; original video retained, partial crop archived."""
import traceback
from c5prime_recovery_v4 import *
from runtime_shared_v4 import verify_manifest

def main():
    lease=lock('recover_c5_shard3_v4.lock');verify_manifest();guard(True)
    assert not Path('/proc/3646012').exists() and not Path('/proc/3857096').exists()
    assert not (P/'recovery_v4/C5_STARTED.json').exists()
    d=P/'c5prime/MCD/3401_FullHDwebcam_after/bf0'
    j=next(j for j in jobs() if j['dataset']=='MCD' and j['recording']=='3401_FullHDwebcam_after')
    assert sha(j['source'])==j['source_sha256']
    # Verify the retained encode decodes to the clock already saved before crop failed.
    encoded=pre.Video(d/'video.mp4',P/'recovery_v4/encoded_verify.log')
    with np.load(d/'clock.npz') as f:
        assert len(encoded.t)==len(f['source_t'])==len(f['encoded_t'])
        assert np.allclose(encoded.t,f['encoded_t'],rtol=0,atol=1e-9)
    probe=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(d/'video.mp4'),'-f','null','-'],capture_output=True,text=True)
    assert probe.returncode==0 and not probe.stderr,('Encoded video decode failed',probe.stderr)
    archive=d/'crop_partial_20261110T1050Z'
    assert d.resolve().is_relative_to(P.resolve()) and not archive.exists()
    partial={str(p.relative_to(d/'crop')):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (d/'crop').rglob('*') if p.is_file()}
    (d/'crop').rename(archive)
    save(P/'recovery_v4/ENCODED_PARTIAL_APPROVED.json',dict(utc=now(),directory=str(d),source_sha256=j['source_sha256'],encoded_sha256=sha(d/'video.mp4'),clock_sha256=sha(d/'clock.npz'),preserved_crop=str(archive),partial_files=partial,reason='Guard exception before crop completion; encoded video and source clock verified; re-run crop only'))
    save(P/'recovery_v4/C5_STARTED.json',dict(utc=now(),pid=os.getpid(),code_sha256=sha(__file__)))
    todo=[(j,v) for j in jobs() for v in ['default','bf0']]
    for i,(j,v) in enumerate(todo):
        if i%4!=3:continue
        one(j,v)
        save(P/'recovery_v4/C5_STATUS.json',dict(utc=now(),pid=os.getpid(),index=i,total=750,recording=j['recording'],variant=v))
    rows=[]
    for j,v in todo:
        a=read(P/'c5prime'/j['dataset']/j['recording']/v/'COMPLETE.json')
        assert a['source_sha256']==j['source_sha256']
        rows.append(dict(dataset=j['dataset'],recording=j['recording'],variant=v,state=a['state'],bitrate_pass=a.get('bitrate_pass')))
    assert len(rows)==750
    save(P/'C5_COMPLETE.json',dict(utc=now(),rows=rows,holdout_evaluations=0,operational_recovery='shard3 only, encoded source retained'))
    save(P/'recovery_v4/C5_COMPLETE.json',dict(utc=now(),pid=os.getpid()))

if __name__=='__main__':
    try:main()
    except Exception as e:save(P/'recovery_v4/C5_ERROR.json',dict(utc=now(),error=str(e),traceback=traceback.format_exc()));raise
