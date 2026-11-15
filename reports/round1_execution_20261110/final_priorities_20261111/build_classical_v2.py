from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'classical_screen.py').read_text()
s=s.replace("a=list(csv.DictReader(file.open()));return", """if not file.exists():
            video=Path(j['path']);assert sha(video)==j['sha256']
            g=frames(video);target=OUT/'additional_gop'/f"{j['recording']}.json"
            save(target,dict(source_sha256=j['sha256'],frames=g,reason='Existing native GOP inventory did not include this dev recording'))
            return [v['pict_type'] for v in g],dict(path=str(target),sha256=sha(target),source_sha256=j['sha256'])
        a=list(csv.DictReader(file.open()));return""")
s=s.replace("assert not (OUT/'LAUNCH.json').exists()","assert not (OUT/'LAUNCH_V2.json').exists();assert not Path('/proc/2608808').exists()")
s=s.replace("OUT/'LAUNCH.json'","OUT/'LAUNCH_V2.json'").replace("OUT/'ERROR.json'","OUT/'ERROR_V2.json'")
s=s.replace("        a=one(j,c,d,windows);save", """        target=OUT/'predictions'/j['dataset']/j['recording']/(c+'.json')
        if target.exists():
            old=read(target);assert old['meta_sha256']==sha(d/'meta.npz') and old['nuisance_sha256']==sha(P/'nuisance.py')
            assert old['code_sha256'] in [sha(P/'classical_screen.py'),sha(__file__)]
            continue
        a=one(j,c,d,windows);save""")
s=s.replace('    report()\n',"    report()\n    save(OUT/'COMPLETE.json',dict(utc=guard.now(),record_conditions=len(jobs),holdout_evaluations=0))\n")
(P/'classical_screen_v2.py').write_text(s)
