"""Fresh interpreter directory with explicit, unchanged installed dependency paths."""
import json,sys,venv,subprocess,sysconfig
from pathlib import Path
out=Path(sys.argv[1]);deps=list(map(Path,sys.argv[2:]))
assert not out.exists(),'Never replace an environment'
venv.EnvBuilder(with_pip=False,system_site_packages=False).create(out)
python=out/'bin/python'
site=Path(subprocess.check_output([str(python),'-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip())
paths=[subprocess.check_output([str(p/'bin/python'),'-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip() for p in deps]
paths.append('/usr/lib/python3/dist-packages')
(site/'frozen_dependency_paths.pth').write_text('\n'.join(paths)+'\n')
probe='import json,sys,numpy,cv2,torch,av,h5py,hdf5storage,mat73,mediapipe;print(json.dumps(dict(prefix=sys.prefix,python=sys.version,versions={m.__name__:getattr(m,"__version__","unknown") for m in [numpy,cv2,torch,av,h5py,hdf5storage,mat73,mediapipe]})))'
r=subprocess.run([str(python),'-c',probe],capture_output=True,text=True)
(out.parent/'FRESH_VENV.json').write_text(json.dumps(dict(environment=str(out),fresh_directory=True,isolated_dependency_install=False,installed_dependency_paths=paths,probe_exit=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2))
print(r.stdout,r.stderr);sys.exit(r.returncode)
