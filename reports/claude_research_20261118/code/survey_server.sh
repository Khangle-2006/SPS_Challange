#!/usr/bin/env bash
# survey_server.sh -- read-only inventory of the FaceVitals work on the lab server (18 Nov 2026).
#
# What it does : prints hardware, running jobs, Python environments, directory layout, git state,
#                a count of data files by type, a few video container headers, recent experiment
#                folders, checkpoints and recently edited code.
# What it does NOT do: it creates, changes and deletes nothing, opens no data file except video
#                headers through ffprobe, and reads no credential file, shell history or environment
#                variable. Nothing is evaluated; the holdout is only listed like any other folder.
#
# Usage (on the server):   bash survey_server.sh > server_survey.txt
# From Windows cmd.exe  :   ssh khangle@100.114.154.22 "bash -s" < survey_server.sh > server_survey.txt
# Optional              :  ROOTS="/path/a /path/b" bash survey_server.sh > server_survey.txt
#                          (default: every non-hidden top-level folder of $HOME, plus folders owned by
#                           this user on other data mounts). Paths must not contain spaces.
# Runtime               :  usually a few minutes; every scan has its own time limit (about 30 min at most).
export LC_ALL=C GIT_OPTIONAL_LOCKS=0
ME=$(id -un)
KEEP_OUT='-name .git -o -name __pycache__ -o -name node_modules -o -name .venv -o -name venv -o -name site-packages -o -name .ipynb_checkpoints -o -name .cache -o -name wandb'
sec() { printf '\n\n######## %s\n' "$*"; }
run() { timeout "$1" bash -c "$2" 2>/dev/null; rc=$?; [ $rc -eq 124 ] && echo "[stopped after $1 s: listing above is partial]"; return 0; }

# The body runs with stdin closed so that the script can also be fed through "ssh host bash -s < survey_server.sh".
main() {
sec "0. HOST"
date '+%Y-%m-%d %H:%M:%S %Z'; echo "host: $(hostname)   user: $ME   home: $HOME"; uname -sr
(. /etc/os-release 2>/dev/null && echo "os: $PRETTY_NAME"); uptime

sec "1. HARDWARE"
echo "cpus: $(nproc)   $(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/  */ /g')"
free -h | sed -n '1,2p'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv 2>/dev/null
  echo "-- GPU processes (pid, owner, GPU memory, start of command)"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while IFS=, read -r pid mem; do
    printf '%s  %s  %s  %s\n' "$pid" "$(ps -o user= -p "$pid" 2>/dev/null)" "$mem" "$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-90)"
  done
else echo "nvidia-smi: not found"; fi
echo "-- filesystems"
df -hT 2>/dev/null | grep -v -E 'tmpfs|squashfs|overlay|udev|efivarfs|/boot|/snap' | cut -c1-140

sec "2. MY SESSIONS AND JOBS"
echo "-- tmux"; tmux ls 2>&1 | head -20
echo "-- screen"; if command -v screen >/dev/null 2>&1; then screen -ls 2>&1 | head -20; else echo "screen: not installed"; fi
echo "-- my processes (python, matlab, octave, ffmpeg, jupyter, codex)"
ps -u "$ME" -o pid,etime,pcpu,pmem,args 2>/dev/null | grep -E -i 'python|matlab|octave|ffmpeg|jupyter|codex' | grep -v grep | cut -c1-170 | head -40
echo "-- other users with heavy jobs right now (cpu > 50%)"
ps -eo user,pcpu,pmem,etime,comm --sort=-pcpu 2>/dev/null | awk 'NR==1 || $2>50' | head -15

sec "3. SOFTWARE"
for b in python python3 conda mamba pip ffmpeg ffprobe matlab octave git tmux nvcc; do printf '%-8s %s\n' "$b" "$(command -v $b 2>/dev/null || echo -)"; done
python3 --version 2>&1
echo "-- conda environments and the key packages each one has"
ENVS=$( (conda env list 2>/dev/null | awk '!/^#/ && NF {print $NF}'; ls -d "$HOME"/miniconda3 "$HOME"/anaconda3 "$HOME"/miniforge3 "$HOME"/mambaforge "$HOME"/miniconda3/envs/* "$HOME"/anaconda3/envs/* "$HOME"/miniforge3/envs/* "$HOME"/mambaforge/envs/* "$HOME"/.conda/envs/* 2>/dev/null) | sort -u)
for e in $ENVS; do
  [ -d "$e" ] || continue
  sp=$(ls -d "$e"/lib/python3*/site-packages 2>/dev/null | head -1)
  printf '%s  [%s]\n' "$e" "$(basename "$(dirname "$sp")" 2>/dev/null)"
  [ -n "$sp" ] && ls "$sp" 2>/dev/null | grep -E -i '^(torch|torchvision|torchaudio|numpy|scipy|opencv_python|opencv_contrib_python|opencv_python_headless|mediapipe|scikit_learn|pandas|timm|einops|h5py|av|decord|onnxruntime|onnxruntime_gpu|insightface|lightning|pytorch_lightning|matplotlib|neurokit2|heartpy|tensorflow|jax|xgboost|lightgbm|numba|cupy[a-z0-9_]*|facenet_pytorch|dlib|yacs|hydra_core)-[0-9]' | sed -E 's/\.(dist-info|egg-info)$//' | tr '\n' ' ' | fold -s -w 150 | sed 's/^/     /'
  echo
done
echo "-- python3 found on PATH by this (non-interactive) shell: $(command -v python3)"
python3 - <<'PY' 2>/dev/null
import importlib
for m in ["torch", "torchvision", "numpy", "scipy", "cv2", "mediapipe", "sklearn", "pandas", "h5py"]:
    try:
        x = importlib.import_module(m); print("  %-12s %s" % (m, getattr(x, "__version__", "?")))
    except Exception as e:
        print("  %-12s missing" % m)
try:
    import torch; print("  cuda available:", torch.cuda.is_available(), "| torch cuda:", torch.version.cuda, "| devices:", torch.cuda.device_count())
except Exception: pass
PY

sec "4. WHERE THE WORK LIVES"
echo "-- top-level folders of home with size (hidden folders skipped)"
for d in "$HOME"/*/; do [ -d "$d" ] && run 90 "du -xsh '$d' | cut -c1-120"; done
echo "-- folders owned by me on other mounts (depth <= 3)"
OUTSIDE=""
for m in $(df -P 2>/dev/null | awk 'NR>1 {print $6}' | grep -v -E '^/($|boot|snap|run|dev|sys|proc|var/lib/docker)' | sort -u); do
  case "$HOME/" in "$m"/*) continue;; esac
  found=$(timeout 45 find "$m" -maxdepth 3 -type d -user "$ME" -not -path "$HOME/*" 2>/dev/null | sort | awk 'NR==1 || index($0, p"/")!=1 {print; p=$0}' | head -8)
  [ -n "$found" ] && { echo "$found"; OUTSIDE="$OUTSIDE $found"; }
done
if [ -z "${ROOTS:-}" ]; then
  ROOTS=""
  for d in "$HOME"/*/; do
    case "$(basename "$d")" in miniconda3|anaconda3|miniforge3|mambaforge|snap|nltk_data|R|go|Desktop|Downloads|Music|Pictures|Public|Templates|Videos|Documents) continue;; esac
    ROOTS="$ROOTS ${d%/}"
  done
  ROOTS="$ROOTS $OUTSIDE"
fi
echo "-- roots scanned below:"; for r in $ROOTS; do echo "   $r"; done

sec "5. GIT REPOSITORIES"
for g in $(run 90 "find $ROOTS -maxdepth 4 -type d -name .git -prune -print" | head -30); do
  r=$(dirname "$g"); echo; echo "== $r"
  git -C "$r" remote -v 2>/dev/null | sed -E 's#(https?://)[^/@ ]+@#\1***@#' | head -4
  echo "branch: $(git -C "$r" rev-parse --abbrev-ref HEAD 2>/dev/null)   head: $(git -C "$r" log -1 --format='%h %ad %s' --date=short 2>/dev/null | cut -c1-120)"
  git -C "$r" log -12 --format='   %h %ad %s' --date=short 2>/dev/null | cut -c1-130
  n=$(git -C "$r" status --porcelain 2>/dev/null | wc -l); echo "uncommitted paths: $n"
  git -C "$r" status --porcelain 2>/dev/null | head -40 | cut -c1-140
done

sec "6. DIRECTORY LAYOUT (depth <= 3, entries per folder)"
for r in $ROOTS; do
  [ -d "$r" ] || continue; echo; echo "== $r"
  [ $SECONDS -gt 1500 ] && { echo "[time budget of 25 min reached, skipped]"; continue; }
  run 120 "find '$r' -maxdepth 3 \( $KEEP_OUT \) -prune -o -type d -print | sort | head -260 | while IFS= read -r d; do printf '%6s  %s\n' \"\$(ls -A \"\$d\" 2>/dev/null | wc -l)\" \"\$d\"; done"
done

sec "7. DATA FILES BY TYPE (count and size, grouped by folder two levels below each root)"
for r in $ROOTS; do
  [ -d "$r" ] || continue; echo; echo "== $r"
  [ $SECONDS -gt 1500 ] && { echo "[time budget of 25 min reached, skipped]"; continue; }
  run 240 "find '$r' \( $KEEP_OUT \) -prune -o -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.webm' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.bmp' -o -iname '*.npz' -o -iname '*.npy' -o -iname '*.h5' -o -iname '*.hdf5' -o -iname '*.mat' -o -iname '*.csv' -o -iname '*.xlsx' -o -iname '*.json' -o -iname '*.pkl' -o -iname '*.zip' -o -iname '*.tar' -o -iname '*.gz' -o -iname '*.7z' \) -printf '%s\t%P\n' | awk -F'\t' '{n=split(\$2,a,\"/\"); k=(n>2? a[1]\"/\"a[2] : (n>1? a[1] : \".\")); e=tolower(\$2); sub(/.*\./,\"\",e); c[k\"\t\"e]++; s[k\"\t\"e]+=\$1} END {for (i in c) printf \"%10.2f GB %8d  .%-5s %s\n\", s[i]/2^30, c[i], substr(i, index(i,\"\t\")+1), substr(i, 1, index(i,\"\t\")-1)}' | sort -rn | head -70"
done

sec "8. VIDEO CONTAINER HEADERS (one file from up to 14 different folders)"
if command -v ffprobe >/dev/null 2>&1; then
  run 120 "find $ROOTS \( $KEEP_OUT \) -prune -o -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) -print | awk -F/ '{d=\$0; sub(/\/[^\/]*\$/,\"\",d); sub(/\/[^\/]*\$/,\"\",d); if (!(d in s)) {s[d]=1; print}}' | head -14" | while IFS= read -r f; do
    [ -f "$f" ] || { echo "$f"; continue; }
    echo "$f"
    timeout 25 ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,bit_rate,color_transfer,color_primaries,color_space:format=duration,size -of default=nw=1 "$f" 2>/dev/null | tr '\n' ' '; echo
  done
else echo "ffprobe: not found"; fi

sec "9. EXPERIMENT FOLDERS TOUCHED IN THE LAST 21 DAYS (newest first)"
run 180 "find $ROOTS -maxdepth 6 \( $KEEP_OUT \) -prune -o -type d -mtime -21 -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort -r | head -150"

sec "10. CHECKPOINTS (newest first)"
run 180 "find $ROOTS \( $KEEP_OUT \) -prune -o -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.onnx' -o -name '*.safetensors' \) -printf '%TY-%Tm-%Td %TH:%TM %10s  %p\n' | sort -r | head -90"

sec "11. REPORTS AND RESULT TABLES TOUCHED IN THE LAST 21 DAYS (newest first)"
run 180 "find $ROOTS \( $KEEP_OUT \) -prune -o -type f -mtime -21 -size -20M \( -iname '*.md' -o -iname '*.log' -o \( \( -iname '*report*' -o -iname '*result*' -o -iname '*metric*' -o -iname '*summary*' -o -iname '*score*' -o -iname '*prediction*' \) -a \( -iname '*.csv' -o -iname '*.json' -o -iname '*.txt' -o -iname '*.tsv' -o -iname '*.html' \) \) \) -printf '%TY-%Tm-%Td %TH:%TM %9s  %p\n' | sort -r | head -220"

sec "12. CODE EDITED IN THE LAST 14 DAYS (newest first)"
run 180 "find $ROOTS \( $KEEP_OUT \) -prune -o -type f -mtime -14 \( -name '*.py' -o -name '*.m' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' -o -name '*.ipynb' \) -printf '%TY-%Tm-%Td %TH:%TM %8s  %p\n' | sort -r | head -260"

sec "END"
echo "finished after $SECONDS s"
}
main </dev/null
