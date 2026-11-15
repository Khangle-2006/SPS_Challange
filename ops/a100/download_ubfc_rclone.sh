#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/data/khangle/facevitals
RCLONE="$ROOT/tools/rclone-v1.75.2-linux-amd64/rclone"
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache"
mkdir -p "$ROOT/cache" "$ROOT/logs" "$ROOT/datasets/UBFC-rPPG/DATASET_2"
exec 9>"$ROOT/logs/ubfc_rclone.lock"
flock -n 9 || { echo 'UBFC downloader is already running'; exit 1; }
CFG="$ROOT/config/rclone.conf"
DEST="$ROOT/datasets/UBFC-rPPG/DATASET_2"
"$RCLONE" --config "$CFG" lsf ubfc: --dirs-only > "$ROOT/logs/ubfc_remote_subjects.txt"
# Trial one video before requesting all 84 files; stop if Google rejects it.
"$RCLONE" --config "$CFG" copy ubfc: "$DEST" --checksum --transfers 1 --checkers 1 --multi-thread-streams 0 --tpslimit 1 --tpslimit-burst 1 --retries 3 --retries-sleep 60s --low-level-retries 3 --stats 30s --log-file "$ROOT/logs/ubfc_rclone.log" --log-level INFO
"$RCLONE" --config "$CFG" check ubfc: "$DEST" --one-way --combined "$ROOT/logs/ubfc_rclone_check.txt"
echo 'Download and remote checksum check completed.'
