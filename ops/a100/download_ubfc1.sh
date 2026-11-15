#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/data/khangle/facevitals
RCLONE="$ROOT/tools/rclone-v1.75.2-linux-amd64/rclone"
CFG="${RCLONE_CONFIG_PATH:-$ROOT/config/rclone.conf}"
REMOTE="${RCLONE_REMOTE:-ubfc:}"
DEST="$ROOT/datasets/UBFC-rPPG/DATASET_1"
FOLDER=1oD7Szb9wE6XbUN9SWq6EmyH9WDnKI1Pk
mkdir -p "$DEST" "$ROOT/logs"
exec 9>"$ROOT/logs/ubfc1_download.lock"
flock -n 9 || exit 1
COMMON=(--config "$CFG" --drive-root-folder-id "$FOLDER" --tpslimit 1
        --contimeout 20s --timeout 120s --retries 3 --retries-sleep 60s --low-level-retries 3)
"$RCLONE" "${COMMON[@]}" lsjson "$REMOTE" --recursive --hash > "$ROOT/logs/ubfc1_remote_manifest.json"
"$RCLONE" "${COMMON[@]}" copy "$REMOTE" "$DEST" --checksum --transfers 1 --checkers 1 \
    --multi-thread-streams 0 --stats 60s --log-level INFO --log-file "$ROOT/logs/ubfc1_download.log"
"$RCLONE" "${COMMON[@]}" check "$REMOTE" "$DEST" --one-way --combined "$ROOT/logs/ubfc1_checksum_check.txt"
printf '{"download_complete":true,"remote_checksum_check_passed":true}\n' > "$ROOT/logs/ubfc1_COMPLETE.json"
