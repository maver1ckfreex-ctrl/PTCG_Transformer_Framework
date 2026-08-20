#!/usr/bin/env bash
#
# DATA-TRANSFER MODE -- RECEIVER side (the GPU box).
#
#   wait for the purser's npz  ->  verify each one  ->  train
#
# It derives the expected npz paths from ITS OWN training config, using the
# same data_paths.sh the purser used. So it knows exactly what to wait for
# without being told. Polls every 30s until everything has landed and
# verified, then hands off to run_dual.sh / run_bulk.sh with
# SKIP_DOWNLOAD=1 -- those skip conversion when the npz already exists, so
# the GPU starts immediately and never parses a replay.
#
# Usage (dual GPU):
#   DATA_TRANSFER=1 PURSER=0 RECEIVER=1 \
#   DATA_DIR=~/raw DUAL_GPU=1 GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b \
#   EVAL_DECK=../submission_r2_t07_torch/deck.csv \
#   ./transfer/run_receiver.sh
#
# Usage (single GPU):
#   DATA_TRANSFER=1 PURSER=0 RECEIVER=1 \
#   DATA_DIR=~/raw DUAL_GPU=0 TRAINER=v3a \
#   EVAL_DECK=../submission_r2_t07_torch/deck.csv \
#   ./transfer/run_receiver.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# PY: interpreter to use (a conda env python, say). Defaults to python3.
PY="${PY:-python3}"
. "$HERE/data_paths.sh"

DATA_TRANSFER="${DATA_TRANSFER:-0}"
PURSER="${PURSER:-0}"
RECEIVER="${RECEIVER:-0}"
if [ "$DATA_TRANSFER" != "1" ] || [ "$RECEIVER" != "1" ]; then
    echo "This script is the RECEIVER half of data-transfer mode." >&2
    echo "Run with: DATA_TRANSFER=1 PURSER=0 RECEIVER=1 ..." >&2
    exit 1
fi
[ "$PURSER" = "1" ] && {
    echo "ERROR: a box is either PURSER or RECEIVER, not both" >&2; exit 1; }

DATA_DIR="${DATA_DIR:?set DATA_DIR=/path/where/the/npz/will/land}"
DUAL_GPU="${DUAL_GPU:-0}"
POLL="${POLL:-30}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-0}"      # 0 = wait forever
mkdir -p "$DATA_DIR"

compute_datasets
echo "================================================================"
echo " DATA-TRANSFER : RECEIVER"
echo " data dir  : $DATA_DIR"
echo " mode      : $([ "$DUAL_GPU" = 1 ] && echo "dual GPU" || echo "single GPU")"
echo " waiting for:"
echo "$DATASETS" | while IFS='|' read -r rel spec label; do
    echo "   $label -> $DATA_DIR/$rel   ($spec)"
done
echo " poll      : every ${POLL}s$([ "$WAIT_TIMEOUT" -gt 0 ] && echo ", timeout ${WAIT_TIMEOUT}s")"
echo "================================================================"

# ---- wait until every dataset has landed AND verifies -------------------
# The purser scp's to <name>.incoming and renames on arrival, so the final
# name appearing means the whole file is here. Verification is still run:
# a rename cannot catch a dataset that was built wrong.
t0=$(date +%s)
while :; do
    missing=0 ; bad=0 ; ready=""
    while IFS='|' read -r rel spec label; do
        f="$DATA_DIR/$rel"
        if [ ! -f "$f" ]; then
            missing=$((missing + 1))
            continue
        fi
        v=(); [ "$spec" = "winners" ] && v=(--winners-only)
        if "$PY" "$ROOT/tools/verify_data.py" "$f" ${v[@]+"${v[@]}"} \
                --label "$label" > /tmp/.recv_verify.$$ 2>&1; then
            ready="$ready $label"
        else
            bad=$((bad + 1))
            echo "--- $label present but FAILED verification ---"
            cat /tmp/.recv_verify.$$
        fi
    done <<< "$DATASETS"
    rm -f /tmp/.recv_verify.$$

    if [ "$missing" -eq 0 ] && [ "$bad" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] all datasets present and verified:$ready"
        break
    fi

    el=$(( $(date +%s) - t0 ))
    if [ "$WAIT_TIMEOUT" -gt 0 ] && [ "$el" -ge "$WAIT_TIMEOUT" ]; then
        echo "ERROR: timed out after ${el}s waiting for data" >&2
        exit 1
    fi
    inc=$(find "$DATA_DIR" -name '*.npz.incoming' 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] waiting: $missing missing, $bad bad" \
         "$([ "$inc" -gt 0 ] && echo "($inc transfer in flight)")" \
         "| ${el}s elapsed"
    sleep "$POLL"
done

# ---- verified -> train --------------------------------------------------
# SKIP_DOWNLOAD=1 and the npz already in place, so run_bulk / run_dual go
# straight to training: no download, no conversion, GPU busy immediately.
echo
if [ "$DUAL_GPU" = "1" ]; then
    echo "--- handing off to run_dual.sh ---"
    exec env SKIP_DOWNLOAD=1 DATA_DIR="$DATA_DIR" DUAL_GPU=1 \
        bash "$ROOT/bulk/run_dual.sh"
else
    echo "--- handing off to run_bulk.sh ---"
    exec env SKIP_DOWNLOAD=1 DATA_DIR="$DATA_DIR" \
        bash "$ROOT/bulk/run_bulk.sh"
fi
