#!/usr/bin/env bash
#
# DATA-TRANSFER MODE -- PURSER side (CPU-only box, big disk, no GPU).
#
#   download kaggle  ->  [optional self-play]  ->  convert with ALL cores
#   ->  verify  ->  scp ONLY the converted npz to the trainer
#
# Raw replays are NEVER transferred. They stay here. The trainer receives
# one .npz per dataset -- a few GB instead of ~840 GB.
#
# Nothing here is capped: conversion and self-play both get every core.
#
# Usage:
#   DATA_TRANSFER=1 PURSER=1 RECEIVER=0 \
#   DEST=user@gpu-box:/home/user/raw \
#   DATA_DIR=~/raw DAYS=20 DUAL_GPU=1 \
#   GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b \
#   ./transfer/run_purser.sh
#
# Single-GPU target instead:
#   DUAL_GPU=0 TRAINER=v3a ...
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
. "$HERE/data_paths.sh"

DATA_TRANSFER="${DATA_TRANSFER:-0}"
PURSER="${PURSER:-0}"
RECEIVER="${RECEIVER:-0}"
if [ "$DATA_TRANSFER" != "1" ] || [ "$PURSER" != "1" ]; then
    echo "This script is the PURSER half of data-transfer mode." >&2
    echo "Run with: DATA_TRANSFER=1 PURSER=1 RECEIVER=0 ..." >&2
    exit 1
fi
[ "$RECEIVER" = "1" ] && {
    echo "ERROR: a box is either PURSER or RECEIVER, not both" >&2; exit 1; }

DATA_DIR="${DATA_DIR:?set DATA_DIR=/path/with/room/for/raw/replays}"

# SEND=0: build and verify the datasets, then STOP -- do not push.
# For receivers whose provider blocks inbound ports (no TCP endpoint), the
# GPU box pulls instead. DEST is not needed in that case.
SEND="${SEND:-1}"
if [ "$SEND" = "1" ]; then
    DEST="${DEST:?set DEST=user@host:/remote/dir, or SEND=0 to build only}"
else
    DEST="${DEST:-}"
fi
DUAL_GPU="${DUAL_GPU:-0}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
# Days per parse cycle. Only this many days of raw JSON exist at once, so
# peak disk is ~SPLIT_DAYS x 21.5 GB rather than the whole corpus. A final
# short cycle is fine.
PURSING_SPLIT="${PURSING_SPLIT:-0}"
SPLIT_DAYS="${SPLIT_DAYS:-20}"
DAYS="${DAYS:-}"
SEED="${SEED:-20260805}"

MIX_V3A="${MIX_V3A:-0}"
BUILDER="${BUILDER:-}"
BUILDER_TEMP="${BUILDER_TEMP:-0.7,0.8,0.9}"
PLAYER_TEMP="${PLAYER_TEMP:-0.0,0.1,0.2}"
NUM_DECK="${NUM_DECK:-2000}"
NUM_SELFPLAY="${NUM_SELFPLAY:-10000}"
SP_PILOT="${SP_PILOT:-$ROOT/../submission_r2_t07_torch/player.pt}"
ENGINE="${ENGINE:-$ROOT/../submission_r2_t07_torch}"

TOTAL_CORES="${TOTAL_CORES:-$( (command -v nproc >/dev/null && nproc) \
    || sysctl -n hw.ncpu 2>/dev/null || echo 8 )}"

# PY lets you point at a specific interpreter, e.g. a conda env:
#   PY=/opt/conda/envs/model_training/bin/python ./transfer/run_purser.sh
PY="${PY:-python3}"
# parallel member-extraction width. Defaults to the core count.
EXTRACT_JOBS="${EXTRACT_JOBS:-$TOTAL_CORES}"
command -v "$PY" >/dev/null || {
    echo "ERROR: interpreter '$PY' not found" >&2; exit 1; }
"$PY" -c 'import numpy' 2>/dev/null || {
    echo "ERROR: '$PY' cannot import numpy." >&2
    echo "  Activate the env first, or pass one explicitly:" >&2
    echo "    PY=/opt/conda/envs/<env>/bin/python $0 ..." >&2
    exit 1; }

DEST_HOST="${DEST%%:*}"
DEST_PATH="${DEST#*:}"
if [ "$SEND" = "1" ] && [ "$DEST_HOST" = "$DEST" ]; then
    echo "ERROR: DEST must be user@host:/remote/path" >&2; exit 1
fi

SSH_OPTS=(-p "$SSH_PORT")
SCP_OPTS=(-P "$SSH_PORT")
if [ -n "$SSH_KEY" ]; then SSH_OPTS+=(-i "$SSH_KEY"); SCP_OPTS+=(-i "$SSH_KEY"); fi

if [ -n "$DAYS" ]; then
    PICK_ARGS=(--days "$DAYS" --seed "$SEED")
    PICK_LABEL="$DAYS random days (seed $SEED)"
else
    PICK_ARGS=(--all)
    PICK_LABEL="ALL days in the manifest"
fi

REPLAYS="$DATA_DIR/replays"
SELFPLAY_DIR="$DATA_DIR/replays_selfplay"
mkdir -p "$REPLAYS"

compute_datasets
echo "================================================================"
echo " DATA-TRANSFER : PURSER"
echo " local raw     : $REPLAYS   ($PICK_LABEL)"
echo " cores         : $TOTAL_CORES  (uncapped)"
if [ "$PURSING_SPLIT" = "1" ]; then
    echo " split parsing : ON, $SPLIT_DAYS days per cycle "\
         "(raw deleted after each verified checkpoint)"
else
    echo " split parsing : off (extract everything, one pass)"
fi
echo " destination   : ${DEST:-<none: SEND=0, receiver pulls>}"
echo " target mode   : $([ "$DUAL_GPU" = 1 ] && echo "dual GPU" || echo "single GPU")"
echo " datasets to build and send:"
echo "$DATASETS" | while IFS='|' read -r rel spec label; do
    echo "   $label -> $rel   ($spec)"
done
[ "$MIX_V3A" = "1" ] && echo " MIX_V3A       : ON (self-play folded into v3a)"
echo " raw replays are NOT transferred"
echo "================================================================"

# ---- 0. fail fast on ssh before doing hours of work ---------------------
if [ "$SEND" = "1" ]; then
echo "--- checking ssh to $DEST_HOST ---"
if ! ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=10 \
        "$DEST_HOST" "mkdir -p '$DEST_PATH' && echo ok" 2>/dev/null; then
    echo "ERROR: cannot ssh to $DEST_HOST without a password." >&2
    echo "  Set up a key first (see transfer/README_TRANSFER.md):" >&2
    echo "     ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519" >&2
    echo "     ssh-copy-id -p $SSH_PORT $DEST_HOST" >&2
    exit 1
fi
echo "ssh ok, remote dir ready"
else
    echo "--- SEND=0: building only, the receiver will pull ---"
fi

# ---- 1. download ALL zips, sequentially, no extraction -----------------
# Extraction is disk-write bound (~1.1 TB for 54 days) and the sequential
# per-day loop already saturates the disk, so nothing is gained by doing
# days in parallel. Downloads are network bound and stay sequential too.
ZIPS="$DATA_DIR/_zips"
MARK="$DATA_DIR/.state"
PARTS="$DATA_DIR/_parts"
mkdir -p "$ZIPS" "$MARK" "$PARTS"

"$PY" "$ROOT/pick_days.py" "${PICK_ARGS[@]}" --emit-list \
    > "$MARK/daylist.tsv" || exit 1
TOTAL_DAYS=$(wc -l < "$MARK/daylist.tsv" | tr -d ' ')

if [ "$SKIP_DOWNLOAD" != "1" ]; then
    echo "--- [1/5] downloading $TOTAL_DAYS zips (sequential, no unzip) ---"
    i=0
    while IFS=$'\t' read -r date slug eps; do
        i=$((i + 1))
        if [ -f "$MARK/$slug.zip_done" ]; then
            echo "  [$i/$TOTAL_DAYS] $date  zip present"
            continue
        fi
        echo "  [$i/$TOTAL_DAYS] $date  downloading ($eps episodes)"
        if kaggle datasets download -d "kaggle/$slug" -p "$ZIPS" \
                > "$MARK/$slug.dl.log" 2>&1; then
            touch "$MARK/$slug.zip_done"
        else
            echo "ERROR: download failed for $slug (see $MARK/$slug.dl.log)" >&2
            exit 1
        fi
    done < "$MARK/daylist.tsv"
else
    echo "--- [1/5] SKIP_DOWNLOAD=1, using zips already in $ZIPS ---"
fi
echo "zips on disk: $(ls -1 "$ZIPS"/*.zip 2>/dev/null | wc -l | tr -d ' ')"

# ---- 2. chunked: extract N days -> convert -> checkpoint -> delete ------
# Only SPLIT_DAYS days of raw JSON exist at any moment, so peak disk is
# ~SPLIT_DAYS x 21.5 GB instead of the full corpus. Each chunk's npz is
# written and verified BEFORE its raw data is deleted, so a crash costs
# one chunk, never the whole run.
if [ "$PURSING_SPLIT" != "1" ]; then
    SPLIT_DAYS=$TOTAL_DAYS          # one cycle = everything, no deletion
    echo "--- [2/5] parsing in ONE pass (PURSING_SPLIT=0) ---"
else
    echo "--- [2/5] split parsing, $SPLIT_DAYS days per cycle ---"
fi
# portable equivalent of mapfile (bash 3.2 has no mapfile)
ALL_DAYS=()
while IFS= read -r _line; do
    [ -n "$_line" ] && ALL_DAYS+=("$_line")
done < "$MARK/daylist.tsv"
NDAYS=${#ALL_DAYS[@]}
CHUNK=0
for (( off=0; off<NDAYS; off+=SPLIT_DAYS )); do
    CHUNK=$((CHUNK + 1))
    SLICE=("${ALL_DAYS[@]:off:SPLIT_DAYS}")
    TAG=$(printf 'chunk_%03d' "$CHUNK")

    # every dataset already has a part for this chunk -> nothing to do
    need=0
    while IFS='|' read -r rel spec label; do
        base=$(printf '%s' "${rel%.npz}" | tr '/' '_')
        [ -f "$PARTS/$base/$TAG.npz" ] || need=1
    done <<< "$DATASETS"
    if [ "$need" -eq 0 ]; then
        echo "  $TAG: all parts present, skipping ${#SLICE[@]} days"
        continue
    fi

    if [ -f "$REPLAYS/.extracted_$TAG" ]; then
        echo "  $TAG: raw json already extracted, reusing"
    else
    echo "  $TAG: extracting ${#SLICE[@]} days -> $REPLAYS"
    rm -rf "$REPLAYS"; mkdir -p "$REPLAYS"
    # Extract MEMBERS in parallel, not whole zips. One `unzip archive.zip`
    # is a single thread walking ~5000 members; listing the members and
    # extracting them concurrently is ~10s/day instead of ~40s. GNU
    # parallel if present, xargs -P otherwise (same thing, always there).
    for line in "${SLICE[@]}"; do
        IFS=$'\t' read -r date slug eps <<< "$line"
        z="$ZIPS/$slug.zip"
        [ -f "$z" ] || { echo "ERROR: missing $z" >&2; exit 1; }
        # GNU parallel preferred (this is the measured-fast path); xargs -P
        # when it is absent. NOTE: parallel calls `ps` for load checks and
        # prints "ps: command not found" on minimal images -- that is
        # cosmetic, extraction proceeds normally.
        if ! command -v unzip >/dev/null; then
            "$PY" -m zipfile -e "$z" "$REPLAYS" || exit 1
        elif command -v parallel >/dev/null; then
            unzip -Z1 "$z" | parallel -j "$EXTRACT_JOBS" --will-cite \
                unzip -qq -o "$z" {} -d "$REPLAYS" || exit 1
        else
            unzip -Z1 "$z" | xargs -P "$EXTRACT_JOBS" -I@ \
                unzip -qq -o "$z" "@" -d "$REPLAYS" || exit 1
        fi
    done
    touch "$REPLAYS/.extracted_$TAG"
    echo "  $TAG: $(find "$REPLAYS" -name '*.json' | wc -l | tr -d ' ') json extracted"
    fi

    while IFS='|' read -r rel spec label; do
        base=$(printf '%s' "${rel%.npz}" | tr '/' '_')
        mkdir -p "$PARTS/$base"
        part="$PARTS/$base/$TAG.npz"
        [ -f "$part" ] && { echo "  $TAG/$label: part exists"; continue; }
        extra=(); [ "$spec" = "winners" ] && extra=(--winners-only)
        srcs=("$REPLAYS")
        case "$rel" in *_mix.npz) srcs+=("$SELFPLAY_DIR") ;; esac
        echo "  $TAG/$label: converting ($spec) with $TOTAL_CORES workers"
        tmp="$PARTS/$base/.${TAG}_writing.npz"   # must end .npz: numpy
        rm -f "$tmp"                             # appends it otherwise
        "$PY" "$ROOT/sequence/replay_to_trajectories.py" \
            --replays "${srcs[@]}" --out "$tmp" \
            --workers "$TOTAL_CORES" ${extra[@]+"${extra[@]}"} || exit 1
        v=(); [ "$spec" = "winners" ] && v=(--winners-only)
        "$PY" "$ROOT/tools/verify_data.py" "$tmp" \
            ${v[@]+"${v[@]}"} --label "$TAG/$label" || {
            echo "ABORT: $TAG/$label failed verification; raw data KEPT" >&2
            exit 1; }
        mv "$tmp" "$part"            # checkpoint is safe only once verified
    done <<< "$DATASETS"

    if [ "$PURSING_SPLIT" = "1" ]; then
        echo "  $TAG: checkpoints verified, deleting this cycle's raw json"
        rm -rf "$REPLAYS"; mkdir -p "$REPLAYS"
    else
        echo "  $TAG: done (raw kept, PURSING_SPLIT=0)"
    fi
done

# ---- 3b. merge the per-chunk checkpoints -------------------------------
echo "--- [3/5] merging chunk checkpoints ---"
while IFS='|' read -r rel spec label; do
    npz="$DATA_DIR/$rel"
    base=$(printf '%s' "${rel%.npz}" | tr '/' '_')
    mkdir -p "$(dirname "$npz")"
    if [ -f "$npz" ]; then
        echo "  $label: $rel exists, skipping merge"
        continue
    fi
    "$PY" "$ROOT/tools/merge_npz.py" --out "$npz" \
        --glob "$PARTS/$base/chunk_*.npz" || exit 1
done <<< "$DATASETS"

# ---- 3. verify before sending ------------------------------------------
echo "--- [4/5] verifying final datasets ---"
FAIL=0
while IFS='|' read -r rel spec label; do
    v=(); [ "$spec" = "winners" ] && v=(--winners-only)
    "$PY" "$ROOT/tools/verify_data.py" "$DATA_DIR/$rel" \
        ${v[@]+"${v[@]}"} --label "$label" || FAIL=1
done <<< "$DATASETS"
[ "$FAIL" -eq 0 ] || { echo "ABORT: bad dataset, nothing sent" >&2; exit 1; }

# ---- 4. transfer, atomically -------------------------------------------
# scp to a .incoming name, then rename on the far side. The receiver only
# ever sees the final name once the whole file has landed, so it can never
# start training on a partial npz.
if [ "$SEND" != "1" ]; then
    echo
    echo "=== PURSER DONE (SEND=0) ==="
    echo "  datasets are built and verified, waiting to be pulled:"
    while IFS='|' read -r rel spec label; do
        echo "    $DATA_DIR/$rel   ($(du -h "$DATA_DIR/$rel" | cut -f1))"
    done <<< "$DATASETS"
    echo "  raw replays stayed here: $REPLAYS"
    exit 0
fi

echo "--- [5/5] transferring to $DEST ---"
while IFS='|' read -r rel spec label; do
    src="$DATA_DIR/$rel"
    dst="$DEST_PATH/$rel"
    sz=$(du -h "$src" | cut -f1)
    echo "  $label: $rel ($sz) -> $DEST_HOST"
    ssh "${SSH_OPTS[@]}" "$DEST_HOST" "mkdir -p '$(dirname "$dst")'" || exit 1
    scp "${SCP_OPTS[@]}" "$src" "$DEST_HOST:$dst.incoming" || {
        echo "ERROR: scp failed for $rel" >&2; exit 1; }
    ssh "${SSH_OPTS[@]}" "$DEST_HOST" "mv '$dst.incoming' '$dst'" || exit 1
    echo "  $label: delivered"
done <<< "$DATASETS"

echo
echo "=== PURSER DONE ==="
echo "  sent $(echo "$DATASETS" | wc -l | tr -d ' ') dataset(s) to $DEST"
echo "  raw replays stayed here: $REPLAYS"
