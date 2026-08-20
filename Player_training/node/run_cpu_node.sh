#!/usr/bin/env bash
#
# NODE TRAINING -- CPU (self-play) node.
#
# Waits for a request from the GPU node, runs self-play with the pilot the
# GPU sent, converts to a winners-only npz, and marks it ready. The GPU
# node pulls it.
#
# All network traffic is initiated by the GPU node (it pushes the request,
# it pulls the result). This box only ever reads and writes its own local
# directory -- which is what makes the whole thing work when the GPU box
# has no inbound port.
#
#   $WORK/cycle_NNN/REQUEST      <- written by the GPU node
#   $WORK/cycle_NNN/pilot.pt     <- the model to self-play with
#   $WORK/cycle_NNN/selfplay_winners.npz   -> produced here
#   $WORK/cycle_NNN/SELFPLAY_DONE          -> "ready to pull"
#
# Usage:
#   NODE_TRAINING=1 CPU_NODE=1 GPU_NODE=0 \
#   WORK=~/node_work BUILDER=~/builder.pt \
#   ./node/run_cpu_node.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-python3}"

NODE_TRAINING="${NODE_TRAINING:-0}"
CPU_NODE="${CPU_NODE:-0}"
GPU_NODE="${GPU_NODE:-0}"
if [ "$NODE_TRAINING" != "1" ] || [ "$CPU_NODE" != "1" ]; then
    echo "This is the CPU half of node training." >&2
    echo "Run with: NODE_TRAINING=1 CPU_NODE=1 GPU_NODE=0 ..." >&2
    exit 1
fi
[ "$GPU_NODE" = "1" ] && {
    echo "ERROR: a box is either CPU_NODE or GPU_NODE, not both" >&2
    exit 1; }

WORK="${WORK:-$HOME/node_work}"
BUILDER="${BUILDER:?set BUILDER=/path/to/builder_tf_deck.pt}"
ENGINE="${ENGINE:-$ROOT/../submission_r2_t07_torch}"
POLL="${POLL:-15}"
WORKERS="${WORKERS:-$( (command -v nproc >/dev/null && nproc) || echo 8 )}"
WORKERS=$(printf '%s' "$WORKERS" | tr -cd '0-9')

[ -f "$BUILDER" ] || { echo "ERROR: missing $BUILDER" >&2; exit 1; }
[ -d "$ENGINE/cg" ] || { echo "ERROR: no cg/ engine in $ENGINE" >&2; exit 1; }
"$PY" -c 'import numpy, torch' 2>/dev/null || {
    echo "ERROR: '$PY' cannot import numpy+torch" >&2; exit 1; }
mkdir -p "$WORK"

echo "================================================================"
echo " NODE TRAINING : CPU (self-play) node"
echo " work dir : $WORK"
echo " builder  : $BUILDER"
echo " workers  : $WORKERS"
echo " polling every ${POLL}s for requests from the GPU node"
echo "================================================================"

serve_one() {                 # serve_one <cycle_dir>
    local CDIR="$1" TAG
    TAG="$(basename "$CDIR")"
    local REQ="$CDIR/REQUEST" PILOT="$CDIR/pilot.pt"

    # The GPU node writes REQUEST last, so its presence means pilot.pt has
    # fully landed. Belt and braces: check the pilot too.
    if [ ! -f "$PILOT" ]; then
        echo "[$TAG] REQUEST present but pilot.pt missing -- waiting"
        return 0
    fi

    # request is KEY=VALUE lines, values are digits/commas only
    local NUM_SELFPLAY NUM_DECK BUILDER_TEMP PLAYER_TEMP CSEED
    NUM_SELFPLAY=$(grep -E '^NUM_SELFPLAY=' "$REQ" | cut -d= -f2 | tr -cd '0-9')
    NUM_DECK=$(grep -E '^NUM_DECK=' "$REQ" | cut -d= -f2 | tr -cd '0-9')
    CSEED=$(grep -E '^SEED=' "$REQ" | cut -d= -f2 | tr -cd '0-9')
    BUILDER_TEMP=$(grep -E '^BUILDER_TEMP=' "$REQ" | cut -d= -f2 | tr -cd '0-9.,')
    PLAYER_TEMP=$(grep -E '^PLAYER_TEMP=' "$REQ" | cut -d= -f2 | tr -cd '0-9.,')
    : "${NUM_SELFPLAY:=10000}" "${NUM_DECK:=2000}" "${CSEED:=0}"
    : "${BUILDER_TEMP:=0.7,0.8,0.9}" "${PLAYER_TEMP:=0.0,0.1,0.2}"

    echo
    echo "########## [$TAG] request received ##########"
    echo "  self-play $NUM_SELFPLAY games | $NUM_DECK decks | seed $CSEED"
    echo "  builder temps $BUILDER_TEMP | play temps $PLAYER_TEMP"
    local T0
    T0=$(date +%s)

    local POOL="$CDIR/deck_pool.npy"
    if [ ! -f "$POOL" ]; then
        echo "[$TAG] sampling $NUM_DECK decks"
        "$PY" "$ROOT/selfplay/build_deck.py" --ckpt "$BUILDER" \
            --n "$NUM_DECK" --temperature "$BUILDER_TEMP" \
            --seed "$CSEED" --out-npy "$POOL" || return 1
    fi

    local SPDIR="$CDIR/replays_selfplay"
    mkdir -p "$SPDIR"
    local HAVE
    HAVE=$(find "$SPDIR" -name '*.json' | wc -l | tr -d ' ')
    if [ "$HAVE" -lt "$NUM_SELFPLAY" ]; then
        echo "[$TAG] self-play $NUM_SELFPLAY games with $(basename "$PILOT")"
        "$PY" "$ROOT/selfplay/selfplay_seq.py" \
            --decks "$POOL" --player "$PILOT" --engine "$ENGINE" \
            --out "$SPDIR" --games "$NUM_SELFPLAY" --workers "$WORKERS" \
            --seed "$CSEED" --play-temps "$PLAYER_TEMP" || return 1
    else
        echo "[$TAG] $HAVE replays already present, reusing"
    fi

    HAVE=$(find "$SPDIR" -name '*.json' | wc -l | tr -d ' ')
    if [ "$HAVE" -lt 1 ]; then
        echo "[$TAG] ERROR: 0 self-play replays produced" >&2
        return 1
    fi
    echo "[$TAG] $HAVE replays on disk"

    echo "[$TAG] converting (winners only)"
    "$PY" "$ROOT/sequence/replay_to_trajectories.py" \
        --replays "$SPDIR" --out "$CDIR/.sp_writing.npz" \
        --winners-only --workers "$WORKERS" || return 1
    "$PY" "$ROOT/tools/verify_data.py" "$CDIR/.sp_writing.npz" \
        --winners-only --label "$TAG" || {
        echo "[$TAG] ABORT: bad dataset; raw KEPT" >&2; return 1; }
    mv "$CDIR/.sp_writing.npz" "$CDIR/selfplay_winners.npz"
    rm -rf "$SPDIR"
    echo "[$TAG] raw json deleted"

    # DONE is written LAST: the GPU node polls for it, so it must not
    # appear before the npz is complete and verified.
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$CDIR/SELFPLAY_DONE"
    echo "########## [$TAG] ready to pull, $(( ($(date +%s) - T0) / 60 )) min ##########"
    return 0
}

while :; do
    if [ -f "$WORK/STOP" ]; then
        echo "STOP file present -- exiting"; break
    fi
    served=0
    shopt -s nullglob
    for CDIR in "$WORK"/cycle_*; do
        [ -f "$CDIR/REQUEST" ] || continue
        [ -f "$CDIR/SELFPLAY_DONE" ] && continue
        serve_one "$CDIR" || {
            echo "ERROR serving $(basename "$CDIR") -- leaving it unmarked" >&2
        }
        served=1
    done
    shopt -u nullglob
    [ "$served" -eq 1 ] || sleep "$POLL"
done
