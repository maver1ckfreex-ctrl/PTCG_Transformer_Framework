#!/usr/bin/env bash
#
# NODE TRAINING -- GPU (training) node.
#
# Per cycle:
#   1. push the current model + a short REQUEST to the CPU node
#   2. wait for the CPU node to mark SELFPLAY_DONE
#   3. pull its selfplay_winners.npz, verify
#   4. draw the kaggle side at RATIO, merge
#   5. train
#   6. 1000-game review tournament vs R2        [REVIEW ONLY]
#   repeat
#
# EVERY network operation is initiated from here -- push, poll, pull. The
# CPU node only touches its own local disk. That is deliberate: the GPU box
# usually has no inbound port, so nothing can connect TO it.
#
# ============================================================
#  THE TOURNAMENT IS HUMAN REVIEW ONLY.
#  NEXT_INIT is fixed BEFORE the tournament runs and its exit status is
#  discarded. Cycle N+1 is identical whatever the tournament says.
# ============================================================
#
# Usage:
#   NODE_TRAINING=1 GPU_NODE=1 CPU_NODE=0 \
#   PEER=ubuntu@cpu-box REMOTE_WORK=/home/ubuntu/node_work \
#   KAGGLE_NPZ=~/raw/trajectories_winners.npz INIT=~/base.pt \
#   EVAL_DECK=../submission_r2_t07_torch/deck.csv \
#   ./node/run_gpu_node.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-python3}"

NODE_TRAINING="${NODE_TRAINING:-0}"
CPU_NODE="${CPU_NODE:-0}"
GPU_NODE="${GPU_NODE:-0}"
if [ "$NODE_TRAINING" != "1" ] || [ "$GPU_NODE" != "1" ]; then
    echo "This is the GPU half of node training." >&2
    echo "Run with: NODE_TRAINING=1 GPU_NODE=1 CPU_NODE=0 ..." >&2
    exit 1
fi
[ "$CPU_NODE" = "1" ] && {
    echo "ERROR: a box is either CPU_NODE or GPU_NODE, not both" >&2
    exit 1; }

PEER="${PEER:?set PEER=user@cpu-box}"
REMOTE_WORK="${REMOTE_WORK:?set REMOTE_WORK=/path/on/the/cpu/node}"
KAGGLE_NPZ="${KAGGLE_NPZ:?set KAGGLE_NPZ=/path/to/trajectories_winners.npz}"
INIT="${INIT:?set INIT=/path/to/base_model.pt}"
EVAL_DECK="${EVAL_DECK:?set EVAL_DECK=/path/to/deck.csv}"

WORK="${WORK:-$HOME/node_work}"
ENGINE="${ENGINE:-$ROOT/../submission_r2_t07_torch}"
R2="${R2:-$ROOT/../submission_r2_t07_torch/player.pt}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"

RATIO="${RATIO:-1:1}"
NUM_SELFPLAY="${NUM_SELFPLAY:-10000}"
NUM_DECK="${NUM_DECK:-2000}"
BUILDER_TEMP="${BUILDER_TEMP:-0.7,0.8,0.9}"
PLAYER_TEMP="${PLAYER_TEMP:-0.0,0.1,0.2}"
CYCLES="${CYCLES:-0}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-1e-4}"
GAMES_PER_STEP="${GAMES_PER_STEP:-64}"
OUTCOME_WEIGHT="${OUTCOME_WEIGHT:-0.1}"
EVAL_GAMES="${EVAL_GAMES:-1000}"
EVAL_WORKERS="${EVAL_WORKERS:-}"
POLL="${POLL:-30}"
SEED="${SEED:-20260805}"

for _v in NUM_SELFPLAY NUM_DECK CYCLES EPOCHS EVAL_GAMES SEED \
          GAMES_PER_STEP POLL SSH_PORT; do
    _clean=$(printf '%s' "${!_v}" | tr -cd '0-9')
    [ -n "$_clean" ] || { echo "ERROR: $_v='${!_v}' is not a number" >&2
        exit 1; }
    eval "$_v=$_clean"
done
RATIO=$(printf '%s' "$RATIO" | tr -cd '0-9:')
K_PART="${RATIO%%:*}"; S_PART="${RATIO##*:}"
case "$K_PART$S_PART" in *[!0-9]*|"") echo "ERROR: RATIO must be K:S" >&2
    exit 1 ;; esac
[ "$S_PART" -gt 0 ] || { echo "ERROR: RATIO selfplay side must be > 0" >&2
    exit 1; }

for f in "$KAGGLE_NPZ" "$INIT" "$EVAL_DECK" "$R2"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[ -d "$ENGINE/cg" ] || { echo "ERROR: no cg/ engine in $ENGINE" >&2; exit 1; }
"$PY" -c 'import numpy, torch' 2>/dev/null || {
    echo "ERROR: '$PY' cannot import numpy+torch" >&2; exit 1; }

SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes)
SCP_OPTS=(-P "$SSH_PORT" -o BatchMode=yes)
if [ -n "$SSH_KEY" ]; then SSH_OPTS+=(-i "$SSH_KEY"); SCP_OPTS+=(-i "$SSH_KEY"); fi

REVIEWS="$ROOT/reviews_node"
mkdir -p "$WORK" "$REVIEWS"

# fail fast: no point starting a cycle we cannot hand off
echo "--- checking ssh to $PEER ---"
ssh "${SSH_OPTS[@]}" -o ConnectTimeout=10 "$PEER" \
    "mkdir -p '$REMOTE_WORK' && echo ok" >/dev/null 2>&1 || {
    echo "ERROR: cannot ssh to $PEER without a password." >&2
    echo "  ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519" >&2
    echo "  ssh-copy-id -p $SSH_PORT $PEER" >&2
    exit 1; }
echo "ssh ok, remote work dir ready"

CYCLE=1
while [ -f "$WORK/cycle_$(printf '%03d' "$CYCLE")/model.pt" ]; do
    CYCLE=$((CYCLE + 1))
done
if [ "$CYCLE" -gt 1 ]; then
    INIT="$WORK/cycle_$(printf '%03d' $((CYCLE - 1)))/model.pt"
    echo "resuming: cycle $CYCLE, init from $INIT"
fi

KAGGLE_TOTAL=$("$PY" -c "import numpy as np;print(len(np.load('$KAGGLE_NPZ')['reward']))")

echo "================================================================"
echo " NODE TRAINING : GPU (training) node"
echo " peer         : $PEER:$REMOTE_WORK"
echo " local work   : $WORK"
echo " kaggle pool  : $KAGGLE_NPZ  ($KAGGLE_TOTAL trajectories)"
echo " ratio        : $RATIO  (kaggle : selfplay, by trajectory count)"
echo " self-play    : $NUM_SELFPLAY games/cycle | $NUM_DECK decks (run on peer)"
echo " train        : $EPOCHS epochs/cycle | lr $LR"
echo " review       : $EVAL_GAMES games vs R2 per cycle [REVIEW ONLY]"
echo " cycles       : $([ "$CYCLES" -eq 0 ] && echo 'until STOP file' || echo "$CYCLES")"
echo "================================================================"

while :; do
    if [ -f "$WORK/STOP" ]; then
        echo "STOP file present -- clean exit before cycle $CYCLE"; break
    fi
    if [ "$CYCLES" -gt 0 ] && [ "$CYCLE" -gt "$CYCLES" ]; then
        echo "reached CYCLES=$CYCLES"; break
    fi

    TAG=$(printf 'cycle_%03d' "$CYCLE")
    CDIR="$WORK/$TAG"
    RDIR="$REMOTE_WORK/$TAG"
    mkdir -p "$CDIR"
    CSEED=$((SEED + CYCLE))
    echo
    echo "########## $TAG | pilot $(basename "$INIT") ##########"
    T0=$(date +%s)

    SPNPZ="$CDIR/selfplay_winners.npz"
    if [ ! -f "$SPNPZ" ]; then
        # ---- 1. push pilot, THEN the request ---------------------------
        # Order matters: the CPU node treats REQUEST as the signal that
        # pilot.pt has fully landed.
        if ! ssh "${SSH_OPTS[@]}" "$PEER" "test -f '$RDIR/REQUEST'" 2>/dev/null
        then
            echo "--- [1/5] sending pilot + request to $PEER ---"
            ssh "${SSH_OPTS[@]}" "$PEER" "mkdir -p '$RDIR'" || exit 1
            scp "${SCP_OPTS[@]}" "$INIT" "$PEER:$RDIR/.pilot_incoming" \
                || { echo "ERROR: pilot upload failed" >&2; exit 1; }
            ssh "${SSH_OPTS[@]}" "$PEER" \
                "mv '$RDIR/.pilot_incoming' '$RDIR/pilot.pt'" || exit 1
            printf 'NUM_SELFPLAY=%s\nNUM_DECK=%s\nSEED=%s\nBUILDER_TEMP=%s\nPLAYER_TEMP=%s\n' \
                "$NUM_SELFPLAY" "$NUM_DECK" "$CSEED" \
                "$BUILDER_TEMP" "$PLAYER_TEMP" > "$CDIR/REQUEST"
            scp "${SCP_OPTS[@]}" "$CDIR/REQUEST" "$PEER:$RDIR/REQUEST" \
                || { echo "ERROR: request upload failed" >&2; exit 1; }
            echo "  request sent, self-play runs on the peer"
        else
            echo "--- [1/5] request already on peer, waiting ---"
        fi

        # ---- 2. wait for the peer -------------------------------------
        echo "--- [2/5] waiting for peer self-play (poll ${POLL}s) ---"
        W0=$(date +%s)
        while ! ssh "${SSH_OPTS[@]}" "$PEER" \
                "test -f '$RDIR/SELFPLAY_DONE'" 2>/dev/null; do
            if [ -f "$WORK/STOP" ]; then
                echo "STOP during wait -- exiting"; exit 0
            fi
            echo "  [$(date '+%H:%M:%S')] peer still working | $(( ($(date +%s) - W0) / 60 )) min"
            sleep "$POLL"
        done
        echo "  peer reports SELFPLAY_DONE"

        # ---- 3. pull and verify ---------------------------------------
        echo "--- [3/5] pulling selfplay_winners.npz ---"
        scp "${SCP_OPTS[@]}" "$PEER:$RDIR/selfplay_winners.npz" \
            "$CDIR/.sp_incoming.npz" || {
            echo "ERROR: pull failed" >&2; exit 1; }
        "$PY" "$ROOT/tools/verify_data.py" "$CDIR/.sp_incoming.npz" \
            --winners-only --label "$TAG/selfplay" || {
            echo "ABORT: pulled dataset failed verification" >&2
            rm -f "$CDIR/.sp_incoming.npz"; exit 1; }
        mv "$CDIR/.sp_incoming.npz" "$SPNPZ"
    else
        echo "--- [1-3/5] $SPNPZ present, reusing ---"
    fi

    # ---- 4. kaggle side at RATIO, merge -------------------------------
    MIXNPZ="$CDIR/training_$TAG.npz"
    if [ ! -f "$MIXNPZ" ]; then
        N_SP=$("$PY" -c "import numpy as np;print(len(np.load('$SPNPZ')['reward']))")
        N_KG=$(( N_SP * K_PART / S_PART ))
        echo "--- [4/5] self-play winners $N_SP | drawing $N_KG kaggle (ratio $RATIO) ---"
        if [ "$N_KG" -gt "$KAGGLE_TOTAL" ]; then
            echo "  WARNING: want $N_KG but only $KAGGLE_TOTAL exist; using all"
            N_KG=$KAGGLE_TOTAL
        fi
        "$PY" "$ROOT/tools/sample_npz.py" --in "$KAGGLE_NPZ" --n "$N_KG" \
            --out "$CDIR/kaggle_sub.npz" --seed "$CSEED" || exit 1
        "$PY" "$ROOT/tools/merge_npz.py" --out "$CDIR/.mix_writing.npz" \
            "$CDIR/kaggle_sub.npz" "$SPNPZ" || exit 1
        "$PY" "$ROOT/tools/verify_data.py" "$CDIR/.mix_writing.npz" \
            --winners-only --label "$TAG/mix" || exit 1
        mv "$CDIR/.mix_writing.npz" "$MIXNPZ"
        rm -f "$CDIR/kaggle_sub.npz"
    else
        echo "--- [4/5] $MIXNPZ present, reusing ---"
    fi

    # ---- 5. train ------------------------------------------------------
    MODEL="$CDIR/model.pt"
    if [ ! -f "$MODEL" ]; then
        echo "--- [5/5] training $EPOCHS epochs ---"
        "$PY" "$ROOT/sequence/train_seq_v3.py" \
            --data "$MIXNPZ" --out "$MODEL" --init "$INIT" \
            --epochs "$EPOCHS" --lr "$LR" \
            --games-per-step "$GAMES_PER_STEP" \
            --outcome-weight "$OUTCOME_WEIGHT" --loser-weight 1.0 \
            --epoch-ckpt-dir "$CDIR/epochs" \
            2>&1 | tee "$CDIR/train.log" || exit 1
        [ -f "$MODEL" ] || { echo "ERROR: no checkpoint from $TAG" >&2
            exit 1; }
    else
        echo "--- [5/5] $MODEL present, reusing ---"
    fi

    # ---- the chain is fixed HERE, before any tournament -----------------
    NEXT_INIT="$MODEL"

    # ---- review tournament: HUMAN REVIEW ONLY --------------------------
    REPORT="$REVIEWS/${TAG}.txt"
    if [ ! -f "$REPORT" ]; then
        echo "--- review: $EVAL_GAMES games vs R2 -> $REPORT ---"
        {
            echo "cycle      : $CYCLE"
            echo "checkpoint : $MODEL"
            echo "opponent   : $R2"
            echo "deck       : $EVAL_DECK  (both sides, seats swapped)"
            echo "ratio      : $RATIO | self-play $NUM_SELFPLAY (on $PEER)"
            echo "generated  : $(date -u '+%Y-%m-%d %H:%M:%SZ')"
            echo "NOTE: human review only. Did not influence training."
            echo
        } > "$REPORT.partial"
        EX=(); [ -n "$EVAL_WORKERS" ] && EX=(--workers "$EVAL_WORKERS")
        "$PY" "$ROOT/eval/tournament_seq.py" --engine "$ENGINE" \
            --deck "$EVAL_DECK" --a "$R2" --b "$MODEL" \
            --games "$EVAL_GAMES" ${EX[@]+"${EX[@]}"} \
            >> "$REPORT.partial" 2>&1 || true
        mv "$REPORT.partial" "$REPORT"
        grep -E "^  (A |B |z =|--> )" "$REPORT" 2>/dev/null || true
    fi

    # free the peer's disk now that the cycle is banked locally
    ssh "${SSH_OPTS[@]}" "$PEER" "rm -rf '$RDIR'" 2>/dev/null || true

    echo "########## $TAG done in $(( ($(date +%s) - T0) / 60 )) min ##########"
    INIT="$NEXT_INIT"
    CYCLE=$((CYCLE + 1))
done

echo
echo "=== NODE TRAINING DONE ==="
echo "  models  : $WORK/cycle_*/model.pt"
echo "  reviews : $REVIEWS/"
