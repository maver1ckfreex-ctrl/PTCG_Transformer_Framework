#!/usr/bin/env bash
#
# V3A MIX-RATIO CYCLES -- iterative self-play at a fixed kaggle:selfplay
# trajectory ratio.
#
# Per cycle:
#   1. sample a fresh deck pool from the builder
#   2. self-play NUM_SELFPLAY games with the CURRENT model
#   3. convert -> selfplay winners npz, delete the raw json
#   4. count n = self-play winning trajectories
#   5. randomly draw n * (K/S) trajectories from the kaggle npz
#   6. merge -> training_cycle_<N>.npz
#   7. train, warm-started from the previous cycle
#   8. 1000-game review tournament vs R2   [REVIEW ONLY]
#   repeat
#
# RATIO=K:S is kaggle:selfplay by TRAJECTORY COUNT. 1:1 (default) means the
# mix pool is half kaggle, half self-play.
#
# ============================================================
#  THE TOURNAMENT IS HUMAN REVIEW ONLY.
#  The checkpoint the next cycle chains from (NEXT_INIT) is fixed BEFORE
#  the tournament runs, and the tournament's exit status is discarded.
#  Cycle N+1 is identical whatever the tournament says.
# ============================================================
#
# Usage:
#   KAGGLE_NPZ=~/raw/trajectories_winners.npz \
#   INIT=~/base.pt BUILDER=~/builder_tf_deck.pt \
#   EVAL_DECK=../submission_r2_t07_torch/deck.csv \
#   ./mix/run_mix_cycles.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-python3}"

KAGGLE_NPZ="${KAGGLE_NPZ:?set KAGGLE_NPZ=/path/to/trajectories_winners.npz}"
INIT="${INIT:?set INIT=/path/to/base_model.pt  (warm start)}"
BUILDER="${BUILDER:?set BUILDER=/path/to/builder_tf_deck.pt}"
EVAL_DECK="${EVAL_DECK:?set EVAL_DECK=/path/to/deck.csv}"

WORK="${WORK:-$HOME/mix_run}"
ENGINE="${ENGINE:-$ROOT/../submission_r2_t07_torch}"
R2="${R2:-$ROOT/../submission_r2_t07_torch/player.pt}"

RATIO="${RATIO:-1:1}"                 # kaggle : selfplay, by trajectory
NUM_SELFPLAY="${NUM_SELFPLAY:-10000}"
NUM_DECK="${NUM_DECK:-2000}"
BUILDER_TEMP="${BUILDER_TEMP:-0.7,0.8,0.9}"
PLAYER_TEMP="${PLAYER_TEMP:-0.0,0.1,0.2}"
CYCLES="${CYCLES:-0}"                 # 0 = until STOP file
EPOCHS="${EPOCHS:-4}"                 # per cycle
LR="${LR:-1e-4}"                      # continuation lr, not the cold 3e-4
GAMES_PER_STEP="${GAMES_PER_STEP:-64}"
OUTCOME_WEIGHT="${OUTCOME_WEIGHT:-0.1}"
EVAL_GAMES="${EVAL_GAMES:-1000}"
EVAL_WORKERS="${EVAL_WORKERS:-}"
WORKERS="${WORKERS:-$( (command -v nproc >/dev/null && nproc) || echo 8 )}"
SEED="${SEED:-20260805}"

# Numeric env vars: strip anything that is not a digit. A value pasted with
# a trailing space or non-breaking space makes `[ -lt ]` fail with
# "integer expression expected", and the script would take the wrong branch.
for _v in NUM_SELFPLAY NUM_DECK CYCLES EPOCHS EVAL_GAMES WORKERS SEED \
          GAMES_PER_STEP; do
    _clean=$(printf '%s' "${!_v}" | tr -cd '0-9')
    [ -n "$_clean" ] || { echo "ERROR: $_v='${!_v}' is not a number" >&2
        exit 1; }
    eval "$_v=$_clean"
done
RATIO=$(printf '%s' "$RATIO" | tr -cd '0-9:')

K_PART="${RATIO%%:*}"
S_PART="${RATIO##*:}"
case "$K_PART$S_PART" in *[!0-9]*|"") echo "ERROR: RATIO must be K:S" >&2
    exit 1 ;; esac
[ "$S_PART" -gt 0 ] || { echo "ERROR: RATIO selfplay side must be > 0" >&2
    exit 1; }

for f in "$KAGGLE_NPZ" "$INIT" "$BUILDER" "$EVAL_DECK" "$R2"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[ -d "$ENGINE/cg" ] || { echo "ERROR: no cg/ engine in $ENGINE" >&2; exit 1; }
"$PY" -c 'import numpy, torch' 2>/dev/null || {
    echo "ERROR: '$PY' cannot import numpy+torch" >&2; exit 1; }

REVIEWS="$ROOT/reviews_mix"
mkdir -p "$WORK" "$REVIEWS"

# resume: continue after the last cycle that produced a checkpoint
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
echo " V3A MIX-RATIO CYCLES"
echo " work dir     : $WORK"
echo " kaggle pool  : $KAGGLE_NPZ  ($KAGGLE_TOTAL trajectories)"
echo " ratio        : $RATIO  (kaggle : selfplay, by trajectory count)"
echo " self-play    : $NUM_SELFPLAY games/cycle | $NUM_DECK decks"
echo "                builder temps $BUILDER_TEMP | play temps $PLAYER_TEMP"
echo " train        : $EPOCHS epochs/cycle | lr $LR | init $(basename "$INIT")"
echo " review       : $EVAL_GAMES games vs R2 after each cycle [REVIEW ONLY]"
echo " cycles       : $([ "$CYCLES" -eq 0 ] && echo 'until STOP file' || echo "$CYCLES")"
echo "================================================================"

while :; do
    if [ -f "$HERE/STOP" ] || [ -f "$WORK/STOP" ]; then
        echo "STOP file present -- clean exit before cycle $CYCLE"; break
    fi
    if [ "$CYCLES" -gt 0 ] && [ "$CYCLE" -gt "$CYCLES" ]; then
        echo "reached CYCLES=$CYCLES"; break
    fi

    TAG=$(printf 'cycle_%03d' "$CYCLE")
    CDIR="$WORK/$TAG"
    mkdir -p "$CDIR"
    CSEED=$((SEED + CYCLE))
    echo
    echo "########## $TAG | pilot $(basename "$INIT") ##########"
    T0=$(date +%s)

    # ---- 1. decks ------------------------------------------------------
    POOL="$CDIR/deck_pool.npy"
    if [ ! -f "$POOL" ]; then
        echo "--- [1/6] sampling $NUM_DECK decks ---"
        "$PY" "$ROOT/selfplay/build_deck.py" --ckpt "$BUILDER" \
            --n "$NUM_DECK" --temperature "$BUILDER_TEMP" \
            --seed "$CSEED" --out-npy "$POOL" || exit 1
    fi

    # ---- 2. self-play with the CURRENT model ---------------------------
    SPDIR="$CDIR/replays_selfplay"
    SPNPZ="$CDIR/selfplay_winners.npz"
    if [ ! -f "$SPNPZ" ]; then
        mkdir -p "$SPDIR"
        HAVE=$(find "$SPDIR" -name '*.json' | wc -l | tr -d ' ')
        if [ "$HAVE" -lt "$NUM_SELFPLAY" ]; then
            echo "--- [2/6] self-play $NUM_SELFPLAY games ---"
            "$PY" "$ROOT/selfplay/selfplay_seq.py" \
                --decks "$POOL" --player "$INIT" --engine "$ENGINE" \
                --out "$SPDIR" --games "$NUM_SELFPLAY" \
                --workers "$WORKERS" --seed "$CSEED" \
                --play-temps "$PLAYER_TEMP" || exit 1
        else
            echo "--- [2/6] $HAVE self-play replays present, reusing ---"
        fi

        # A cycle with no self-play replays means self-play failed or was
        # skipped by a bad comparison. Converting 0 files would abort with
        # "no usable trajectories found" three steps later; stop here with
        # a reason instead.
        HAVE=$(find "$SPDIR" -name '*.json' | wc -l | tr -d ' ')
        if [ "$HAVE" -lt 1 ]; then
            echo "ERROR: $TAG produced 0 self-play replays." >&2
            echo "  expected $NUM_SELFPLAY games in $SPDIR" >&2
            echo "  check the self-play step above for errors" >&2
            exit 1
        fi
        echo "  self-play replays on disk: $HAVE"

        # ---- 3. convert winners, then drop the raw ---------------------
        echo "--- [3/6] converting self-play (winners only) ---"
        "$PY" "$ROOT/sequence/replay_to_trajectories.py" \
            --replays "$SPDIR" --out "$CDIR/.sp_writing.npz" \
            --winners-only --workers "$WORKERS" || exit 1
        "$PY" "$ROOT/tools/verify_data.py" "$CDIR/.sp_writing.npz" \
            --winners-only --label "$TAG/selfplay" || {
            echo "ABORT: self-play dataset bad; raw KEPT" >&2; exit 1; }
        mv "$CDIR/.sp_writing.npz" "$SPNPZ"
        echo "  raw self-play json deleted"
        rm -rf "$SPDIR"
    else
        echo "--- [2-3/6] $SPNPZ exists, reusing ---"
    fi

    # ---- 4/5. count n, draw the kaggle side at RATIO -------------------
    MIXNPZ="$CDIR/training_$TAG.npz"
    if [ ! -f "$MIXNPZ" ]; then
        N_SP=$("$PY" -c "import numpy as np;print(len(np.load('$SPNPZ')['reward']))")
        N_KG=$(( N_SP * K_PART / S_PART ))
        echo "--- [4/6] self-play winners: $N_SP"
        if [ "$N_KG" -gt "$KAGGLE_TOTAL" ]; then
            echo "  WARNING: ratio $RATIO wants $N_KG kaggle trajectories but"
            echo "           only $KAGGLE_TOTAL exist; using all of them."
            N_KG=$KAGGLE_TOTAL
        fi
        echo "--- [5/6] drawing $N_KG kaggle trajectories (ratio $RATIO) ---"
        "$PY" "$ROOT/tools/sample_npz.py" --in "$KAGGLE_NPZ" --n "$N_KG" \
            --out "$CDIR/kaggle_sub.npz" --seed "$CSEED" || exit 1
        "$PY" "$ROOT/tools/merge_npz.py" --out "$CDIR/.mix_writing.npz" \
            "$CDIR/kaggle_sub.npz" "$SPNPZ" || exit 1
        "$PY" "$ROOT/tools/verify_data.py" "$CDIR/.mix_writing.npz" \
            --winners-only --label "$TAG/mix" || exit 1
        mv "$CDIR/.mix_writing.npz" "$MIXNPZ"
        rm -f "$CDIR/kaggle_sub.npz"
    else
        echo "--- [4-5/6] $MIXNPZ exists, reusing ---"
    fi

    # ---- 6. train ------------------------------------------------------
    MODEL="$CDIR/model.pt"
    if [ ! -f "$MODEL" ]; then
        echo "--- [6/6] training $EPOCHS epochs ---"
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
        echo "--- [6/6] $MODEL exists, reusing ---"
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
            echo "ratio      : $RATIO | self-play games $NUM_SELFPLAY"
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

    echo "########## $TAG done in $(( ($(date +%s) - T0) / 60 )) min ##########"
    INIT="$NEXT_INIT"
    CYCLE=$((CYCLE + 1))
done

echo
echo "=== MIX CYCLES DONE ==="
echo "  models  : $WORK/cycle_*/model.pt"
echo "  reviews : $REVIEWS/"
