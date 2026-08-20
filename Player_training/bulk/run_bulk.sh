#!/usr/bin/env bash
#
# BULK MODE -- cold start, every kaggle day at once, no self-play.
#
#   download all 49 days  ->  convert to one trajectories.npz  ->  train
#   from RANDOM init, with a 1000-game tournament against R2 fired
#   automatically after every epoch.
#
# Contrast with the round mode (train_forever.sh): that one warm-starts
# from R2, mixes in self-play, and works 3 days at a time. This one takes
# the whole corpus in a single pass and starts from nothing, so what it
# measures is what the trajectory design learns on its own.
#
# The per-epoch tournament runs in a SEPARATE watcher process
# (bulk/watch_eval.sh). The trainer writes checkpoints and never looks at
# a result; the watcher reads checkpoints and never writes one. Review
# only, same rule as round mode -- verify with tools/check_isolation.py.
#
# Usage:
#     DATA_DIR=/mnt/big/pokemon EVAL_DECK=../submission_r2_t07_torch/deck.csv \
#         ./bulk/run_bulk.sh
#
# Resume: re-running skips the download for days already present and skips
# conversion if trajectories.npz exists. Delete the npz to force a redo.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# PY: interpreter to use (a conda env python, say). Defaults to python3.
PY="${PY:-python3}"

DATA_DIR="${DATA_DIR:?set DATA_DIR=/path/with/room/for/the/whole/corpus}"
EVAL_DECK="${EVAL_DECK:?set EVAL_DECK=/path/to/deck.csv}"

R2="${R2:-$ROOT/../submission_r2_t07_torch/player.pt}"
ENGINE="${ENGINE:-$ROOT/../submission_r2_t07_torch}"
EPOCHS="${EPOCHS:-8}"
DECISIONS_PER_BATCH="${DECISIONS_PER_BATCH:-256}"
LR="${LR:-3e-4}"
MEM_LAYERS="${MEM_LAYERS:-2}"

# TRAINER selects which trainer file runs. All three are separate files;
# none overwrites another.
#   v1  original: +/-1 outcome stamped on every decision, 4 games/step
#   v2  v1 + gradient accumulation + p clamp + lr schedule
#   v3  per-step action cross-entropy, outcome ONLY at end of trajectory
TRAINER="${TRAINER:-v1}"
GAMES_PER_STEP="${GAMES_PER_STEP:-64}"    # v2, v3
P_MAX="${P_MAX:-0.95}"                    # v2 only
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"      # v2, v3
WARMUP_STEPS="${WARMUP_STEPS:-500}"       # v2, v3
OUTCOME_WEIGHT="${OUTCOME_WEIGHT:-0.1}"   # v3, v3a
LOSER_WEIGHT="${LOSER_WEIGHT:-1.0}"       # v3 only

# v3a self-play mixing, same semantics as dual mode
MIX_V3A="${MIX_V3A:-0}"
BUILDER="${BUILDER:-}"
BUILDER_TEMP="${BUILDER_TEMP:-0.7,0.8,0.9}"
PLAYER_TEMP="${PLAYER_TEMP:-0.0,0.1,0.2}"
NUM_DECK="${NUM_DECK:-2000}"
NUM_SELFPLAY="${NUM_SELFPLAY:-10000}"

CONVERT_EXTRA=()                          # v3a adds --winners-only
NPZ_NAME="trajectories.npz"
case "$TRAINER" in
    v1) TRAIN_SCRIPT="$ROOT/sequence/train_seq.py"; TRAIN_EXTRA=() ;;
    v2) TRAIN_SCRIPT="$ROOT/sequence/train_seq_v2.py"
        TRAIN_EXTRA=(--games-per-step "$GAMES_PER_STEP"
                     --p-max "$P_MAX"
                     --lr-schedule "$LR_SCHEDULE"
                     --warmup-steps "$WARMUP_STEPS") ;;
    v3) TRAIN_SCRIPT="$ROOT/sequence/train_seq_v3.py"
        TRAIN_EXTRA=(--games-per-step "$GAMES_PER_STEP"
                     --outcome-weight "$OUTCOME_WEIGHT"
                     --loser-weight "$LOSER_WEIGHT"
                     --lr-schedule "$LR_SCHEDULE"
                     --warmup-steps "$WARMUP_STEPS") ;;
    v3a) TRAIN_SCRIPT="$ROOT/sequence/train_seq_v3.py"
         # V3a differs at CONVERSION: the loser's side is dropped, so it
         # gets its own npz rather than silently reusing a both-sides one.
         CONVERT_EXTRA=(--winners-only)
         # MIX_V3A changes the npz NAME too, so data-transfer mode and
         # bulk mode agree on where the dataset lives.
         [ "$MIX_V3A" = "1" ] && NPZ_NAME="trajectories_mix.npz" \
                              || NPZ_NAME="trajectories_winners.npz"
         TRAIN_EXTRA=(--games-per-step "$GAMES_PER_STEP"
                      --outcome-weight "$OUTCOME_WEIGHT"
                      --loser-weight 1.0
                      --lr-schedule "$LR_SCHEDULE"
                      --warmup-steps "$WARMUP_STEPS") ;;
    v3b) TRAIN_SCRIPT="$ROOT/sequence/train_seq_v3b.py"
         TRAIN_EXTRA=(--games-per-step "$GAMES_PER_STEP"
                      --p-max "$P_MAX"
                      --lr-schedule "$LR_SCHEDULE"
                      --warmup-steps "$WARMUP_STEPS") ;;
    *)  echo "ERROR: TRAINER must be v1, v2, v3, v3a or v3b" \
             "(got '$TRAINER')" >&2
        exit 1 ;;
esac
EVAL_GAMES="${EVAL_GAMES:-1000}"
EVAL_WORKERS="${EVAL_WORKERS:-}"
INIT="${INIT:-}"                 # empty = COLD START (random init)
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

# Day selection: ALL days by default. Set DAYS=n to take a random n
# instead, reproducible from SEED.
#     DAYS=7 ./bulk/run_bulk.sh              # 7 random days, seed 20260805
#     DAYS=7 SEED=1234 ./bulk/run_bulk.sh    # a different 7
DAYS="${DAYS:-}"
SEED="${SEED:-20260805}"
if [ -n "$DAYS" ]; then
    PICK_ARGS=(--days "$DAYS" --seed "$SEED")
    PICK_LABEL="$DAYS random days (seed $SEED)"
else
    PICK_ARGS=(--all)
    PICK_LABEL="ALL days in the manifest"
fi

# Checkpoints, best-model and reviews are tagged with the trainer so two
# variants can share one DATA_DIR (and the downloaded replays) without
# overwriting each other. Only replays/ and the npz are shared.
REPLAYS="$DATA_DIR/replays"
SELFPLAY_DIR="$DATA_DIR/replays_selfplay"
NPZ="$DATA_DIR/$NPZ_NAME"
CKPTS="$DATA_DIR/epoch_ckpts_$TRAINER"
BEST="$DATA_DIR/best_$TRAINER.pt"
REVIEWS="$ROOT/reviews_bulk_$TRAINER"
mkdir -p "$REPLAYS" "$CKPTS"

for f in "$EVAL_DECK" "$R2"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[ -d "$ENGINE/cg" ] || { echo "ERROR: no cg/ engine in $ENGINE" >&2; exit 1; }

echo "================================================================"
echo " BULK MODE  (cold start, all days, no self-play)"
echo " data dir : $DATA_DIR"
echo " replays  : $REPLAYS"
echo " npz      : $NPZ"
echo " ckpts    : $CKPTS"
echo " best     : $BEST"
echo " reviews  : $REVIEWS"
echo " trainer  : $TRAINER  ($(basename "$TRAIN_SCRIPT"))"
echo " init     : ${INIT:-COLD START (random)}"
echo " days     : $PICK_LABEL"
echo " epochs   : $EPOCHS   | eval $EVAL_GAMES games after EVERY epoch"
echo " eval deck: $EVAL_DECK   [same deck both sides]"
echo "================================================================"
"$PY" "$ROOT/pick_days.py" "${PICK_ARGS[@]}"
echo

# ---- 1. download every day ---------------------------------------------
if [ "$SKIP_DOWNLOAD" != "1" ]; then
    echo "--- downloading $PICK_LABEL -> $REPLAYS ---"
    "$PY" "$ROOT/pick_days.py" "${PICK_ARGS[@]}" --emit-sh \
        > "$DATA_DIR/download_all.sh"
    bash "$DATA_DIR/download_all.sh" "$REPLAYS" || {
        echo "ERROR: download failed. Check the kaggle CLI token." >&2
        exit 1; }
else
    echo "--- SKIP_DOWNLOAD=1, using what is already in $REPLAYS ---"
fi
echo "replay files present: $(find "$REPLAYS" -name '*.json' | wc -l)"

# ---- 1b. self-play, only when mixing into v3a --------------------------
# Skipped entirely when the npz already exists (e.g. data-transfer mode
# delivered it), so the GPU box never generates self-play.
if [ "$MIX_V3A" = "1" ] && [ "$TRAINER" = "v3a" ] && [ ! -f "$NPZ" ]; then
    [ -n "$BUILDER" ] && [ -f "$BUILDER" ] || {
        echo "ERROR: MIX_V3A=1 needs BUILDER=/path/to/builder_tf_deck.pt" >&2
        exit 1; }
    mkdir -p "$SELFPLAY_DIR"
    HAVE=$(find "$SELFPLAY_DIR" -name '*.json' | wc -l | tr -d ' ')
    if [ "$HAVE" -ge "$NUM_SELFPLAY" ]; then
        echo "--- [1b] $HAVE self-play replays present, reusing ---"
    else
        POOL="$DATA_DIR/selfplay_deck_pool.npy"
        [ -f "$POOL" ] || {
            echo "--- [1b] sampling $NUM_DECK decks ---"
            "$PY" "$ROOT/selfplay/build_deck.py" --ckpt "$BUILDER" \
                --n "$NUM_DECK" --temperature "$BUILDER_TEMP" \
                --seed "$SEED" --out-npy "$POOL" || exit 1; }
        echo "--- [1b] self-play $NUM_SELFPLAY games ---"
        "$PY" "$ROOT/selfplay/selfplay_replay.py" \
            --decks "$POOL" --player "${SP_PILOT:-$R2}" --engine "$ENGINE" \
            --out "$SELFPLAY_DIR" --games "$NUM_SELFPLAY" \
            --workers "$(nproc 2>/dev/null || echo 8)" --seed "$SEED" \
            --play-temps "$PLAYER_TEMP" || exit 1
    fi
    CONVERT_SRCS=("$REPLAYS" "$SELFPLAY_DIR")
else
    CONVERT_SRCS=("$REPLAYS")
fi

# ---- 2. one conversion pass over everything -----------------------------
if [ -f "$NPZ" ]; then
    echo "--- $NPZ exists, skipping conversion (delete it to redo) ---"
else
    echo "--- converting to trajectories ---"
    "$PY" "$ROOT/sequence/replay_to_trajectories.py" \
        --replays "${CONVERT_SRCS[@]}" --out "$NPZ" \
        --workers "$(nproc 2>/dev/null || echo 8)" \
        ${CONVERT_EXTRA[@]+"${CONVERT_EXTRA[@]}"} || exit 1
fi

# ---- 3. start the review watcher (separate process) ---------------------
echo "--- starting per-epoch evaluator ---"
EVAL_DECK="$EVAL_DECK" R2="$R2" ENGINE="$ENGINE" \
EVAL_GAMES="$EVAL_GAMES" EVAL_WORKERS="$EVAL_WORKERS" \
REVIEWS="$REVIEWS" \
    bash "$HERE/watch_eval.sh" "$CKPTS" &
WATCHER=$!
trap 'kill $WATCHER 2>/dev/null' EXIT

# ---- 4. train -----------------------------------------------------------
echo "--- training (trainer=$TRAINER: $(basename "$TRAIN_SCRIPT")) ---"
INIT_ARG=()
[ -n "$INIT" ] && INIT_ARG=(--init "$INIT")
"$PY" "$TRAIN_SCRIPT" \
    --data "$NPZ" \
    --out "$BEST" \
    --epochs "$EPOCHS" \
    --decisions-per-batch "$DECISIONS_PER_BATCH" \
    --lr "$LR" \
    --mem-layers "$MEM_LAYERS" \
    --epoch-ckpt-dir "$CKPTS" \
    "${TRAIN_EXTRA[@]}" \
    "${INIT_ARG[@]}"
TRAIN_RC=$?

echo "--- training exited ($TRAIN_RC); waiting for the evaluator to finish ---"
wait $WATCHER 2>/dev/null
trap - EXIT

echo
echo "=== BULK MODE DONE ==="
echo "  best checkpoint : $BEST"
echo "  per-epoch       : $CKPTS/epoch_NNN.pt"
echo "  reviews         : $REVIEWS/"
exit $TRAIN_RC
