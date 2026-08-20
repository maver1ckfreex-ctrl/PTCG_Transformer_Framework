#!/usr/bin/env bash
# One round of the loop:
#   self-play N replays  +  3 fresh kaggle days  ->  trajectories  ->  train
#
#   ./run_round.sh 1 ../submission_r2_t07_torch/player.pt 20260805
#   ./run_round.sh 2 rounds/r1/sequence.pt        20260806
#
# Round 1 warm-starts from R2 (encoder carried, memory zero-init, so the
# model starts numerically AS R2). Later rounds resume the previous
# sequence.pt. Each round must use a DIFFERENT seed so it draws different
# kaggle days.
set -euo pipefail

ROUND="${1:?usage: run_round.sh <round> <init_ckpt> <day_seed>}"
INIT="${2:?}"
SEED="${3:?}"

SELFPLAY_GAMES="${SELFPLAY_GAMES:-10000}"
PLAY_TEMPS="${PLAY_TEMPS:-0.0,0.3,0.6,1.0}"
DECK_N="${DECK_N:-5000}"
DECK_TEMPS="${DECK_TEMPS:-0.7,0.8,0.9}"
EPOCHS="${EPOCHS:-4}"

HERE="$(cd "$(dirname "$0")" && pwd)"

# Defaults are resolved against the SCRIPT's directory, never the caller's
# cwd, and a few likely layouts are searched. A bare relative default breaks
# the moment you `cd` somewhere else before launching.
find_one() {   # find_one <name> <dir> [dir...] -> prints absolute path
    local name="$1"; shift
    local d abs
    for d in "$@"; do
        if [ -e "$d/$name" ]; then
            abs="$(cd "$d" && pwd)"
            printf '%s/%s\n' "$abs" "$name"
            return 0
        fi
    done
    return 1
}

if [ -z "${BUILDER:-}" ]; then
    BUILDER="$(find_one builder_tf_deck.pt \
        "$HERE" "$HERE/selfplay" "$HERE/.." \
        "$HERE/../selfplay_mixed_temp" "$HERE/../selfplay_mixed" || true)"
fi
if [ -z "${ENGINE:-}" ]; then
    for c in "$HERE/../submission_r2_t07_torch" "$HERE/../selfplay_mixed_temp" \
             "$HERE" "$HERE/.."; do
        [ -d "$c/cg" ] && { ENGINE="$(cd "$c" && pwd)"; break; }
    done
fi

if [ -z "${BUILDER:-}" ] || [ ! -f "$BUILDER" ]; then
    echo "ERROR: deck builder checkpoint not found." >&2
    echo "  searched: $HERE, $HERE/selfplay, $HERE/.., \\" >&2
    echo "            $HERE/../selfplay_mixed_temp, $HERE/../selfplay_mixed" >&2
    echo "  fix: BUILDER=/full/path/to/builder_tf_deck.pt $0 ..." >&2
    exit 1
fi
if [ -z "${ENGINE:-}" ] || [ ! -d "$ENGINE/cg" ]; then
    echo "ERROR: no cg/ engine package found (ENGINE=${ENGINE:-unset})." >&2
    echo "  fix: ENGINE=/full/path/to/dir_containing_cg $0 ..." >&2
    exit 1
fi

OUT="$HERE/rounds/r$ROUND"
mkdir -p "$OUT/replays"
echo "builder : $BUILDER"
echo "engine  : $ENGINE"

echo "=== round $ROUND | init=$INIT | day-seed=$SEED ==="

# --- 0. sample a fresh deck pool from the builder ------------------------
# build_deck.py decodes on the GPU in lockstep (~800 decks/s) and writes one
# (n, 60) int32 .npy, which selfplay_replay.py takes directly via --decks.
echo "--- sampling $DECK_N decks (temps $DECK_TEMPS) ---"
python3 "$HERE/selfplay/build_deck.py" \
    --ckpt "$BUILDER" \
    --n "$DECK_N" \
    --temperature "$DECK_TEMPS" \
    --seed "$SEED" \
    --out-npy "$OUT/deck_pool.npy"

# --- 1. self-play, mixed temperature for diversity -----------------------
echo "--- self-play $SELFPLAY_GAMES games (temps $PLAY_TEMPS) ---"
python3 "$HERE/selfplay/selfplay_replay.py" \
    --decks "$OUT/deck_pool.npy" \
    --player "$INIT" \
    --engine "$ENGINE" \
    --out "$OUT/replays" \
    --games "$SELFPLAY_GAMES" \
    --seed "$SEED" \
    --play-temps "$PLAY_TEMPS"

# --- 2. three fresh kaggle days -----------------------------------------
echo "--- kaggle days (seed $SEED) ---"
python3 "$HERE/pick_days.py" --seed "$SEED" --days 3
python3 "$HERE/pick_days.py" --seed "$SEED" --days 3 --emit-sh > "$OUT/dl.sh"
bash "$OUT/dl.sh" "$OUT/replays"

# --- 3. one trajectory file from BOTH sources ---------------------------
echo "--- convert to trajectories ---"
python3 "$HERE/sequence/replay_to_trajectories.py" \
    --replays "$OUT/replays" --out "$OUT/trajectories.npz"

# --- 4. train ------------------------------------------------------------
echo "--- train (sequence) ---"
python3 "$HERE/sequence/train_seq.py" \
    --data "$OUT/trajectories.npz" \
    --out "$OUT/sequence.pt" \
    --epochs "$EPOCHS" \
    --init "$INIT"

echo
echo "=== round $ROUND done -> $OUT/sequence.pt ==="
# Nothing here evaluates the checkpoint. Evaluation lives in
# train_forever.sh and is human-review only; keeping this file free of any
# reference to it is what tools/check_isolation.py enforces.
