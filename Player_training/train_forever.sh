#!/usr/bin/env bash
#
# Endless training. Each round:
#     deck pool -> 10k self-play -> 3 kaggle days -> trajectories -> train
# then a 1000-game tournament against R2 on a FIXED deck you choose.
#
# ============================================================
#  THE TOURNAMENT IS HUMAN REVIEW ONLY.
#
#  Its result does not select a checkpoint, does not gate a round, does
#  not early-stop, and is never read back by anything. The checkpoint the
#  next round chains from is decided BEFORE the tournament is launched
#  (see NEXT_INIT below), and the tournament runs with its exit status
#  discarded. Round N+1 is byte-for-byte the same whether round N's
#  tournament wins 90%, loses 90%, crashes, or is skipped entirely.
#
#  Verify that claim yourself, don't take my word:
#      python3 tools/check_isolation.py
# ============================================================
#
# Usage:
#     EVAL_DECK=/path/to/your_deck.csv ./train_forever.sh
#
#     EVAL_DECK=my_deck.csv START_ROUND=1 START_SEED=20260805 \
#         SELFPLAY_GAMES=10000 KEEP_ROUNDS=2 ./train_forever.sh
#
# Stop it with Ctrl-C, or `touch STOP` in this folder for a clean stop at
# the end of the current round. Re-running resumes from the last completed
# round.

set -uo pipefail          # deliberately NOT -e: a failed tournament must
                          # never take the training loop down with it

HERE="$(cd "$(dirname "$0")" && pwd)"

EVAL_DECK="${EVAL_DECK:?set EVAL_DECK=/path/to/the/deck.csv you want every round judged on}"
R2="${R2:-$HERE/../submission_r2_t07_torch/player.pt}"
ENGINE="${ENGINE:-$HERE/../submission_r2_t07_torch}"
START_SEED="${START_SEED:-20260805}"
EVAL_GAMES="${EVAL_GAMES:-1000}"
KEEP_ROUNDS="${KEEP_ROUNDS:-2}"
MAX_RETRIES="${MAX_RETRIES:-2}"

REVIEWS="$HERE/reviews"
ROUNDS="$HERE/rounds"
mkdir -p "$REVIEWS" "$ROUNDS"

for f in "$EVAL_DECK" "$R2"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[ -d "$ENGINE/cg" ] || { echo "ERROR: no cg/ engine in $ENGINE" >&2; exit 1; }

# ---- resume: continue after the last round that produced a checkpoint ----
if [ -n "${START_ROUND:-}" ]; then
    ROUND="$START_ROUND"
else
    ROUND=1
    while [ -f "$ROUNDS/r$ROUND/sequence.pt" ]; do ROUND=$((ROUND + 1)); done
fi
SEED=$((START_SEED + ROUND - 1))

echo "================================================================"
echo " endless training | starting at round $ROUND (seed $SEED)"
echo " eval deck : $EVAL_DECK   [same deck every round, both sides]"
echo " opponent  : $R2          [fixed R2, never updated]"
echo " tournament: $EVAL_GAMES games, REVIEW ONLY -> $REVIEWS/"
echo " retention : last $KEEP_ROUNDS rounds of replays/npz kept"
echo "================================================================"
echo

while :; do
    if [ -f "$HERE/STOP" ]; then
        echo "STOP file present -- clean exit before round $ROUND"
        rm -f "$HERE/STOP"
        break
    fi

    OUT="$ROUNDS/r$ROUND"
    if [ "$ROUND" -eq 1 ]; then
        INIT="$R2"
    else
        INIT="$ROUNDS/r$((ROUND - 1))/sequence.pt"
    fi
    if [ ! -f "$INIT" ]; then
        echo "ERROR: round $ROUND needs $INIT, which does not exist" >&2
        exit 1
    fi

    echo "########## ROUND $ROUND | seed $SEED | init $(basename "$INIT") ##########"
    START_TS=$(date +%s)

    # ---------------- training (the only thing that shapes the model) ----
    attempt=1
    ok=0
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        if ENGINE="$ENGINE" bash "$HERE/run_round.sh" "$ROUND" "$INIT" "$SEED"; then
            ok=1; break
        fi
        echo "!! round $ROUND attempt $attempt failed; retrying" >&2
        attempt=$((attempt + 1))
        sleep 30
    done
    if [ "$ok" -ne 1 ] || [ ! -f "$OUT/sequence.pt" ]; then
        echo "ERROR: round $ROUND failed $MAX_RETRIES times -- stopping so the" >&2
        echo "       chain is not continued from a bad checkpoint." >&2
        exit 1
    fi

    # ---- the chain is fixed HERE, before any tournament is launched -----
    NEXT_INIT="$OUT/sequence.pt"
    echo "round $ROUND checkpoint: $NEXT_INIT"
    echo "next round will init from it unconditionally"

    # ---------------- tournament: REVIEW ONLY ----------------------------
    # stdout/stderr go to a review file and nowhere else; the exit status
    # is discarded so nothing downstream can branch on the outcome.
    REPORT="$REVIEWS/round_$(printf '%03d' "$ROUND").txt"
    echo "--- review tournament ($EVAL_GAMES games vs R2) -> $REPORT ---"
    {
        echo "round      : $ROUND"
        echo "checkpoint : $NEXT_INIT"
        echo "opponent   : $R2"
        echo "deck       : $EVAL_DECK  (both sides, seats swapped)"
        echo "generated  : $(date -u '+%Y-%m-%d %H:%M:%SZ')"
        echo "NOTE: human review only. This result did not influence training."
        echo
    } > "$REPORT"
    python3 "$HERE/eval/tournament_seq.py" \
        --engine "$ENGINE" \
        --deck "$EVAL_DECK" \
        --a "$R2" \
        --b "$NEXT_INIT" \
        --games "$EVAL_GAMES" >> "$REPORT" 2>&1 || true
    echo "--- review written (A = R2, B = round $ROUND) ---"
    grep -E "^  (A |B |z =|--> )" "$REPORT" 2>/dev/null || true

    # ---------------- retention ------------------------------------------
    OLD=$((ROUND - KEEP_ROUNDS))
    if [ "$OLD" -ge 1 ] && [ -d "$ROUNDS/r$OLD" ]; then
        echo "pruning round $OLD bulk data (checkpoint + review kept)"
        rm -rf "$ROUNDS/r$OLD/replays" "$ROUNDS/r$OLD/deck_pool.npy" \
               "$ROUNDS/r$OLD/trajectories.npz"
    fi

    echo "########## ROUND $ROUND done in $(( ($(date +%s) - START_TS) / 60 )) min ##########"
    echo
    ROUND=$((ROUND + 1))
    SEED=$((SEED + 1))
done
