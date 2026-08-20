#!/usr/bin/env bash
#
# Watches a directory of per-epoch checkpoints and plays a 1000-game
# tournament against R2 for each new one, on a fixed deck.
#
# ============================================================
#  REVIEW ONLY. This runs as a SEPARATE PROCESS from the trainer.
#  It reads checkpoints; it never writes one, never signals the
#  trainer, and the trainer never reads this directory back. Killing
#  this watcher changes nothing about the model that gets produced.
# ============================================================
#
# Usage (normally launched by run_bulk.sh, but standalone works):
#     EVAL_DECK=/path/deck.csv ./watch_eval.sh /path/to/epoch_ckpts
set -uo pipefail

CKPT_DIR="${1:?usage: watch_eval.sh <epoch_ckpt_dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# PY: interpreter to use (a conda env python, say). Defaults to python3.
PY="${PY:-python3}"

EVAL_DECK="${EVAL_DECK:?set EVAL_DECK=/path/to/deck.csv}"
R2="${R2:-$ROOT/../submission_r2_t07_torch/player.pt}"
ENGINE="${ENGINE:-$ROOT/../submission_r2_t07_torch}"
EVAL_GAMES="${EVAL_GAMES:-1000}"
EVAL_WORKERS="${EVAL_WORKERS:-}"          # empty = tournament's own default
POLL="${POLL:-60}"

REVIEWS="${REVIEWS:-$ROOT/reviews_bulk}"
mkdir -p "$REVIEWS"

echo "[eval] watching $CKPT_DIR"
echo "[eval] $EVAL_GAMES games vs $(basename "$R2") on $(basename "$EVAL_DECK")"
echo "[eval] reports -> $REVIEWS"

run_one() {
    local ckpt="$1" tag report
    tag="$(basename "$ckpt" .pt)"
    report="$REVIEWS/${tag}.txt"
    [ -f "$report" ] && return 0

    echo "[eval] $tag -> $report"
    {
        echo "checkpoint : $ckpt"
        echo "opponent   : $R2"
        echo "deck       : $EVAL_DECK  (both sides, seats swapped)"
        echo "games      : $EVAL_GAMES"
        echo "generated  : $(date -u '+%Y-%m-%d %H:%M:%SZ')"
        echo "NOTE: human review only. Did not influence training."
        echo
    } > "$report.partial"

    local extra=()
    [ -n "$EVAL_WORKERS" ] && extra=(--workers "$EVAL_WORKERS")
    "$PY" "$ROOT/eval/tournament_seq.py" \
        --engine "$ENGINE" --deck "$EVAL_DECK" \
        --a "$R2" --b "$ckpt" --games "$EVAL_GAMES" \
        "${extra[@]}" >> "$report.partial" 2>&1 || true
    mv "$report.partial" "$report"

    echo "[eval] --- $tag (A = R2, B = $tag) ---"
    grep -E "^  (A |B |z =|--> )" "$report" 2>/dev/null || true
}

while :; do
    shopt -s nullglob
    for ck in "$CKPT_DIR"/epoch_*.pt; do
        run_one "$ck"
    done
    shopt -u nullglob

    if [ -f "$CKPT_DIR/TRAINING_DONE" ]; then
        pending=0
        shopt -s nullglob
        for ck in "$CKPT_DIR"/epoch_*.pt; do
            [ -f "$REVIEWS/$(basename "$ck" .pt).txt" ] || pending=1
        done
        shopt -u nullglob
        if [ "$pending" -eq 0 ]; then
            echo "[eval] training finished and every epoch reviewed -- exiting"
            break
        fi
    fi
    sleep "$POLL"
done
