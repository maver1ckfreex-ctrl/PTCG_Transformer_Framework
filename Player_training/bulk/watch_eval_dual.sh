#!/usr/bin/env bash
#
# Dual-GPU evaluator. Watches BOTH trainers' checkpoint directories and
# plays 1000 games against R2 for each new checkpoint -- one tournament at
# a time, each using every core.
#
# ============================================================
#  REVIEW ONLY. Separate process from both trainers. It reads
#  checkpoints, never writes one, and neither trainer reads this
#  directory back. Killing it changes neither model.
# ============================================================
#
# Order is GPU0's checkpoint first, then GPU1's, matching the launch order.
# Tournaments never run concurrently: two 44-worker tournaments at once
# would each get half a machine and take twice as long.
#
# Usage:
#   EVAL_DECK=deck.csv ./watch_eval_dual.sh <ckpt0> <label0> <ckpt1> <label1>
set -uo pipefail

CKPT0="${1:?usage: watch_eval_dual.sh <ckpt0> <label0> <ckpt1> <label1>}"
LABEL0="${2:?}"
CKPT1="${3:?}"
LABEL1="${4:?}"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# PY: interpreter to use (a conda env python, say). Defaults to python3.
PY="${PY:-python3}"

EVAL_DECK="${EVAL_DECK:?set EVAL_DECK=/path/to/deck.csv}"
R2="${R2:-$ROOT/../submission_r2_t07_torch/player.pt}"
ENGINE="${ENGINE:-$ROOT/../submission_r2_t07_torch}"
EVAL_GAMES="${EVAL_GAMES:-1000}"
EVAL_WORKERS="${EVAL_WORKERS:-}"       # empty -> tournament default (all)
POLL="${POLL:-60}"
REVIEWS_BASE="${REVIEWS_BASE:-$ROOT/reviews_dual}"

mkdir -p "$REVIEWS_BASE/$LABEL0" "$REVIEWS_BASE/$LABEL1"

echo "[eval] GPU0 $LABEL0 <- $CKPT0"
echo "[eval] GPU1 $LABEL1 <- $CKPT1"
echo "[eval] $EVAL_GAMES games vs $(basename "$R2") on $(basename "$EVAL_DECK")"
echo "[eval] sequential, all cores per tournament -> $REVIEWS_BASE"

run_one() {
    local ckpt="$1" label="$2" tag report
    tag="$(basename "$ckpt" .pt)"
    report="$REVIEWS_BASE/$label/${tag}.txt"
    [ -f "$report" ] && return 0

    echo "[eval] $label/$tag -> $report"
    {
        echo "trainer    : $label"
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
        ${extra[@]+"${extra[@]}"} >> "$report.partial" 2>&1 || true
    mv "$report.partial" "$report"

    echo "[eval] --- $label/$tag (A = R2, B = $label) ---"
    grep -E "^  (A |B |z =|--> )" "$report" 2>/dev/null || true
}

pending_for() {          # pending_for <ckpt_dir> <label> -> prints count
    local n=0 ck
    shopt -s nullglob
    for ck in "$1"/epoch_*.pt; do
        [ -f "$REVIEWS_BASE/$2/$(basename "$ck" .pt).txt" ] || n=$((n + 1))
    done
    shopt -u nullglob
    echo "$n"
}

while :; do
    # GPU0's queue first, then GPU1's -- one at a time, never concurrent
    shopt -s nullglob
    for ck in "$CKPT0"/epoch_*.pt; do run_one "$ck" "$LABEL0"; done
    for ck in "$CKPT1"/epoch_*.pt; do run_one "$ck" "$LABEL1"; done
    shopt -u nullglob

    if [ -f "$CKPT0/TRAINING_DONE" ] && [ -f "$CKPT1/TRAINING_DONE" ]; then
        p0=$(pending_for "$CKPT0" "$LABEL0")
        p1=$(pending_for "$CKPT1" "$LABEL1")
        if [ "$p0" -eq 0 ] && [ "$p1" -eq 0 ]; then
            echo "[eval] both trainers finished and every epoch reviewed -- exiting"
            break
        fi
    fi
    sleep "$POLL"
done
