#!/usr/bin/env bash
#
# Kaggle download -> builder dataset. Builder phase ONLY.
#
# Two modes:
#
#   --stage_pursing=0   (default)  one straight pipeline.py run: every day
#                                  appends into one growing dataset.npz.
#
#   --stage_pursing=1              staged. Download --split_days days
#                                  (default 20), parse them into their own
#                                  checkpoint, verify it, delete that
#                                  block's raw replays, then the next
#                                  block, and so on to the end of the
#                                  manifest. All checkpoints are merged
#                                  into one dataset.npz at the finish.
#
# pipeline.py is byte-identical to the copy in deck_builder_tf_v2, and its
# --convert default is still "both" -- which would call
# replay_to_decisions.py, the player-phase converter that does not exist in
# this folder. This wrapper pins --convert dataset in both modes.
#
#   ./run_builder_data.sh --manifest manifest_01Aug.csv --work /local/ptcg_work
#   ./run_builder_data.sh --stage_pursing=1 --manifest m.csv --work /local/w
#   ./run_builder_data.sh --stage_pursing=1 --split_days=30 --manifest m.csv --work /w
#
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"

STAGE=0
SPLIT_DAYS=20
HAVE_CARDS=0
PASS=()

for a in "$@"; do
    case "$a" in
        --stage_pursing=*)  STAGE="${a#*=}" ;;
        --split_days=*)     SPLIT_DAYS="${a#*=}" ;;
        --cards|--cards=*)  HAVE_CARDS=1; PASS+=("$a") ;;
        --convert|--convert=*)
            echo "ERROR: --convert is fixed to 'dataset' in the builder-only" \
                 "folder. Use deck_builder_tf_v2 for the player phase." >&2
            exit 1 ;;
        *) PASS+=("$a") ;;
    esac
done

STAGE=$(printf '%s' "$STAGE" | tr -cd '0-9')
SPLIT_DAYS=$(printf '%s' "$SPLIT_DAYS" | tr -cd '0-9')
[ -n "$STAGE" ] || { echo "ERROR: --stage_pursing must be 0 or 1" >&2; exit 1; }
[ -n "$SPLIT_DAYS" ] && [ "$SPLIT_DAYS" -ge 1 ] || {
    echo "ERROR: --split_days must be a positive integer" >&2; exit 1; }
case "$STAGE" in
    0|1) ;;
    *) echo "ERROR: --stage_pursing must be 0 or 1, got '$STAGE'" >&2
       exit 1 ;;
esac

# card_vocab.py resolves its default card data as
# <folder>/../EN_Card_Data.csv, which only exists when this folder sits
# beside the shared copy. Once it is moved somewhere else -- a server, a
# scratch disk -- that path is gone and every converter dies on
# FileNotFoundError. The csv ships inside this folder, so point at it
# unless the caller named their own.
if [ "$HAVE_CARDS" = "0" ] && [ -f "$HERE/EN_Card_Data.csv" ]; then
    PASS+=(--cards "$HERE/EN_Card_Data.csv")
fi
if [ "$HAVE_CARDS" = "0" ] && [ ! -f "$HERE/EN_Card_Data.csv" ]; then
    echo "ERROR: no EN_Card_Data.csv beside this script and no --cards" >&2
    echo "  copy it here, or pass --cards /path/to/EN_Card_Data.csv" >&2
    exit 1
fi

# Fail here, not after a 20-day download: "python3" is often not the env
# that has numpy. PY=$(which python) picks the active one.
"$PY" -c 'import numpy' 2>/dev/null || {
    echo "ERROR: '$PY' cannot import numpy." >&2
    echo "  Your active interpreter looks like: $(command -v python || echo '?')" >&2
    echo "  Re-run with:  PY=\$(which python) $0 $*" >&2
    exit 1; }

if [ "$STAGE" = "1" ]; then
    echo "=== staged parse: ${SPLIT_DAYS} days per checkpoint ==="
    exec "$PY" "$HERE/stage_pursing.py" \
        --split_days "$SPLIT_DAYS" ${PASS[@]+"${PASS[@]}"}
fi

echo "=== single-pass parse (--stage_pursing=1 for staged) ==="
exec "$PY" "$HERE/pipeline.py" --convert dataset ${PASS[@]+"${PASS[@]}"}
