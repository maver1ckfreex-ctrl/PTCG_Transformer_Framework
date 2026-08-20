#!/usr/bin/env bash
#
# DUAL-GPU MODE -- two trainers, two GPUs, two models, fully isolated.
#
#   download (once)  ->  unzip (once)  ->  convert each trainer's dataset
#   SEQUENTIALLY with every core  ->  VERIFY both datasets  ->  only then
#   start both trainers, one per GPU, each pinned to half the cores.
#
#   Tournaments run in ONE separate watcher, sequentially, all cores each:
#   GPU0's model vs R2, then GPU1's model vs R2. Review only.
#
# The two training processes share nothing: separate CUDA device, separate
# core set, separate npz, separate checkpoint dir, separate output. Killing
# one does not touch the other.
#
# Usage:
#   DUAL_GPU=1 DATA_DIR=~/raw DAYS=20 \
#   GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b \
#   EVAL_DECK=../submission_r2_t07_torch/deck.csv ./bulk/run_dual.sh
#
# DUAL_GPU defaults to 0; with 0 this script tells you to use run_bulk.sh
# instead, so the single-GPU path stays the default everywhere.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# PY: interpreter to use (a conda env python, say). Defaults to python3.
PY="${PY:-python3}"

DUAL_GPU="${DUAL_GPU:-0}"
if [ "$DUAL_GPU" != "1" ]; then
    echo "DUAL_GPU is $DUAL_GPU (default 0 = single GPU)." >&2
    echo "Set DUAL_GPU=1 to use this script, or run ./bulk/run_bulk.sh" >&2
    exit 1
fi

DATA_DIR="${DATA_DIR:?set DATA_DIR=/path/with/room/for/the/corpus}"
EVAL_DECK="${EVAL_DECK:?set EVAL_DECK=/path/to/deck.csv}"
GPU_0_TRAINER="${GPU_0_TRAINER:?set GPU_0_TRAINER=v1|v2|v3|v3a|v3b}"
GPU_1_TRAINER="${GPU_1_TRAINER:?set GPU_1_TRAINER=v1|v2|v3|v3a|v3b}"

R2="${R2:-$ROOT/../submission_r2_t07_torch/player.pt}"
ENGINE="${ENGINE:-$ROOT/../submission_r2_t07_torch}"
EPOCHS="${EPOCHS:-8}"
DECISIONS_PER_BATCH="${DECISIONS_PER_BATCH:-256}"
LR="${LR:-3e-4}"
MEM_LAYERS="${MEM_LAYERS:-2}"
GAMES_PER_STEP="${GAMES_PER_STEP:-64}"
P_MAX="${P_MAX:-0.95}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
OUTCOME_WEIGHT="${OUTCOME_WEIGHT:-0.1}"
LOSER_WEIGHT="${LOSER_WEIGHT:-1.0}"
EVAL_GAMES="${EVAL_GAMES:-1000}"
INIT_0="${INIT_0:-}"
INIT_1="${INIT_1:-}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

# ---- v3a self-play mixing (OFF by default) ------------------------------
# MIX_V3A=1 generates self-play games and folds them into the v3a dataset
# ONLY. They go through the same winners-only conversion as the kaggle
# replays, so v3a still trains on winning trajectories exclusively.
# v3b's dataset is untouched.
MIX_V3A="${MIX_V3A:-0}"
BUILDER="${BUILDER:-}"
BUILDER_TEMP="${BUILDER_TEMP:-0.7,0.8,0.9}"
PLAYER_TEMP="${PLAYER_TEMP:-0.0,0.1,0.2}"
NUM_DECK="${NUM_DECK:-2000}"
NUM_SELFPLAY="${NUM_SELFPLAY:-10000}"
DAYS="${DAYS:-}"
SEED="${SEED:-20260805}"

TOTAL_CORES="${TOTAL_CORES:-$( (command -v nproc >/dev/null && nproc) \
    || sysctl -n hw.ncpu 2>/dev/null || echo 8 )}"
HALF=$(( TOTAL_CORES / 2 ))
[ "$HALF" -lt 1 ] && HALF=1

if [ -n "$DAYS" ]; then
    PICK_ARGS=(--days "$DAYS" --seed "$SEED")
    PICK_LABEL="$DAYS random days (seed $SEED)"
else
    PICK_ARGS=(--all)
    PICK_LABEL="ALL days in the manifest"
fi

# ---- per-trainer: script, extra flags, and which DATASET it needs -------
# data_spec is "winners" or "both": trainers sharing a spec share one npz
# rather than converting the same thing twice.
trainer_script() {
    case "$1" in
        v1)  echo "$ROOT/sequence/train_seq.py" ;;
        v2)  echo "$ROOT/sequence/train_seq_v2.py" ;;
        v3|v3a) echo "$ROOT/sequence/train_seq_v3.py" ;;
        v3b) echo "$ROOT/sequence/train_seq_v3b.py" ;;
        *)   echo "" ;;
    esac
}
data_spec() { [ "$1" = "v3a" ] && echo "winners" || echo "both"; }

trainer_flags() {           # prints one flag per line
    case "$1" in
        v1) ;;
        v2) printf '%s\n' --games-per-step "$GAMES_PER_STEP" \
                          --p-max "$P_MAX" \
                          --lr-schedule "$LR_SCHEDULE" \
                          --warmup-steps "$WARMUP_STEPS" ;;
        v3) printf '%s\n' --games-per-step "$GAMES_PER_STEP" \
                          --outcome-weight "$OUTCOME_WEIGHT" \
                          --loser-weight "$LOSER_WEIGHT" \
                          --lr-schedule "$LR_SCHEDULE" \
                          --warmup-steps "$WARMUP_STEPS" ;;
        v3a) printf '%s\n' --games-per-step "$GAMES_PER_STEP" \
                           --outcome-weight "$OUTCOME_WEIGHT" \
                           --loser-weight 1.0 \
                           --lr-schedule "$LR_SCHEDULE" \
                           --warmup-steps "$WARMUP_STEPS" ;;
        v3b) printf '%s\n' --games-per-step "$GAMES_PER_STEP" \
                           --p-max "$P_MAX" \
                           --lr-schedule "$LR_SCHEDULE" \
                           --warmup-steps "$WARMUP_STEPS" ;;
    esac
}

for t in "$GPU_0_TRAINER" "$GPU_1_TRAINER"; do
    [ -n "$(trainer_script "$t")" ] || {
        echo "ERROR: unknown trainer '$t' (v1|v2|v3|v3a|v3b)" >&2; exit 1; }
done
for f in "$EVAL_DECK" "$R2"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[ -d "$ENGINE/cg" ] || { echo "ERROR: no cg/ engine in $ENGINE" >&2; exit 1; }

REPLAYS="$DATA_DIR/replays"
DIR0="$DATA_DIR/$GPU_0_TRAINER"
DIR1="$DATA_DIR/$GPU_1_TRAINER"
mkdir -p "$REPLAYS" "$DIR0" "$DIR1"
SPEC0="$(data_spec "$GPU_0_TRAINER")"
SPEC1="$(data_spec "$GPU_1_TRAINER")"

# self-play replays live in their own tree so they can be folded into the
# v3a dataset WITHOUT leaking into v3b's
SELFPLAY_DIR="$DATA_DIR/replays_selfplay"
mix_for() {            # mix_for <trainer> -> 1 if this trainer gets self-play
    [ "$MIX_V3A" = "1" ] && [ "$1" = "v3a" ] && echo 1 || echo 0
}
MIX0="$(mix_for "$GPU_0_TRAINER")"
MIX1="$(mix_for "$GPU_1_TRAINER")"
[ "$MIX0" = "1" ] && NPZ0="$DIR0/trajectories_mix.npz" \
                  || NPZ0="$DIR0/trajectories.npz"
[ "$MIX1" = "1" ] && NPZ1="$DIR1/trajectories_mix.npz" \
                  || NPZ1="$DIR1/trajectories.npz"

SHARED=0
if [ "$SPEC0" = "$SPEC1" ] && [ "$MIX0" = "$MIX1" ] \
   && [ "$GPU_0_TRAINER" != "$GPU_1_TRAINER" ]; then
    SHARED=1                        # identical dataset: convert once
fi

ANY_MIX=0
if [ "$MIX0" = "1" ] || [ "$MIX1" = "1" ]; then ANY_MIX=1; fi
if [ "$ANY_MIX" = "1" ]; then
    # the self-play pilot must be a checkpoint selfplay_replay.py can load,
    # i.e. an R2-era PlayerDecoder. A sequence_v3/v3b checkpoint uses
    # different parameter names and would crash inside a worker.
    SP_PILOT="${SP_PILOT:-${INIT_0:-$R2}}"
    if python3 - "$SP_PILOT" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(1 if "arch" in ck else 0)
EOF
    then :; else
        echo "ERROR: self-play pilot '$SP_PILOT' is a sequence-arch" >&2
        echo "       checkpoint. selfplay_replay.py can only pilot an" >&2
        echo "       R2-era player.pt. Set SP_PILOT=/path/to/player.pt" >&2
        exit 1
    fi
    [ -n "$BUILDER" ] || {
        echo "ERROR: MIX_V3A=1 needs BUILDER=/path/to/builder_tf_deck.pt" >&2
        exit 1; }
    [ -f "$BUILDER" ] || {
        echo "ERROR: builder not found: $BUILDER" >&2; exit 1; }
fi

echo "================================================================"
echo " DUAL-GPU MODE"
echo " data dir  : $DATA_DIR"
echo " replays   : $REPLAYS   ($PICK_LABEL)"
echo " cores     : $TOTAL_CORES total -> GPU0 0-$((HALF-1)), GPU1 $HALF-$((TOTAL_CORES-1))"
echo " GPU 0     : $GPU_0_TRAINER  ($(basename "$(trainer_script "$GPU_0_TRAINER")"))"
echo "             data=$SPEC0  npz=$NPZ0"
echo "             init=${INIT_0:-COLD START}"
echo " GPU 1     : $GPU_1_TRAINER  ($(basename "$(trainer_script "$GPU_1_TRAINER")"))"
echo "             data=$SPEC1  npz=$NPZ1"
echo "             init=${INIT_1:-COLD START}"
[ "$SHARED" = "1" ] && echo " (both need '$SPEC0' data -- converting ONCE and sharing)"
if [ "$ANY_MIX" = "1" ]; then
    echo " MIX_V3A   : ON -> v3a only"
    echo "   pilot   : $SP_PILOT"
    echo "   builder : $BUILDER  temps $BUILDER_TEMP"
    echo "   decks   : $NUM_DECK | games $NUM_SELFPLAY | play temps $PLAYER_TEMP"
    echo "   selfplay: $SELFPLAY_DIR  (winners-only, same as kaggle)"
fi
echo " epochs    : $EPOCHS | review $EVAL_GAMES games/epoch, sequential"
echo "================================================================"
"$PY" "$ROOT/pick_days.py" "${PICK_ARGS[@]}"
echo

# ---- 1. download once, shared by both trainers -------------------------
if [ "$SKIP_DOWNLOAD" != "1" ]; then
    echo "--- [1/5] downloading $PICK_LABEL -> $REPLAYS ---"
    "$PY" "$ROOT/pick_days.py" "${PICK_ARGS[@]}" --emit-sh \
        > "$DATA_DIR/download_all.sh"
    bash "$DATA_DIR/download_all.sh" "$REPLAYS" || {
        echo "ERROR: download failed. Check the kaggle CLI token." >&2
        exit 1; }
else
    echo "--- [1/5] SKIP_DOWNLOAD=1, using $REPLAYS as-is ---"
fi
N_JSON=$(find "$REPLAYS" -name '*.json' | wc -l | tr -d ' ')
echo "replay files present: $N_JSON"
[ "$N_JSON" -gt 0 ] || { echo "ERROR: no replays to convert" >&2; exit 1; }

# ---- 1b. self-play generation (v3a mixing only) -------------------------
if [ "$ANY_MIX" = "1" ]; then
    mkdir -p "$SELFPLAY_DIR"
    HAVE_SP=$(find "$SELFPLAY_DIR" -name '*.json' | wc -l | tr -d ' ')
    if [ "$HAVE_SP" -ge "$NUM_SELFPLAY" ]; then
        echo "--- [1b/5] $HAVE_SP self-play replays already present, reusing ---"
    else
        POOL="$DATA_DIR/selfplay_deck_pool.npy"
        if [ ! -f "$POOL" ]; then
            echo "--- [1b/5] sampling $NUM_DECK decks (temps $BUILDER_TEMP) ---"
            "$PY" "$ROOT/selfplay/build_deck.py" \
                --ckpt "$BUILDER" --n "$NUM_DECK" \
                --temperature "$BUILDER_TEMP" --seed "$SEED" \
                --out-npy "$POOL" || exit 1
        else
            echo "--- [1b/5] reusing deck pool $POOL ---"
        fi
        echo "--- [1b/5] self-play $NUM_SELFPLAY games (temps $PLAYER_TEMP) ---"
        # --workers explicitly: selfplay_replay.py's own default is
        # cpu_count-2, and nothing on the server should be capped.
        "$PY" "$ROOT/selfplay/selfplay_replay.py" \
            --decks "$POOL" --player "$SP_PILOT" --engine "$ENGINE" \
            --out "$SELFPLAY_DIR" --games "$NUM_SELFPLAY" \
            --workers "$TOTAL_CORES" \
            --seed "$SEED" --play-temps "$PLAYER_TEMP" || exit 1
    fi
    echo "self-play replays: $(find "$SELFPLAY_DIR" -name '*.json' | wc -l | tr -d ' ')"
fi

# ---- 2. convert, SEQUENTIALLY, every core ------------------------------
convert() {                 # convert <npz> <spec> <label> <mix>
    local npz="$1" spec="$2" label="$3" mix="$4" extra=() srcs=("$REPLAYS")
    [ "$spec" = "winners" ] && extra=(--winners-only)
    [ "$mix" = "1" ] && srcs+=("$SELFPLAY_DIR")
    if [ -f "$npz" ]; then
        echo "--- $label: $npz exists, skipping (delete to redo) ---"
        return 0
    fi
    echo "--- $label: converting ($spec$([ "$mix" = "1" ] && echo " + self-play")) with $TOTAL_CORES workers ---"
    "$PY" "$ROOT/sequence/replay_to_trajectories.py" \
        --replays "${srcs[@]}" --out "$npz" --workers "$TOTAL_CORES" \
        ${extra[@]+"${extra[@]}"} || return 1
}

echo "--- [2/5] converting datasets (sequential, all $TOTAL_CORES cores) ---"
convert "$NPZ0" "$SPEC0" "GPU0/$GPU_0_TRAINER" "$MIX0" || exit 1
if [ "$SHARED" = "1" ]; then
    if [ ! -f "$NPZ1" ]; then
        echo "--- GPU1/$GPU_1_TRAINER: same '$SPEC1' data, linking to GPU0's ---"
        ln -sf "$NPZ0" "$NPZ1"
    fi
else
    convert "$NPZ1" "$SPEC1" "GPU1/$GPU_1_TRAINER" "$MIX1" || exit 1
fi

# ---- 3. verify BOTH before any GPU starts ------------------------------
echo
echo "--- [3/5] verifying datasets ---"
V0=(); [ "$SPEC0" = "winners" ] && V0=(--winners-only)
V1=(); [ "$SPEC1" = "winners" ] && V1=(--winners-only)
"$PY" "$ROOT/tools/verify_data.py" "$NPZ0" ${V0[@]+"${V0[@]}"} \
    --label "GPU0/$GPU_0_TRAINER" || {
    echo "ABORT: GPU0 dataset is not usable; no training started." >&2
    exit 1; }
"$PY" "$ROOT/tools/verify_data.py" "$NPZ1" ${V1[@]+"${V1[@]}"} \
    --label "GPU1/$GPU_1_TRAINER" || {
    echo "ABORT: GPU1 dataset is not usable; no training started." >&2
    exit 1; }
echo "both datasets verified -- starting training"

# ---- 4. two trainers, one per GPU, half the cores each -----------------
PIN=""
command -v taskset >/dev/null && PIN="taskset"

launch() {                  # launch <gpu> <trainer> <npz> <core_lo> <core_hi> <init>
    local gpu="$1" tname="$2" npz="$3" lo="$4" hi="$5" init="$6"
    local script ckpts out logf flags=()
    script="$(trainer_script "$tname")"
    ckpts="$DATA_DIR/epoch_ckpts_gpu${gpu}_${tname}"
    out="$DATA_DIR/best_gpu${gpu}_${tname}.pt"
    logf="$DATA_DIR/train_gpu${gpu}_${tname}.log"
    mkdir -p "$ckpts"
    while IFS= read -r line; do [ -n "$line" ] && flags+=("$line"); done \
        < <(trainer_flags "$tname")
    [ -n "$init" ] && flags+=(--init "$init")

    local pre=(env "CUDA_VISIBLE_DEVICES=$gpu"
               "OMP_NUM_THREADS=$((hi - lo + 1))"
               "MKL_NUM_THREADS=$((hi - lo + 1))")
    [ -n "$PIN" ] && pre+=(taskset -c "${lo}-${hi}")

    echo "  GPU$gpu $tname : cores $lo-$hi -> $out"
    echo "          log $logf"
    "${pre[@]}" python3 "$script" \
        --data "$npz" --out "$out" --epochs "$EPOCHS" \
        --decisions-per-batch "$DECISIONS_PER_BATCH" --lr "$LR" \
        --mem-layers "$MEM_LAYERS" --epoch-ckpt-dir "$ckpts" \
        ${flags[@]+"${flags[@]}"} > "$logf" 2>&1 &
    LAUNCHED_PID=$!          # global: do NOT print, the caller must not
}                            # capture stdout (progress lines live there)

echo
echo "--- [4/5] launching trainers ---"
[ -z "$PIN" ] && echo "  (taskset unavailable: using thread limits only)"
CK0="$DATA_DIR/epoch_ckpts_gpu0_${GPU_0_TRAINER}"
CK1="$DATA_DIR/epoch_ckpts_gpu1_${GPU_1_TRAINER}"
launch 0 "$GPU_0_TRAINER" "$NPZ0" 0 $((HALF - 1)) "$INIT_0"
PID0=$LAUNCHED_PID
launch 1 "$GPU_1_TRAINER" "$NPZ1" "$HALF" $((TOTAL_CORES - 1)) "$INIT_1"
PID1=$LAUNCHED_PID

# ---- 5. one sequential evaluator over both ----------------------------
echo
echo "--- [5/5] starting evaluator (sequential, all cores per tournament) ---"
EVAL_DECK="$EVAL_DECK" R2="$R2" ENGINE="$ENGINE" \
EVAL_GAMES="$EVAL_GAMES" REVIEWS_BASE="$ROOT/reviews_dual" \
    bash "$HERE/watch_eval_dual.sh" \
        "$CK0" "gpu0_$GPU_0_TRAINER" "$CK1" "gpu1_$GPU_1_TRAINER" &
WATCHER=$!
trap 'kill $PID0 $PID1 $WATCHER ${MONITOR:-} 2>/dev/null' EXIT

# ---- combined progress line: whichever GPU advances prints a new line,
# ---- the other just shows its latest status. Read-only, blocks neither.
LOG0="$DATA_DIR/train_gpu0_${GPU_0_TRAINER}.log"
LOG1="$DATA_DIR/train_gpu1_${GPU_1_TRAINER}.log"
"$PY" "$ROOT/tools/dual_monitor.py" \
    "$LOG0" "$GPU_0_TRAINER" "$LOG1" "$GPU_1_TRAINER" \
    --poll "${MONITOR_POLL:-5}" &
MONITOR=$!

echo
echo "full logs:  tail -f $LOG0"
echo "            tail -f $LOG1"
echo

wait "$PID0"; RC0=$?
wait "$PID1"; RC1=$?
kill "$MONITOR" 2>/dev/null
echo "--- GPU0 $GPU_0_TRAINER exited ($RC0) | GPU1 $GPU_1_TRAINER exited ($RC1) ---"
echo "--- waiting for the evaluator to drain ---"
wait "$WATCHER" 2>/dev/null
trap - EXIT

echo
echo "=== DUAL-GPU RUN DONE ==="
echo "  GPU0 $GPU_0_TRAINER : $DATA_DIR/best_gpu0_${GPU_0_TRAINER}.pt"
echo "  GPU1 $GPU_1_TRAINER : $DATA_DIR/best_gpu1_${GPU_1_TRAINER}.pt"
echo "  reviews             : $ROOT/reviews_dual/"
[ "$RC0" -eq 0 ] && [ "$RC1" -eq 0 ]
