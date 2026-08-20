#!/usr/bin/env bash
#
# Shared dataset-path logic for data-transfer mode.
#
# Both the purser and the receiver source this and derive the SAME list of
# npz paths from the SAME env vars. That is what makes the handoff work:
# the purser writes files to exactly the paths the trainer will look for,
# so the receiver never needs to be told what is coming.
#
# Sets DATASETS to lines of:   <relative/path.npz>|<spec>|<label>
#   spec  = winners | both     (winners -> converter runs --winners-only)
#   label = what to call it in logs
#
# Requires: DUAL_GPU, and either GPU_0_TRAINER/GPU_1_TRAINER (dual) or
#           TRAINER (single). MIX_V3A optional.

_spec_of()  { [ "$1" = "v3a" ] && echo "winners" || echo "both"; }
_mix_of()   { [ "${MIX_V3A:-0}" = "1" ] && [ "$1" = "v3a" ] && echo 1 || echo 0; }

# npz basename must match what run_bulk.sh / run_dual.sh look for
_npz_of() {                       # _npz_of <trainer> <dual?>
    local t="$1" dual="$2" mix base
    mix="$(_mix_of "$t")"
    if [ "$mix" = "1" ]; then base="trajectories_mix.npz"
    elif [ "$t" = "v3a" ];  then base="trajectories_winners.npz"
    else                          base="trajectories.npz"
    fi
    # dual mode puts each trainer's npz under DATA_DIR/<trainer>/
    if [ "$dual" = "1" ]; then
        # dual always uses trajectories.npz / trajectories_mix.npz
        [ "$mix" = "1" ] && echo "$t/trajectories_mix.npz" \
                         || echo "$t/trajectories.npz"
    else
        echo "$base"
    fi
}

compute_datasets() {
    DATASETS=""
    if [ "${DUAL_GPU:-0}" = "1" ]; then
        local t0="${GPU_0_TRAINER:?}" t1="${GPU_1_TRAINER:?}"
        DATASETS="$(_npz_of "$t0" 1)|$(_spec_of "$t0")|GPU0/$t0"
        local d1="$(_npz_of "$t1" 1)|$(_spec_of "$t1")|GPU1/$t1"
        # identical target path -> one dataset, not two
        [ "${d1%%|*}" = "${DATASETS%%|*}" ] || DATASETS="$DATASETS
$d1"
    else
        local t="${TRAINER:-v1}"
        DATASETS="$(_npz_of "$t" 0)|$(_spec_of "$t")|$t"
    fi
}

# how many distinct datasets, and their relative paths
dataset_paths() { compute_datasets; echo "$DATASETS" | cut -d'|' -f1; }
