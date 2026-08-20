# V3A mix-ratio cycles

Iterative self-play at a fixed kaggle:selfplay trajectory ratio, warm
starting from your base model.

```bash
KAGGLE_NPZ=~/raw/trajectories_winners.npz INIT=~/base.pt BUILDER=~/builder_tf_deck.pt EVAL_DECK=../submission_r2_t07_torch/deck.csv RATIO=1:1 ./mix/run_mix_cycles.sh
```

## Per cycle

```
1. sample NUM_DECK decks from the builder
2. self-play NUM_SELFPLAY games with the CURRENT model
3. convert -> selfplay winners npz, VERIFY, delete the raw json
4. n = self-play winning trajectories
5. randomly draw n * (K/S) trajectories from the kaggle npz
6. merge -> training_cycle_<N>.npz, VERIFY
7. train, warm-started from the previous cycle
8. 1000-game review tournament vs R2          [REVIEW ONLY]
repeat
```

`RATIO=K:S` is kaggle : self-play **by trajectory count**:

| RATIO | self-play | kaggle drawn | self-play share |
|---|---|---|---|
| 1:1 (default) | 9,800 | 9,800 | 50% |
| 2:1 | 9,800 | 19,600 | 33% |
| 1:2 | 9,800 | 4,900 | 67% |
| 5:1 | 9,800 | 49,000 | 17% |

The kaggle side is drawn **fresh each cycle** with a per-cycle seed, so
successive cycles see different slices of the corpus.

## Self-play pilots the current model

`selfplay_replay.py` can only load R2-era checkpoints, which makes
iterative self-play impossible past cycle 1. `selfplay/selfplay_seq.py`
uses the same `Policy` wrapper the tournament uses, so r2 / baseline /
sequence / sequence_v3 / sequence_v3b all work.

It runs **two policies, one per seat**. A trajectory model carries per-game
memory conditioned on its own deck and its own earlier decisions; sharing
one runner across both seats would interleave two games into one memory
chain and corrupt both.

Verified: two seats given different decks and histories score the same
probe decision 0.1043 apart (independent memory), and `reset()` returns a
policy bit-identical to a freshly constructed one (0.00e+00).

## Review is still review-only

`NEXT_INIT` is fixed before the tournament runs (line 202 vs 219), the exit
status is discarded, and the report content is never consumed — only its
existence is tested, to skip a review already written on resume.
`tools/check_isolation.py` verifies all three.

## Resume

Every stage is checkpointed: deck pool, self-play npz, mix npz and model
are each skipped if present. Re-running continues after the last cycle that
produced a `model.pt`. `touch $WORK/STOP` for a clean stop between cycles.

## Env

`KAGGLE_NPZ` `INIT` `BUILDER` `EVAL_DECK` required. `RATIO` (1:1),
`NUM_SELFPLAY` (10000), `NUM_DECK` (2000), `BUILDER_TEMP` (0.7,0.8,0.9),
`PLAYER_TEMP` (0.0,0.1,0.2), `CYCLES` (0 = until STOP), `EPOCHS` (4),
`LR` (1e-4), `EVAL_GAMES` (1000), `EVAL_WORKERS`, `WORKERS` (all cores),
`WORK` (~/mix_run), `SEED`, `PY`.
