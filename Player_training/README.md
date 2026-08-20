# seq_trial — decision-level vs full-trajectory, A/B

Two player designs trained on the **same replay files** with the **same loss,
optimiser, lr, epochs and decisions-per-batch**, then played head to head.

| arm | sample | context per decision |
|---|---|---|
| `baseline/` | one decision, shuffled across all games | that decision's tokens only |
| `sequence/` | one whole trajectory, in order | deck + every earlier decision of the same game |

Trajectory format in the sequence arm:

```
DECK -> obs_1 -> act_1 -> obs_2 -> act_2 -> ... -> obs_N -> act_N -> win/lose
```

Both arms share `common/encoder.py` (identical per-decision encoder) and
`common/player_loss` (identical reward weighting). The sequence arm adds a
deck embedding and a 2-layer causal transformer over the game's decision
summaries. That memory is the only difference.

## Data

3 days picked at random from `manifest.csv`, seed `20260805`:

| date | episodes | GB |
|---|---|---|
| 2026-06-24 | 5,516 | 21.5 |
| 2026-07-18 | 4,811 | 21.5 |
| 2026-08-01 | 4,518 | 21.5 |
| **total** | **14,845** | **64.4** |

Reproduce or reroll:

```bash
python3 pick_days.py --seed 20260805 --days 3
```

## Run

**1. Download the replays**

```bash
./download_days.sh replays
```

**2. Convert once — both arms read this one file**

```bash
python3 sequence/replay_to_trajectories.py --replays replays --out trajectories.npz
```

**3. Train both arms**

```bash
python3 baseline/train_baseline.py --data trajectories.npz --out baseline/baseline.pt --epochs 4
```

```bash
python3 sequence/train_seq.py --data trajectories.npz --out sequence/sequence.pt --epochs 4
```

`--decisions-per-batch` defaults to 256 in both and must stay equal.

**4. Head to head, 1000 games, same deck both sides, seats swapped**

```bash
python3 eval/tournament_seq.py --engine ../submission_r2_t07_torch --deck ../submission_r2_t07_torch/deck.csv --a baseline/baseline.pt --b sequence/sequence.pt --games 1000
```

Prints win counts, draws, and a two-proportion z-test (|z| >= 1.96 is p<0.05).

## Warm start from R2

R2's weights load into both arms exactly. Parameter names differ
(`nn.TransformerEncoderLayer` vs the trial's `CausalBlock`) but the math and
every tensor shape are identical, so `tools/warm_start.py` remaps them.

For the sequence arm, `mem_out` is **zero-initialised**, so the trajectory
context contributes exactly 0 at step 0 — the warm-started sequence model is
numerically R2 on the first batch and only diverges as it learns to use the
memory. Without that, a random memory projection would inject noise straight
into the option scores and throw the warm start away.

Verified: `max |R2 - baseline arm| = 4.8e-06` and `max |R2 - sequence arm| =
4.8e-06` over 60 real decisions — float32 noise.

```bash
python3 tools/verify_r2.py --r2 ../submission_r2_t07_torch/player.pt --replay <any_replay.json>
```

Both trainers take `--init`, which accepts either an R2-era `player.pt` or a
previous round's checkpoint.

## The self-play + kaggle loop

```bash
./run_round.sh 1 ../submission_r2_t07_torch/player.pt 20260805
./run_round.sh 2 rounds/r1/sequence.pt 20260806
```

Each round: sample a fresh deck pool from the builder → 10k self-play games
(mixed play temperature) + 3 fresh kaggle days → one trajectory file → train
from the previous checkpoint. Use a different seed per round so it draws
different days and different decks.

Deck pool comes from `selfplay/build_deck.py --out-npy`, which decodes on the
GPU in lockstep and writes one `(n, 60)` int32 array that
`selfplay_replay.py --decks` takes directly. Defaults: `DECK_N=5000` at
`DECK_TEMPS=0.7,0.8,0.9`, builder `../selfplay_mixed_temp/builder_tf_deck.pt`
— all overridable by env var.

**One incompatibility this fixes.** `selfplay_replay.py` starts recording
*after* `battle_start(deck0, deck1)`, so a self-play replay has no 60-card
list at step 1 — where Kaggle replays keep it. Unpatched, every self-play
replay converts to **zero** trajectories (verified: "no usable trajectories
found"). `selfplay/selfplay_replay.py` now writes an explicit `"decks"`
field and the converter reads either source.

## Endless training

```bash
EVAL_DECK=/path/to/your_deck.csv ./train_forever.sh
```

Runs rounds back to back forever. After each round it plays 1000 games
against R2 on **your** fixed deck and writes the result to
`reviews/round_NNN.txt`.

Resumes automatically: re-running scans `rounds/` and continues after the
last round that produced a checkpoint. `touch STOP` for a clean stop at the
end of the current round; Ctrl-C for an immediate one.

Env: `EVAL_DECK` (required), `R2`, `ENGINE`, `START_ROUND`, `START_SEED`
(20260805), `EVAL_GAMES` (1000), `KEEP_ROUNDS` (2), `MAX_RETRIES` (2), plus
everything `run_round.sh` takes.

`KEEP_ROUNDS` matters — each round pulls ~64 GB of kaggle replays plus 10k
self-play games. Bulk data older than the last 2 rounds is deleted;
checkpoints and reviews are kept forever.

### The tournament cannot touch training

It is human review only, enforced structurally rather than by promise:

- `NEXT_INIT` — the checkpoint round N+1 chains from — is assigned
  **before** the tournament is launched (line 110 vs line 128).
- The tournament's exit status is discarded with `|| true`.
- Its output goes only to `reviews/`, and nothing reads that directory back.
- No file on the training path mentions the tournament, the reviews
  directory, or a win rate.

Round N+1 is identical whether round N's tournament wins 90%, loses 90%, or
crashes. Check it yourself instead of trusting the above:

```bash
python3 tools/check_isolation.py
```

It exits non-zero if anything can leak. It already caught one real hit while
being written — `run_round.sh` was echoing a suggested tournament command —
which is now removed.

To run a review tournament by hand at any time:

```bash
python3 eval/tournament_seq.py --engine ../submission_r2_t07_torch --deck <your_deck.csv> --a ../submission_r2_t07_torch/player.pt --b rounds/r1/sequence.pt --games 1000
```

The tournament loads a raw R2 checkpoint directly, so R2 can be one side.

## Data-transfer mode — parse on one box, train on another

Stops the GPU idling for hours while a CPU parses replays.

```
CPU box (purser)                       GPU box (receiver)
  download kaggle    ~840 GB
  [self-play]                            waiting... polls every 30s
  convert, ALL cores
  verify
  scp the .npz ────── a few GB ───────▶  verify, then train
  (raw replays stay put)
```

**Receiver** (start it any time — it waits):

```bash
cd seq_trial && DATA_TRANSFER=1 PURSER=0 RECEIVER=1 DATA_DIR=~/raw DUAL_GPU=1 GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b EVAL_DECK=../submission_r2_t07_torch/deck.csv ./transfer/run_receiver.sh
```

**Purser**:

```bash
cd seq_trial && DATA_TRANSFER=1 PURSER=1 RECEIVER=0 DEST=user@gpu-box:/home/user/raw DATA_DIR=~/raw DAYS=20 DUAL_GPU=1 GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b ./transfer/run_purser.sh
```

`DEST` is `user@host:` plus the **receiver's** `DATA_DIR`. Works with
`DUAL_GPU=1` or `DUAL_GPU=0 TRAINER=<v>`, and with `MIX_V3A=1`.

Both sides source `transfer/data_paths.sh` and derive the npz paths from
the same training config, so the receiver knows what to wait for without
being told. Only the `.npz` is transferred — raw replays never move.

**A partial transfer cannot start a run.** The purser scp's to
`<name>.npz.incoming` then renames on the far side, so the final name only
appears once the whole file has landed — and the receiver still runs
`verify_data.py` on it, because a rename proves the bytes arrived, not that
the dataset is right. Verified:

```
nothing yet   -> waiting: 1 missing, 0 bad | 4s elapsed
truncated     -> FAILED verification: cannot load (BadZipFile) ... 1 bad, keeps waiting
valid         -> all datasets present and verified: v3b -> handing off to run_bulk.sh
```

The purser also checks ssh **before** downloading, so a missing key costs
seconds rather than hours. Key setup and full env list:
[transfer/README_TRANSFER.md](transfer/README_TRANSFER.md).

## Dual-GPU mode

Two trainers, two GPUs, two models, fully isolated.

```bash
DUAL_GPU=1 DATA_DIR=~/raw DAYS=20 GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b EVAL_DECK=../submission_r2_t07_torch/deck.csv ./bulk/run_dual.sh
```

`DUAL_GPU` defaults to 0; without it the script refuses and points at
`run_bulk.sh`, so single-GPU stays the default everywhere.

**Flow.** Download once → convert each trainer's dataset **sequentially with
every core** → verify both → only then start the GPUs.

```
[1/5] download (shared, skipped if present)
[2/5] convert  GPU0 dataset, then GPU1 dataset -- one at a time, all cores
[3/5] verify   both npz; ABORT before any GPU starts if either is bad
[4/5] launch   GPU0 cores 0..N/2-1, GPU1 cores N/2..N-1
[5/5] evaluate one watcher, sequential, all cores per tournament
```

At 44 cores: conversion uses 44, then GPU0 gets 0-21 and GPU1 gets 22-43
via `taskset` plus `OMP_NUM_THREADS` (falls back to thread limits where
`taskset` is unavailable). `CUDA_VISIBLE_DEVICES` is 0 and 1.

**Datasets.** v3a needs winners-only, everything else needs both sides. Each
trainer gets `$DATA_DIR/<trainer>/trajectories.npz`. If both trainers need
the *same* spec it converts once and symlinks, rather than doing identical
work twice.

**Verification gates training** — `tools/verify_data.py` checks that each
npz loads, carries every array, has monotonic offsets ending at the right
lengths, one 60-card deck per trajectory, ±1 rewards, no empty
trajectories, and (for v3a) that every trajectory is a win. A truncated npz
is caught: `cannot load (BadZipFile)`. Nothing starts unless both pass.

**Combined progress line.** Whichever GPU advances first prints a new line;
the other shows its latest status. Read-only, so neither blocks the other:

```
[00:00:14] GPU0 v3a | epoch 1 train ce 1.8175 out 0.3836 | val ce 2.0161   ||   GPU1 v3b | starting...
[00:00:54] GPU0 v3a | DONE   ||   GPU1 v3b | epoch 1 train loss 1.4229 wce 2.0098 | val loss 0.2618
```

Full logs stay in `$DATA_DIR/train_gpu{0,1}_<trainer>.log`.

**Tournaments** run in one watcher, never concurrently: GPU0's checkpoint vs
R2 first, then GPU1's, each with every core. Reviews go to
`reviews_dual/gpu0_<trainer>/` and `reviews_dual/gpu1_<trainer>/`. Same
review-only rule, and `tools/check_isolation.py` now verifies dual mode too
— including that `verify_data` runs *before* `launch`.

Env: `DUAL_GPU`, `GPU_0_TRAINER`, `GPU_1_TRAINER`, `INIT_0`, `INIT_1`
(empty = cold start), `TOTAL_CORES` (auto), `MONITOR_POLL` (5), plus
everything bulk mode takes.

## Bulk mode — cold start, all days, no self-play

A second training mode. Where round mode warm-starts from R2, mixes in
self-play, and works 3 days at a time, bulk mode takes the **whole corpus in
one pass from random init** — so what it measures is what the trajectory
design learns on its own.

```bash
DATA_DIR=/mnt/big/pokemon EVAL_DECK=../submission_r2_t07_torch/deck.csv ./bulk/run_bulk.sh
```

`DATA_DIR` is the custom storage path for the corpus. It holds:

```
$DATA_DIR/replays/            all 49 days, decompressed
$DATA_DIR/trajectories.npz    one conversion over everything
$DATA_DIR/epoch_ckpts/        epoch_001.pt, epoch_002.pt, ... + TRAINING_DONE
$DATA_DIR/sequence_best.pt    best-val checkpoint
```

Reviews land in `reviews_bulk/epoch_NNN.txt`, one per epoch, automatically.

Corpus: **49 days, 245,680 episodes, 1,033 GB compressed** (`pick_days.py --all`).
At the 6,900 dec/s measured in round 1 that is ~28.2M decisions, ~68 min/epoch.

**Fewer days:** `DAYS=n` takes a random n instead of the whole manifest,
reproducible from `SEED`.

```bash
DATA_DIR=/mnt/big/pokemon EVAL_DECK=../submission_r2_t07_torch/deck.csv DAYS=7 ./bulk/run_bulk.sh
```

```bash
DATA_DIR=/mnt/big/pokemon EVAL_DECK=../submission_r2_t07_torch/deck.csv DAYS=7 SEED=1234 ./bulk/run_bulk.sh
```

Env: `DATA_DIR` and `EVAL_DECK` required; `EPOCHS` (8), `LR` (3e-4),
`DECISIONS_PER_BATCH` (256), `MEM_LAYERS` (2), `EVAL_GAMES` (1000),
`EVAL_WORKERS`, `R2`, `ENGINE`, `SKIP_DOWNLOAD`, `DAYS` (empty = all),
`SEED` (20260805), and `INIT` (empty = cold start; set it to warm-start
instead).

Resumable: re-running skips days already downloaded and skips conversion if
`trajectories.npz` exists. Delete the npz to force a redo, or set
`SKIP_DOWNLOAD=1` to reuse what is on disk.

### Per-epoch evaluation without touching training

The trainer writes `epoch_NNN.pt` after every epoch and a `TRAINING_DONE`
marker at the end. A **separate process**, `bulk/watch_eval.sh`, polls that
directory and plays 1000 games against R2 for each new checkpoint.

The trainer never reads a review; the watcher never writes a checkpoint.
Killing the watcher changes nothing about the model produced.
`tools/check_isolation.py` now checks bulk mode too:

```
bulk mode trainer/evaluator separation:
  OK  run_bulk.sh never reads a review
  OK  trainer invoked with no eval-derived argument
  OK  watch_eval.sh only reads checkpoints, writes only reviews
  OK  watch_eval.sh swallows tournament failures
```

## V3a and V3b — two ways to use the outcome

v3 as shipped is behaviour cloning of *everyone*: with `LOSER_WEIGHT=1.0`
the loser's actions are reinforced exactly as much as the winner's. V3a and
V3b are the two ways to fix that.

**V3a — drop the loser at conversion time.**

```bash
DATA_DIR=~/replay EVAL_DECK=../submission_r2_t07_torch/deck.csv DAYS=20 TRAINER=v3a ./bulk/run_bulk.sh
```

`replay_to_trajectories.py --winners-only` keeps only the winning side of
each replay, so training never sees a losing trajectory at all. Training is
byte-identical to v3. Halves the decisions (verified: 6 trajectories/439
decisions -> 3/266). Gets its own `trajectories_winners.npz` so it cannot
silently reuse a both-sides file.

**V3b — keep both sides, condition on the outcome, punish the loser.**

```bash
DATA_DIR=~/replay EVAL_DECK=../submission_r2_t07_torch/deck.csv DAYS=20 TRAINER=v3b ./bulk/run_bulk.sh
```

Level 2 reads `[WIN|LOSE, deck, s_1..s_N]` — the outcome sits at the head,
so every decision is scored knowing how the game ended. At play time the
token is pinned to WIN. The action loss flips per trajectory:

    winner : -log p(action)        raise what the winner played
    loser  : -log(1 - p(action))   push down what the loser played

Applied to the dense per-step CE, so supervision stays ~150 bits per game
rather than v1's 1. `P_MAX` (0.95) caps the push-down gradient at 19; v1's
unbounded version reached ~1e6 and is what destabilised it.

No outcome-prediction head in v3b — the outcome is an input, so predicting
it would be degenerate.

Verified: the head token changes scores (max |WIN − LOSE| = 0.0469, so the
conditioning is used, not ignored), and incremental play pinned to WIN
matches the training forward to 5.7e-07 over 66 decisions.

Each trainer writes to `epoch_ckpts_<TRAINER>/`, `best_<TRAINER>.pt` and
`reviews_bulk_<TRAINER>/`, so variants can share one `DATA_DIR` and its
downloaded replays without overwriting each other.

## Trainer v3 — dense per-step action, outcome only at the end

`TRAINER=v3`. v1 and v2 are untouched.

v1/v2 stamped the game's +/-1 onto all ~57 decisions and flipped the loss on
it: **1 bit of supervision per game**, with an optimum at the binary entropy
of the win rate. On a 50/50 corpus the floor is H(0.5) = 0.6931 and there is
nothing below it to reach — the observed run sat at 0.7526, above its own
floor.

v3 uses two losses:

- **action** (dense, one per decision) — cross-entropy on the option
  actually taken, conditioned on the deck and every earlier decision. No
  outcome weighting, no sign flip.
- **outcome** (sparse, one per trajectory) — win/lose predicted once, from
  the level-2 position that has read every decision. Never applied to
  intermediate steps.

| | v1/v2 | v3 |
|---|---|---|
| supervision per game | 1 bit | **150 bits** |
| starting loss | — | ln(n_options) = **1.8240** |
| floor | 0.6931 | **0** |

The trainer prints the chance level each epoch, so `action_ce` below 1.82
means it is predicting actions better than random.

`LOSER_WEIGHT` (default 1.0) is how the outcome informs the action loss —
by weighting **whole trajectories**, never by flipping individual steps.
1.0 learns from both sides equally; 0.0 is winners-only behaviour cloning.
`OUTCOME_WEIGHT` (0.1) scales the end-of-trajectory head; 0 disables it.

Verified: the outcome head reads position `n_real`, so padding never shifts
it (game 0 alone `+0.424896`, padded 66 -> 117 slots `+0.424896`, diff
1.8e-07), and incremental play matches the training forward to 5.4e-07 over
66 decisions.

## Trainer v1 vs v2

`train_seq.py` (v1) is untouched. `train_seq_v2.py` is a separate file with
the batching fix, so you can run both and compare.

```bash
DATA_DIR=/mnt/big/pokemon EVAL_DECK=../submission_r2_t07_torch/deck.csv TRAINER=v2 ./bulk/run_bulk.sh
```

`TRAINER` defaults to `v1`, so nothing changes unless you ask for it.

**The bug v2 fixes.** Every decision in a trajectory shares one +/-1 label,
so batching by *game* collapsed the reward-signal batch size. Measured on
realistic trajectory lengths:

| | steps/epoch | games/step | mean \|frac_win-0.5\| | steps 100% one sign |
|---|---|---|---|---|
| original `train_v2.py` | — | ~64 decisions | 0.05 | **0.00%** |
| v1 | 11,920 | 3.9 | 0.21 | **21.0%** |
| v2 `--games-per-step 64` | 709 | 66.3 | **0.049** | **0.00%** |

One update in five was 100% wins or 100% losses. The losing branch's
gradient is `p/(1-p)` — ~1e6 near p=1 — and in the original those spikes
were always counterbalanced inside the same update. v2 keeps 4 games per
forward pass (same memory, same throughput) and accumulates ~16 of them
before stepping.

Trajectories are still read whole. Batching runs them in parallel, not
concatenated: perturbing one decision in game 0 of a 3-game batch changes
game 0's scores by 1.3e-02 and leaves games 1 and 2 bit-identical.

Two secondary changes, independently switchable:

- `P_MAX` (0.95) clamps the unlikelihood branch so its gradient caps at 19
  instead of ~1e6.
- `LR_SCHEDULE` (cosine) + `WARMUP_STEPS` (500). v1 ran constant 3e-4 for
  ~50k steps with no warmup or decay.

Reproduce v1 exactly from v2, to confirm nothing else drifted:

```bash
python3 sequence/train_seq_v2.py --data traj.npz --out x.pt --games-per-step 4 --p-max 0.999999 --lr-schedule none --warmup-steps 0
```

Note this also un-confounds the A/B: `train_baseline.py` batches 256
*shuffled decisions*, so it always had ~256 independent reward draws per
step while the sequence arm had 4. v1 vs baseline was partly measuring
batch statistics, not memory.

## Layout

```
common/     encoder.py (shared), player_vocab.py, card_vocab.py,
            builder_model.py, EN_Card_Data.csv
baseline/   baseline_model.py, train_baseline.py
sequence/   seq_model.py, seq_model_v3.py, seq_model_v3b.py,
            replay_to_trajectories.py (--winners-only for v3a),
            train_seq.py (v1), train_seq_v2.py (v2),
            train_seq_v3.py (v3 / v3a), train_seq_v3b.py (v3b)
selfplay/   build_deck.py, selfplay_replay.py (patched), selfplay_loop.py
bulk/       run_bulk.sh, watch_eval.sh          <- bulk mode
            run_dual.sh, watch_eval_dual.sh    <- dual-GPU mode
transfer/   run_purser.sh, run_receiver.sh,    <- data-transfer mode
            data_paths.sh, README_TRANSFER.md
tools/      warm_start.py, verify_r2.py, verify_data.py,
            dual_monitor.py, check_isolation.py
tools/      warm_start.py, verify_r2.py
eval/       tournament_seq.py, tournament_tf_orig.py (unmodified original)
train_forever.sh, run_round.sh, pick_days.py, download_days.sh,
manifest.csv
```

`tools/check_isolation.py` is the guard on the review-only rule; run it
after any edit to the training path or the loop.

## Verified before packaging

- Converter run on 3 real replays: 6 trajectories, 439 decisions, mean 73.2
  decisions/trajectory, 108 tokens/decision.
- Both trainers complete epochs and checkpoint. 4.11M params baseline,
  5.89M sequence.
- Incremental inference matches the training forward to **6.0e-07** max
  absolute difference over all option scores in a 66-decision game — the
  tournament runs the model that was trained.
- Trajectory memory measurably changes scores (0.59 max delta on the same
  decision with full history vs none), so the context is not being ignored.
- Both `Policy` wrappers load their checkpoint and return legal selections
  on 40 real observations.

## Two implementation notes

**Attention.** `common/encoder.py` uses
`F.scaled_dot_product_attention(is_causal=True)` instead of an explicit
`L x L` float mask. Mathematically identical; the explicit mask disqualifies
the fused kernel and makes PyTorch materialise a `B x H x L x L` tensor
(~0.8 GB per layer at B=64, L=640), which is what pushes long-sequence
batches into CUDA OOM. Applied to **both** arms, so it is not a confound.

**Val split.** Both arms split train/val **by game**. A decision-level split
leaks, because consecutive states inside one game are nearly identical. The
tournament is the real comparison; val loss is only a training monitor and
is not comparable across the two arms (different sample definitions).
