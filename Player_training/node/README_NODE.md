# Node training

Self-play on the CPU box, training on the GPU box, one cycle at a time.

```
GPU node                              CPU node
  push pilot.pt + REQUEST  ────────▶    (polling)
  wait for SELFPLAY_DONE                sample decks
                                        self-play NUM_SELFPLAY games
                                        convert -> winners npz, VERIFY
                                        delete raw json
                              ◀────     touch SELFPLAY_DONE
  pull selfplay_winners.npz
  verify
  draw kaggle side at RATIO, merge
  train
  1000-game review vs R2   [REVIEW ONLY]
  repeat
```

**Every network operation is initiated by the GPU node** — it pushes, it
polls, it pulls. The CPU node only touches its own local disk. That is
deliberate: the GPU box usually has no inbound port, so nothing can connect
*to* it.

## Run

CPU node first (it waits):

```bash
cd seq_trial && NODE_TRAINING=1 CPU_NODE=1 GPU_NODE=0 PY=$(which python) WORK=~/node_work BUILDER=~/builder_c1.pt ./node/run_cpu_node.sh 2>&1 | tee ~/cpu_node.log
```

GPU node:

```bash
cd seq_trial && NODE_TRAINING=1 GPU_NODE=1 CPU_NODE=0 PY=$(which python) PEER=ubuntu@CPU_HOST REMOTE_WORK=/home/ubuntu/node_work WORK=~/node_work KAGGLE_NPZ=~/raw/trajectories_winners.npz INIT=~/base.pt EVAL_DECK=../submission_r2_t07_torch/deck.csv RATIO=1:1 NUM_SELFPLAY=21000 ./node/run_gpu_node.sh 2>&1 | tee ~/gpu_node.log
```

The GPU node needs passwordless ssh to the CPU node, and checks it before
starting a cycle.

## Handshake

`REQUEST` is written **after** `pilot.pt` lands, so its presence means the
model is complete. `SELFPLAY_DONE` is written **after** the npz is built
and verified, so its presence means the result is complete. Both sides also
use `.incoming` temp names plus rename, so a partial file never carries a
final name.

`REQUEST` is a few lines of `KEY=VALUE` — the "very short message":

```
NUM_SELFPLAY=21000
NUM_DECK=2000
SEED=20260806
BUILDER_TEMP=0.7,0.8,0.9
PLAYER_TEMP=0.0,0.1,0.2
```

The GPU node deletes the remote cycle directory after the cycle is banked
locally, so the CPU node's disk does not grow.

## Review is still review-only

`NEXT_INIT` is fixed before the tournament (line 243 vs 260), the exit
status is discarded, and the **CPU node contains no reference to the
tournament, the reviews directory, or a win rate at all**.
`tools/check_isolation.py` verifies all of it.

## Resume

Both sides resume. The GPU node continues after the last cycle with a
`model.pt`, and skips any stage already present (npz pulled, mix built,
model trained). The CPU node skips a cycle that already has
`SELFPLAY_DONE`, and reuses replays already on disk within a cycle.
`touch $WORK/STOP` on either box for a clean stop.

## Env

**CPU node:** `NODE_TRAINING=1 CPU_NODE=1 GPU_NODE=0`, `BUILDER` required,
`WORK` (~/node_work), `ENGINE`, `POLL` (15), `WORKERS` (all cores), `PY`.

**GPU node:** `NODE_TRAINING=1 GPU_NODE=1 CPU_NODE=0`, `PEER`,
`REMOTE_WORK`, `KAGGLE_NPZ`, `INIT`, `EVAL_DECK` required. `RATIO` (1:1),
`NUM_SELFPLAY` (10000), `NUM_DECK` (2000), `BUILDER_TEMP`, `PLAYER_TEMP`,
`CYCLES` (0 = until STOP), `EPOCHS` (4), `LR` (1e-4), `EVAL_GAMES` (1000),
`POLL` (30), `SSH_PORT` (22), `SSH_KEY`, `WORK`, `SEED`, `PY`.

Deck sampling happens on the CPU node, so `BUILDER` lives there; the GPU
node only sends the temperatures and counts.
