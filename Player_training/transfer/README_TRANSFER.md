# Data-transfer mode

Split parsing from training across two boxes, so the GPU never sits idle
for four hours while a CPU chews through replays.

```
CPU box (purser)                          GPU box (receiver)
  download kaggle          ~840 GB
  [self-play]                                waiting... (polls every 30s)
  convert, ALL cores
  verify
  scp the .npz only  ──────── a few GB ───▶  verify, then train
  (raw replays stay here)
```

Only the converted `.npz` crosses the wire. Raw replays never leave the
purser.

## 1. SSH key setup (once)

`scp` needs to log in without a password. On the **purser**:

```bash
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
```

Press enter through any prompt. Then copy the public half to the GPU box:

```bash
ssh-copy-id user@gpu-box
```

It asks for the GPU box's password once — that's the last time. Test it:

```bash
ssh user@gpu-box 'echo it works'
```

If that prints `it works` with no password prompt, you're done. Non-standard
port: add `-p 2222` to `ssh-copy-id` and set `SSH_PORT=2222` below. Custom
key file: `SSH_KEY=/path/to/key`.

`run_purser.sh` checks this **before** downloading anything, so a broken key
costs you seconds, not hours.

## 2. Start the receiver first (GPU box)

It waits, so start it whenever — before or after the purser.

```bash
cd seq_trial && DATA_TRANSFER=1 PURSER=0 RECEIVER=1 DATA_DIR=~/raw DUAL_GPU=1 GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b EVAL_DECK=../submission_r2_t07_torch/deck.csv ./transfer/run_receiver.sh
```

Single GPU instead:

```bash
cd seq_trial && DATA_TRANSFER=1 PURSER=0 RECEIVER=1 DATA_DIR=~/raw DUAL_GPU=0 TRAINER=v3a EVAL_DECK=../submission_r2_t07_torch/deck.csv ./transfer/run_receiver.sh
```

## 3. Start the purser (CPU box)

Same training flags, plus `DEST` and `DAYS`:

```bash
cd seq_trial && DATA_TRANSFER=1 PURSER=1 RECEIVER=0 DEST=user@gpu-box:/home/user/raw DATA_DIR=~/raw DAYS=20 DUAL_GPU=1 GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b ./transfer/run_purser.sh
```

`DEST` is `user@host:` followed by **the receiver's `DATA_DIR`**.

With self-play mixed into v3a:

```bash
cd seq_trial && DATA_TRANSFER=1 PURSER=1 RECEIVER=0 DEST=user@gpu-box:/home/user/raw DATA_DIR=~/raw DAYS=20 DUAL_GPU=1 GPU_0_TRAINER=v3a GPU_1_TRAINER=v3b MIX_V3A=1 BUILDER=../selfplay_mixed_temp/builder_tf_deck.pt ./transfer/run_purser.sh
```

## Split parsing (peak disk control)

Extracting all 54 days at once needs ~1.1 TB. `PURSING_SPLIT=1` processes
the corpus in cycles instead:

```
download ALL zips (sequential, no extraction)   ~38 GB
repeat:
    extract SPLIT_DAYS days   (members in parallel, EXTRACT_JOBS wide)
    convert -> per-cycle npz checkpoint
    VERIFY the checkpoint
    delete that cycle's raw json
merge all checkpoints -> final npz
```

```bash
PURSING_SPLIT=1 SPLIT_DAYS=20 ... ./transfer/run_purser.sh
```

`PURSING_SPLIT` defaults to **0** (extract everything in one pass, raw
kept). `SPLIT_DAYS` defaults to 20. A final short cycle is fine — 54 days
at 20 gives 20 / 20 / 14.

Peak disk becomes `SPLIT_DAYS x 21.5 GB` plus the zips, so 20 days is
~430 GB instead of 1.1 TB.

**Extraction parallelises over zip MEMBERS, not over zips.** One
`unzip archive.zip` is a single thread walking ~5,000 members; listing them
and extracting concurrently is what uses the machine:

```bash
unzip -Z1 day.zip | parallel -j 128 'unzip -q day.zip {}'
```

`EXTRACT_JOBS` (defaults to the core count) sets the width. GNU `parallel`
is used when present, `xargs -P` otherwise. On minimal images `parallel`
prints `ps: command not found` from its load check -- cosmetic, extraction
proceeds normally.

**Raw data is only deleted after its checkpoint verifies.** If conversion
or verification fails, the raw json for that cycle is kept and the run
aborts, so a bad cycle costs one cycle rather than the whole corpus.

Both phases resume: a cycle whose checkpoints already exist is skipped
entirely, and an existing final npz skips the merge.

Verified: split-on and split-off produce **byte-identical** datasets
(same trajectory fingerprints), and `tools/merge_npz.py` output matches a
single-pass conversion exactly.

Applies to every dataset spec, so v3a (winners-only) and v3b (both sides)
each get their own checkpoints from one extraction per cycle.

## How the two sides agree on paths

Both source `transfer/data_paths.sh` and derive the npz paths from the same
training config, so the receiver knows what to wait for without being told:

| config | dataset(s) |
|---|---|
| dual, v3a + v3b | `v3a/trajectories.npz` (winners), `v3b/trajectories.npz` (both) |
| dual, v3a + v3b, `MIX_V3A=1` | `v3a/trajectories_mix.npz` (winners), `v3b/trajectories.npz` (both) |
| single, v3a | `trajectories_winners.npz` |
| single, v3b | `trajectories.npz` |

These are exactly the paths `run_bulk.sh` / `run_dual.sh` look for, so once
the file lands the trainer starts with no conversion.

## Partial transfers cannot start a run

The purser scp's to `<name>.npz.incoming`, then renames on the far side.
The final name only appears once the whole file has arrived. The receiver
**also** runs `tools/verify_data.py` on it — a rename proves the bytes
arrived, not that the dataset is right.

Verified behaviour:

- nothing delivered → waits, reports `1 missing`, times out only if
  `WAIT_TIMEOUT` is set (default 0 = forever)
- truncated file → `cannot load (BadZipFile)`, reported as `1 bad`, keeps
  waiting instead of training on it
- valid file → `all datasets present and verified`, hands off to the trainer

## Env

**Purser:** `DEST` (required), `DATA_DIR`, `DAYS`, `SEED`, `SKIP_DOWNLOAD`,
`SSH_PORT` (22), `SSH_KEY`, `TOTAL_CORES` (auto, uncapped), plus `MIX_V3A`
/ `BUILDER` / `BUILDER_TEMP` / `PLAYER_TEMP` / `NUM_DECK` / `NUM_SELFPLAY`
/ `SP_PILOT`.

**Receiver:** `DATA_DIR`, `POLL` (30), `WAIT_TIMEOUT` (0 = forever), plus
every flag the trainer takes — `EPOCHS`, `LR`, `INIT_0`, `INIT_1`,
`EVAL_GAMES`, and so on. It passes them straight through.

Both need the same `DUAL_GPU` / `GPU_0_TRAINER` / `GPU_1_TRAINER` (or
`TRAINER`) and the same `MIX_V3A`, or they will disagree about which files
are coming.

## Resuming

The purser skips conversion for any npz it already built, and
`SKIP_DOWNLOAD=1` reuses replays already on disk. Re-running after a failed
scp re-sends without re-converting. On the receiver, a dataset that already
verified is not waited for again.
