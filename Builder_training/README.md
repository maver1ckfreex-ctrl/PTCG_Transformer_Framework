# deck_builder_tf_v2_builder_only

`deck_builder_tf_v2` with the builder phase and nothing else. Kaggle
replays in, builder checkpoint out.

**The training mechanism is unchanged.** Every Python file here is
byte-identical to its counterpart in `deck_builder_tf_v2` — verified by
md5, not by eye:

| file | role |
|---|---|
| `replay_to_dataset.py` | replays → `dataset.npz` (reverse-read deck sequences) |
| `train_builder.py` | `dataset.npz` → builder checkpoint |
| `build_deck.py` | sample legality-masked 60-card decks |
| `builder_model.py` | `DeckDecoder` + `reward_weighted_loss` |
| `card_vocab.py` | card vocabulary |
| `pipeline.py` | streaming Kaggle download → convert → delete |

The only new file is `run_builder_data.sh`, a wrapper — it changes no
training code.

## Not included

Everything on the player side: `replay_to_decisions.py`,
`player_model.py`, `player_vocab.py`, `train_v2.py`, `train_unified.py`,
`export_player_npz.py`, `tf_infer.py`, `submission_main*.py`,
`audit_coverage.py`, and every `.pt` / `.npz` checkpoint. Use
`deck_builder_tf_v2` for any of those.

## Card data

`EN_Card_Data.csv` **is** shipped in this folder, because `card_vocab.py`
resolves its default as `__file__.parent.parent / "EN_Card_Data.csv"` — one
level ABOVE the folder. That only exists while this folder sits beside the
shared copy in `pokemon-tcg-ai-battle/`. Move it to a server or a scratch
disk and every converter dies with

    FileNotFoundError: .../EN_Card_Data.csv

`run_builder_data.sh` therefore passes `--cards <folder>/EN_Card_Data.csv`
automatically unless you name your own, and refuses to start if neither
exists. `card_vocab.py` itself is untouched.

Running the converters directly, outside the wrapper, from a moved folder:
pass `--cards ./EN_Card_Data.csv` yourself. `build_deck.py` takes the same
flag. `train_builder.py` does not need it — vocab size comes from the npz.

## Use

```bash
python3 replay_to_dataset.py --replays /path/to/replays --out dataset.npz --workers 96
```

```bash
python3 train_builder.py --data dataset.npz --epochs 30 --out builder_tf.pt
```

```bash
python3 build_deck.py --ckpt builder_tf.pt --n 5 --temperature 0.9
```

`--append` on `replay_to_dataset.py` accumulates across replay folders, as
before.

## Full Kaggle archive

```bash
./run_builder_data.sh --manifest manifest_01Aug.csv --work /local/ptcg_work
```

`pipeline.py` is copied byte-identical, so its `--convert` default is still
`both` — which would call `replay_to_decisions.py`, the player-phase
converter that does not exist here. The wrapper pins `--convert dataset` and
rejects an explicit `--convert`, so the player phase cannot be reached by
accident. Every other flag (`--start-date`, `--end-date`, `--prefetch`,
`--keep-raw`, `--dataset-out`, …) passes straight through.

Calling `pipeline.py` directly still works if you pass `--convert dataset`
yourself.

## Staged parsing — `--stage_pursing`

`--stage_pursing=1` (default `0` = off) parses the archive in blocks
instead of one long append:

```bash
./run_builder_data.sh --stage_pursing=1 --split_days=20 --manifest manifest_01Aug.csv --work /local/ptcg_work
```

```
stage 001  days 1-20    download -> parse -> stages/stage_001.npz -> verify -> raw deleted
stage 002  days 21-40   download -> parse -> stages/stage_002.npz -> verify -> raw deleted
...
merge_dataset.py stages/stage_*.npz  ->  dataset.npz
```

`--split_days` defaults to 20. Raw replays are deleted as each day is
converted (pipeline.py already does that), so disk never holds more than
one block, and a block is only counted as done once its npz loads and its
arrays agree — a truncated checkpoint is caught here rather than at merge
time.

Why bother: a single `--append` run rewrites the whole `dataset.npz` on
every day, so the per-day cost climbs with the archive, and a crash on day
300 leaves one half-written file. Staged runs cap both.

Resume is free. A stage whose npz verifies is skipped; inside a block
`pipeline_state.json` still tracks finished days, so an interrupted block
restarts where it stopped. If a stage file has to be rebuilt, its days are
first removed from `pipeline_state.json` — otherwise pipeline.py would see
them as already done and write nothing.

Extra flags: `--keep-stages` (don't delete the per-stage npz after
merging), `--merge-only` (skip downloading, just merge what's on disk),
`--stages-dir`.

The merge is a plain concatenation — verified row-for-row identical to what
a single-pass run over the same replays produces. It refuses to merge
stages that consumed the same episode ids (double-counting) or that were
built against different card data.

## Smoke test run

6 replays → `replay_to_dataset.py` → 11 sequences (6 win / 5 loss) →
`train_builder.py` 1 epoch → checkpoint saved → `build_deck.py` sampled two
legal 60-card decks. Full chain, no missing imports.
