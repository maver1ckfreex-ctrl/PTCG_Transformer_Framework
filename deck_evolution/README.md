# deck_evolution_v3a

Same evolutionary deck search as `../deck_evolution`, piloted by a v3a
(`arch: sequence_v3`) checkpoint instead of the old `PlayerDecoder`.

The original is untouched. This is a separate folder so the two can be run
side by side on the same builder.

## Why a new file was needed

A v3a checkpoint cannot be loaded by the old script at all — 87 tensors
under `enc / mem_blocks / mem_pos / mem_norm / mem_out / deck_proj / score /
value` against `PlayerDecoder`'s 54 `embed.* / blocks.layers.N.*`. Beyond
the names, v3a is two-level (per-decision encoder plus a causal memory over
decision summaries) and takes the decklist as a prefix, so it needs per-game
state that the old `pick(obs)` had nowhere to keep.

This version drives the pilot through `Policy` from
`../seq_trial/eval/tournament_seq.py`, which already owns that memory and
exposes `reset(deck)` + `select(obs)`. It holds **two** policies, one per
seat: a single memory chain shared across both sides would interleave two
games and corrupt both.

## The selection mechanism is unchanged

At temperature 0, `Policy.select` reproduces the old `pick()` decision rule
exactly — same decline-option handling, same `minCount`/`maxCount` fill
order, same random fallback when tokenization fails. Ranking, elimination,
refill, manifest columns and deck-deletion behaviour are all carried over
verbatim. Temperature is the only thing layered on top, and it is off by
default.

## Run

```bash
cd deck_evolution_v3a && python3 deck_evolution_v3a.py --builder ../deck_evolution/builder_tf_deck.pt --player ../seq_trial/best_v3a.pt --engine ../submission_r2_t07_torch --NUM_DECK 100 --NUM_CYCLE 8 --games-per-deck 1000 --player_temp_mix 1 --workers 126
```

## Flags

| flag | default | meaning |
|---|---|---|
| `--player` | required | v3a checkpoint; warns if `arch != sequence_v3` |
| `--builder` | required | builder checkpoint for deck sampling |
| `--engine` | required | folder containing `cg/` |
| `--NUM_DECK` | `100` | pool size per cycle |
| `--NUM_CYCLE` | `8` | number of cycles |
| `--deck_ratio` | `0.7,0.8,0.9` | builder temperatures; the pool is split evenly (100 → 34/33/33) |
| `--player_temp_mix` | `0` | `0` = pilot fixed at 0.0 (greedy, matches the old script); `1` = mix |
| `--player_ratio` | `0.0,0.1,0.2` | pilot temperatures, equal share each; only read when `--player_temp_mix=1` |
| `--games-per-deck` | `1000` | games each deck plays per cycle |
| `--survivors` | `NUM_DECK//4` | decks kept per cycle — the old 9-of-36 fraction |
| `--workers` | cores − 2 | tournament processes |
| `--seed` | `20260803` | master seed; every deck seed derives from it |
| `--tf-dir` | `../deck_evolution` | where `build_deck.py` / `builder_model.py` live |
| `--seq-dir` | `../seq_trial` | seq_trial root, for `eval/tournament_seq.py` |
| `--out` | `./evolution_v3a` | manifest + surviving decks |

## How the temperature mix is assigned

Per **round**, not per game. Every deck plays exactly one pair per round, so
cycling the temperature by round gives every deck an identical spread —
measured at `NUM_DECK=100`, `--games-per-deck 1000`: all 100 decks get
334/334/332 games at 0.0/0.1/0.2. Assigning per game instead leaves a ±5%
per-deck imbalance, which would confound deck strength with how noisy its
pilot happened to be. Both seats of a seat-swapped pair always share a
temperature.

## Memory

`Policy` loads a ~23 MB model. Two seats × N workers would be 2N copies, so
the second seat's policy is re-pointed at the first's weights and given its
own runner — the memory chain is per seat, the weights are not. At 126
workers that is ~2.9 GB instead of ~5.8 GB.

## Notes

- `deck_evolution/card_vocab.py` and `seq_trial/common/card_vocab.py` are
  the same module under the same name, differing only in their default CSV
  path (both resolve to identical data). The builder imports happen in the
  parent process and the `Policy` imports only inside spawned workers, so
  the two never collide in one interpreter.
- Deck filenames use `t70/t80/t90` (temperature × 100) rather than the old
  `t7/t8/t9`, so `--deck_ratio` values like `0.75` stay distinguishable.
- Eliminated decks are deleted from disk; the manifest keeps their seed,
  which is enough to rebuild any of them:
  `python3 ../deck_evolution/build_deck.py --ckpt <builder> --n 1 --temperature <t> --seed <seed>`
