# PTCG Transformer Framework
Deck Builder || Trajectory Player || Applied Project

![Python](https://img.shields.io/badge/python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-ml-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Status](https://img.shields.io/badge/status-research-orange)

Transformer training framework for the Pokémon TCG: a decoder that builds 60-card decks and a two-level sequence model that plays them.

## Table of contents
- [Overview](#overview)
- [Highlights](#highlights)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Recommended hardware](#recommended-hardware)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration notes](#configuration-notes)
- [Contribution](#contribution)
- [License](#license)
- [Copyright and Ownership](#copyright-and-ownership)

## Overview
This project trains two transformers for the Pokémon TCG. The **builder** is a decoder over the full card pool that generates legal 60-card decks. The **player** is a two-level sequence model: a per-decision encoder produces a summary of each decision, and a causal memory reads every earlier summary in the same game, so a move is chosen with the whole game so far — and the decklist — in context.

Both are trained from public replay archives, and both can be trained further on data the models generate themselves. The pipeline is built for large corpora: replays are streamed in, converted to compact array datasets, and deleted as soon as their dataset is verified, so disk never holds the whole archive at once.

Agents trained with this framework compete in the Kaggle Pokémon TCG AI Battle
competition under the team name **Lennox** — live standings:
<https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard?search=Lennox>

## Highlights
- Two-level player transformer with per-game memory over decision summaries, conditioned on the deck
- Dense per-decision supervision rather than a single win/lose label per game, giving orders of magnitude more signal per replay
- Deck builder trained on decks read backward from the end of the game, so generation decides the win condition first and filler last
- Streaming data pipeline: download → convert → verify → delete, with staged checkpoints and resume
- Self-play training for both models, with temperature mixing over deck sampling and play
- Two-machine mode that separates CPU game generation from GPU training, with all transfers initiated by the GPU side
- Evolutionary deck search that ranks decks by how well the trained player performs with them
- Head-to-head tournaments with seat swapping and a two-proportion z-test
- Dataset tooling that verifies structural consistency before any training run begins

## How it works
1. Stream a replay archive and convert it into datasets — reverse-order deck sequences for the builder, per-decision trajectories for the player.
2. Train the builder on decks weighted by the game's outcome, revealed only after the whole deck is read.
3. Train the player on the action actually taken at every decision, conditioned on the deck and all earlier decisions of that game.
4. Sample decks from the builder and play them with the trained player to produce fresh self-play data.
5. Continue training on that data, optionally mixed with archive data at a chosen ratio.
6. Search for a deck the trained player wins with, by evolving a pool over repeated tournaments.
7. Run head-to-head tournaments with seats swapped and apply a z-test to decide whether a difference is real.

## Repository layout

### Builder_training
- [Builder_training/replay_to_dataset.py](Builder_training/replay_to_dataset.py): replays → builder dataset (reverse-read deck sequences).
- [Builder_training/train_builder.py](Builder_training/train_builder.py): trains the deck decoder.
- [Builder_training/builder_model.py](Builder_training/builder_model.py): the decoder architecture.
- [Builder_training/build_deck.py](Builder_training/build_deck.py): samples legal 60-card decks (4-copy cap, single ACE SPEC, Basic Pokémon requirement).
- [Builder_training/card_vocab.py](Builder_training/card_vocab.py): card vocabulary and legality tables.
- [Builder_training/pipeline.py](Builder_training/pipeline.py): streaming download → unzip → convert → delete.
- [Builder_training/stage_pursing.py](Builder_training/stage_pursing.py): staged parsing in day blocks, with per-block checkpoints.
- [Builder_training/merge_dataset.py](Builder_training/merge_dataset.py): merges staged checkpoints into one dataset.
- [Builder_training/run_builder_data.sh](Builder_training/run_builder_data.sh): entry point for data acquisition.

### deck_evolution
- [deck_evolution/deck_evolution_v3a.py](deck_evolution/deck_evolution_v3a.py): evolutionary deck search, piloted by the V3a player.
- [deck_evolution/build_deck.py](deck_evolution/build_deck.py), [deck_evolution/builder_model.py](deck_evolution/builder_model.py): builder side, used to refill the pool each cycle.
- `deck_evolution.py` is the superseded version driven by the legacy stateless player. Kept for reference only.

### cg
The game engine and card database. **Third party — see [Copyright and Ownership](#copyright-and-ownership).**

### Player_training
- [Player_training/sequence/](Player_training/sequence/): the player models and their trainers, including the trajectory model and its variants.
- [Player_training/common/](Player_training/common/): shared encoder, card and decision vocabularies.
- [Player_training/selfplay/](Player_training/selfplay/): deck sampling and self-play game generation.
- [Player_training/eval/](Player_training/eval/): head-to-head tournaments, same deck or different decks per side.
- [Player_training/tools/](Player_training/tools/): dataset verification, merging, sampling, checkpoint remapping.
- [Player_training/bulk/](Player_training/bulk/): full-archive training runs, single or dual GPU.
- [Player_training/transfer/](Player_training/transfer/): split data preparation from training across two machines.
- [Player_training/mix/](Player_training/mix/): iterative cycles mixing archive and self-play data at a set ratio.
- [Player_training/node/](Player_training/node/): CPU generates, GPU trains, GPU initiates every transfer.
- [Player_training/baseline/](Player_training/baseline/): the stateless per-decision baseline used for comparison.

## Requirements
- Python 3.10+ (3.x supported)
- The game engine `cg/` and the card database `EN_Card_Data.csv`, both included here. They are **not** covered by this project's licence — see [Copyright and Ownership](#copyright-and-ownership)
- The `kaggle` CLI with an API token, only if you use the streaming download pipeline
- Python dependencies:

```text
torch
numpy
```

## Recommended hardware

| Stage | Requirement |
|---|---|
| Deck builder training | NVIDIA GPU, **4 GB VRAM minimum** |
| Player training (V3a) | NVIDIA GPU, **48 GB VRAM minimum** — L40S recommended |
| Deck evolution | **96+ CPU cores**, **64 GB+ RAM** |

Deck evolution and self-play are CPU-bound: they run thousands of full games
per cycle, and throughput scales close to linearly with core count. Player
training is the only stage that needs a large GPU.

## Setup
1. Create and activate a virtual environment:
	```bash
	python -m venv .venv
	source .venv/bin/activate
	```
	Windows (PowerShell):
	```powershell
	.venv\Scripts\Activate.ps1
	```
	Windows (cmd.exe):
	```bat
	.venv\Scripts\activate.bat
	```
2. Install dependencies:
	```bash
	python -m pip install --upgrade pip
	pip install torch numpy
	```
3. Point scripts at the engine with `--engine`, giving the directory that **contains** `cg/` — the repository root, if you keep the layout as shipped.
4. If your interpreter is not `python3`, pass it explicitly: every shell entry point honours `PY=$(which python)`.
5. On high-core machines, the thread-limiting block at the top of each script keeps BLAS from opening one pool per worker; leave it in place.

## Usage

Every entry point honours `PY=$(which python)` if your interpreter is not
`python3`. `--engine` takes the directory containing `cg/`.

### 1. Deck Builder

Build the dataset from a replay archive:
```bash
cd Builder_training && python replay_to_dataset.py --replays /path/to/replays --out dataset.npz --workers 96
```

Or stream a whole archive, parsing in day blocks and deleting raw data as it goes:
```bash
cd Builder_training && ./run_builder_data.sh --stage_pursing=1 --split_days=20 --manifest manifest.csv --work /path/to/work
```

Train:
```bash
cd Builder_training && python train_builder.py --data dataset.npz --epochs 30 --out builder.pt
```

Sample decks:
```bash
cd Builder_training && python build_deck.py --ckpt builder.pt --n 5 --temperature 0.9
```

### 2. Player — V3a

V3a is the current player. The other trainers in `sequence/` are earlier
designs kept for comparison; new work should use `train_seq_v3.py`.

Convert replays to trajectories:
```bash
cd Player_training && python sequence/replay_to_trajectories.py --replays /path/to/replays --out trajectories.npz --winners-only --workers 96
```

Verify before committing to a long run:
```bash
cd Player_training && python tools/verify_data.py trajectories.npz --winners-only
```

Train:
```bash
cd Player_training && python sequence/train_seq_v3.py --data trajectories.npz --out player.pt --epochs 16 --epoch-ckpt-dir epochs/
```
Add `--init existing.pt` to continue from a checkpoint instead of training from scratch. Per-epoch snapshots land in `--epoch-ckpt-dir`.

Generate self-play data with a trained player:
```bash
cd Player_training && python selfplay/selfplay_seq.py --decks pool.npy --player player.pt --engine .. --out replays/ --games 20000 --workers 96
```

Evaluate, seats swapped, with a z-test:
```bash
cd Player_training && python eval/tournament_seq.py --engine .. --deck deck.csv --a baseline.pt --b player.pt --games 2000
```
`eval/tournament_decks.py` gives each side its own deck.

#### V3a — node mode

Splits a training run across two machines: the CPU box generates and parses
games, the GPU box trains. **Every transfer is initiated by the GPU box**, so
the GPU machine needs no inbound port — only the CPU box has to be reachable
over ssh. Both sides resume, and neither can read a partial file.

CPU box, started first — it waits for work:
```bash
cd Player_training && ./node/run_cpu_node.sh
```

GPU box:
```bash
cd Player_training && ./node/run_gpu_node.sh
```

See [Player_training/node/README_NODE.md](Player_training/node/README_NODE.md) for the full environment and the handshake.

### 3. Deck evolution

Searches for a deck the **V3a** player actually wins with: sample a pool from
the builder, play every deck, keep the survivors, refill, repeat. Both sides
of every game are driven by the same V3a checkpoint, so only the decks differ.

```bash
cd deck_evolution && python deck_evolution_v3a.py --builder builder.pt --player player.pt --engine .. --NUM_DECK 100 --NUM_CYCLE 8 --games-per-deck 1000 --workers 126
```

| flag | default | meaning |
|---|---|---|
| `--NUM_DECK` | `100` | pool size per cycle |
| `--NUM_CYCLE` | `8` | number of cycles |
| `--deck_ratio` | `0.7,0.8,0.9` | builder temperatures; the pool is split evenly across them |
| `--games-per-deck` | `1000` | games each deck plays per cycle |
| `--survivors` | `NUM_DECK//4` | decks kept per cycle |
| `--player_temp_mix` | `0` | `1` mixes pilot temperatures from `--player_ratio` |
| `--base_deck` / `--start_at` | off | keep the first `start_at-1` cards of an existing deck and generate the rest |

Surviving decks and a manifest are written to `--out`. Every deck keeps its
seed, so any deck in the manifest can be rebuilt exactly.

## Configuration notes
- Datasets are stored as flat arrays plus offset arrays. Every tool that filters, samples or merges them rebuilds the offsets, and `tools/verify_data.py` checks that result before training accepts it.
- Deck sequences are read from the end of the game backward: final board first, then cards in reverse consumption order, dead cards last. Prefixes are therefore ordered by importance, not by position in a list.
- A replay only reveals the cards the game actually showed — roughly three quarters of a deck on average. Unrevealed cards are absent from the sequence rather than padded.
- The player's memory is per game **and** per seat. Anything driving both sides of a game must hold two policy objects, or the two hands interleave into one history.
- Self-play replays carry an explicit decklist field, so the trajectory converter never has to infer the deck.
- Tournaments swap seats every game; a win rate against a single opponent is not a total order over decks, and non-transitive matchups do occur.
- Long runs resume. Work directories are keyed by cycle, and a stage is only marked done once its output has been verified.

## Contribution
- Designed and implemented the deck-builder transformer and its reverse-read training formulation
- Designed and implemented the two-level trajectory player, including the per-game memory over decision summaries
- Built the streaming data pipeline: download, conversion, staged checkpoints, verification and resume
- Built the self-play systems for both models, including deck-pool sampling and temperature mixing
- Built the distributed two-machine training mode and its transfer protocol
- Built the evolutionary deck search that selects the deck the player actually wins with
- Built the evaluation stack: head-to-head tournaments, seat swapping and statistical testing
- Ran all experiments and led the experimental direction

## License
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Copyright and Ownership
Copyright © 2026 SIZHONG ZHANG.

The framework provided in this repository is licensed under Apache 2.0. Ownership of any agent trained using this framework, including its learned parameters, weights, and derived artifacts, belongs to the user who trained it. Users retain full rights to agents they train and are free to use, distribute, or commercialize them without restriction.

### Third-party components
The game engine in `cg/` and the card database `EN_Card_Data.csv` are **not my
intellectual property** and are not covered by this project's licence. They
are redistributed here only so the training code can be run. All rights
remain with their respective owners.

- Competition and card data: <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview>
- Engine documentation: <https://matsuoinstitute.github.io/cabt/index.html>

Anyone using this repository should consult those pages for the terms that
apply to the engine and the card data.


