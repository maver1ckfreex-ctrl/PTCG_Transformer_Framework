"""Self-produced-data training loop: the models generate their own data.

Same training process as the Kaggle-data runs. The ONLY difference is
where the replays come from: the engine plays them, using the current
builder's decks and the current player as pilot.

One cycle:

  PHASE 1 - builder
    sample --p1-decks decks from the builder (1:1:1 across T=0.7/0.8/0.9)
    play --p1-games games with the current player  -> Kaggle-format replays
    replay_to_dataset.py  -> dataset.npz   (reverse-order deck sequences)
    train_builder.py --init <current builder>  -> new builder
    delete the raw replays

  PHASE 2 - player
    sample --p2-decks decks from the NEW builder
    play --p2-games games                    -> Kaggle-format replays
    replay_to_decisions.py -> decisions.npz
    train_player.py --init <current player>  -> new player
    delete the raw replays

  repeat --cycles times, each cycle continuing from the previous models.

Usage:
    python3 selfplay_loop.py --engine ../submission_SD \
        --builder builder_tf_deck.pt --player player_r2.pt \
        --cycles 5 --workers 120
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import argparse
import multiprocessing as mp
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
TEMPERATURES = (0.7, 0.8, 0.9, 1.0, 1.1)   # equal share; --temps overrides


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _unused_sample_chunk(job):
    """Worker: build a slice of the deck batch. ~6 decks/s per core, so
    this MUST be parallel — 200k decks single-threaded is ~10 hours."""
    builder_ckpt, cards, out_dir, specs = job
    import torch
    torch.set_num_threads(1)
    import build_deck as bd
    from builder_model import DeckDecoder
    from card_vocab import Vocab

    vocab = Vocab(cards) if cards else Vocab()
    ck = torch.load(builder_ckpt, map_location="cpu")
    model = DeckDecoder(ck["vocab_size"], max_len=ck["max_len"])
    model.load_state_dict(ck["model"])
    model.eval()
    device = torch.device("cpu")
    out_dir = pathlib.Path(out_dir)
    n = 0
    for (idx, t, s) in specs:
        cards_ = bd.sample_deck(model, vocab, ck["vocab_size"], t, device,
                               np.random.default_rng(s))
        (out_dir / f"d{idx:07d}_t{int(t*10)}_s{s}.csv").write_text(
            "\n".join(str(c) for c in cards_) + "\n")
        n += 1
    return n


def sample_decks(builder_ckpt, cards, out_dir, n_decks, seed, workers,
                 temps=None, reuse=False, batch=4096):
    """Sample n_decks with the BATCHED GPU sampler (build_deck.py).

    Decodes `batch` decks in lockstep with the legality mask as device
    tensors — ~800 decks/s on a GPU versus ~190/s spreading single-deck
    CPU sampling over 120 workers. Writes one (n, 60) int32 .npy, which
    selfplay_replay.generate() reads directly.
    """
    temps = temps or TEMPERATURES
    out_dir = pathlib.Path(out_dir)
    pool_npy = out_dir.with_suffix(".npy")
    if reuse and pool_npy.exists():
        have = len(np.load(pool_npy))
        if have >= n_decks:
            log(f"    reusing {have} decks in {pool_npy.name}")
            return pool_npy
        log(f"    only {have}/{n_decks} decks present — resampling")
    pool_npy.parent.mkdir(parents=True, exist_ok=True)
    mix = ":".join(str(t) for t in temps)          # equal split across temps
    t0 = time.time()
    run([sys.executable, str(HERE / "build_deck.py"),
         "--ckpt", str(builder_ckpt), "--n", str(n_decks),
         "--temperature", mix, "--batch", str(batch),
         "--out-npy", str(pool_npy)]
        + (["--cards", cards] if cards else []))
    made = len(np.load(pool_npy))
    log(f"    {made} decks in {(time.time()-t0)/60:.1f} min "
        f"({made/max(time.time()-t0,1e-9):.0f}/s) -> {pool_npy.name}")
    return pool_npy


def run(cmd):
    log("    $ " + " ".join(str(c) for c in cmd[1:4]) + " ...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(str(c) for c in cmd)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser(description="self-produced-data training")
    ap.add_argument("--engine", default=str(HERE),
                    help="folder containing cg/ (default: bundled)")
    ap.add_argument("--builder", default=str(HERE / "builder_tf_deck.pt"))
    ap.add_argument("--player", default=str(HERE / "player_r2.pt"))
    ap.add_argument("--cards", default=None)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--workers", type=int, default=None)
    # phase 1 (builder)
    ap.add_argument("--p1-games", type=int, default=120_000)
    ap.add_argument("--p1-decks", type=int, default=12_000)
    ap.add_argument("--p1-epochs", type=int, default=30)
    ap.add_argument("--p1-batch", type=int, default=512)
    # phase 2 (player)
    ap.add_argument("--p2-games", type=int, default=100_000)
    ap.add_argument("--p2-decks", type=int, default=200_000)
    ap.add_argument("--p2-epochs", type=int, default=1)
    ap.add_argument("--p2-batch", type=int, default=512)
    ap.add_argument("--keep-replays", action="store_true")
    ap.add_argument("--deck-batch", type=int, default=4096,
                    help="decks decoded in parallel on the GPU")
    ap.add_argument("--reuse-decks", action="store_true",
                    help="reuse decks already in the cycle folder instead of "
                         "resampling (saves ~18 min per phase on a restart)")
    ap.add_argument("--work", default=str(HERE / "selfplay_work"))
    ap.add_argument("--phases", choices=["both", "builder", "player"],
                    default="both",
                    help="which phases to run each cycle. 'player' skips "
                         "builder training and uses --builder as-is")
    ap.add_argument("--play-temps", default="0",
                    help="player move temperatures, one drawn per game "
                         "(0 = argmax). e.g. 0,0.3,0.6,1.0")
    ap.add_argument("--temps", default=None,
                    help="comma-separated builder temperatures, equal share "
                         "(default 0.7,0.8,0.9,1.0,1.1)")
    ap.add_argument("--kaggle-data", default=None,
                    help="kaggle decisions.npz; when given, phase 2 trains on "
                         "a MIX of kaggle + self-play (see --mix)")
    ap.add_argument("--mix", type=float, default=0.25,
                    help="self-play fraction of each batch (default 0.25)")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    engine = pathlib.Path(args.engine).resolve()
    nw = args.workers or max(1, (os.cpu_count() or 2) - 2)
    cards_args = ["--cards", args.cards] if args.cards else []

    builder = str(pathlib.Path(args.builder).resolve())
    player = str(pathlib.Path(args.player).resolve())
    master = np.random.default_rng(args.seed)
    temps = (tuple(float(x) for x in args.temps.split(","))
             if args.temps else TEMPERATURES)
    ptemps = tuple(float(x) for x in args.play_temps.split(","))

    log(f"self-play training | {args.cycles} cycles | workers {nw}")
    log(f"  base builder: {pathlib.Path(builder).name}")
    log(f"  base player : {pathlib.Path(player).name}")
    log(f"  phase1: {args.p1_games:,} games / {args.p1_decks:,} decks")
    log(f"  phase2: {args.p2_games:,} games / {args.p2_decks:,} decks")
    log(f"  deck temperatures: {temps} (equal share)")
    log(f"  player temperatures: {ptemps} (one per game)")
    if args.kaggle_data:
        log(f"  phase2 player data: MIX {1-args.mix:.0%} kaggle / "
            f"{args.mix:.0%} self-play")

    import selfplay_replay as spr

    t_start = time.time()
    for cyc in range(1, args.cycles + 1):
        cdir = work / f"cycle{cyc}"
        cdir.mkdir(exist_ok=True)

        # ---------------- PHASE 1: builder ----------------
        if args.phases in ("both", "builder"):
            log(f"=== cycle {cyc} PHASE 1 (builder) ===")
            d1 = cdir / "decks_p1"
            pool1 = sample_decks(builder, args.cards, d1, args.p1_decks,
                                 int(master.integers(0, 2**31 - 1)), nw,
                                 temps, reuse=args.reuse_decks,
                                 batch=args.deck_batch)
            r1 = cdir / "replays_p1"
            spr.generate([pool1], player, engine, HERE, r1,
                         args.p1_games, nw,
                         int(master.integers(0, 2**31 - 1)),
                         play_temps=ptemps)
            ds = cdir / "dataset.npz"
            run([sys.executable, str(HERE / "replay_to_dataset.py"),
                 "--replays", str(r1), "--out", str(ds),
                 "--workers", str(nw)] + cards_args)
            new_builder = str(cdir / f"builder_c{cyc}.pt")
            run([sys.executable, str(HERE / "train_builder.py"),
                 "--data", str(ds), "--init", builder,
                 "--epochs", str(args.p1_epochs),
                 "--batch", str(args.p1_batch), "--out", new_builder])
            if not args.keep_replays:
                shutil.rmtree(r1, ignore_errors=True)
                log("    deleted phase-1 raw replays")
            builder = new_builder
        else:
            log(f"=== cycle {cyc} PHASE 1 skipped "
                f"(using {pathlib.Path(builder).name} as-is) ===")

        # ---------------- PHASE 2: player ----------------
        if args.phases in ("both", "player"):
            log(f"=== cycle {cyc} PHASE 2 (player) ===")
            d2 = cdir / "decks_p2"
            pool2 = sample_decks(builder, args.cards, d2, args.p2_decks,
                                 int(master.integers(0, 2**31 - 1)), nw,
                                 temps, reuse=args.reuse_decks,
                                 batch=args.deck_batch)
            r2 = cdir / "replays_p2"
            spr.generate([pool2], player, engine, HERE, r2,
                         args.p2_games, nw,
                         int(master.integers(0, 2**31 - 1)),
                         play_temps=ptemps)
            dec = cdir / "decisions.npz"
            run([sys.executable, str(HERE / "replay_to_decisions.py"),
                 "--replays", str(r2), "--out", str(dec),
                 "--workers", str(nw)] + cards_args)
            new_player = str(cdir / f"player_c{cyc}.pt")
            if args.kaggle_data:
                run([sys.executable, str(HERE / "train_player_mixed.py"),
                     "--kaggle-data", args.kaggle_data,
                     "--selfplay-data", str(dec), "--init", player,
                     "--mix", str(args.mix),
                     "--epochs", str(args.p2_epochs),
                     "--batch", str(args.p2_batch), "--out", new_player])
            else:
                run([sys.executable, str(HERE / "train_player.py"),
                     "--data", str(dec), "--init", player,
                     "--epochs", str(args.p2_epochs),
                     "--batch", str(args.p2_batch), "--out", new_player])
            if not args.keep_replays:
                shutil.rmtree(r2, ignore_errors=True)
                log("    deleted phase-2 raw replays")
            player = new_player
        else:
            log(f"=== cycle {cyc} PHASE 2 skipped ===")

        log(f"=== cycle {cyc} done ({(time.time()-t_start)/60:.0f} min) | "
            f"builder {pathlib.Path(builder).name} | "
            f"player {pathlib.Path(player).name} ===")

    log(f"finished {args.cycles} cycles in {(time.time()-t_start)/60:.0f} min")
    log(f"final builder: {builder}")
    log(f"final player : {player}")
