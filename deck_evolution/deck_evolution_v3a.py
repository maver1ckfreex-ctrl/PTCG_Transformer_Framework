"""Evolutionary deck search piloted by a v3a (sequence_v3) player.

Same search as deck_evolution/deck_evolution.py -- sample -> tournament ->
eliminate -> refill, decks ranked by win rate, losers deleted, seeds kept in
the manifest so any deck can be rebuilt. What changed is the PILOT.

The old script instantiated PlayerDecoder directly and scored every decision
from scratch. A v3a checkpoint cannot be driven that way: it is two-level
(per-decision encoder + causal memory over decision summaries) and it wants
the decklist as a prefix, so it needs per-game state. This uses the Policy
wrapper from seq_trial/eval/tournament_seq.py, which already carries that
memory and exposes reset(deck) + select(obs), and it holds TWO policies --
one per seat -- because a single memory chain shared across both sides would
interleave two games and corrupt both.

At temperature 0 Policy.select reproduces the old pick() decision rule
exactly: same decline-option handling, same minCount/maxCount fill, same
random fallback when tokenization fails. The selection mechanism is
unchanged; temperature is the only thing layered on top.

One cycle:
  1. the pool holds NUM_DECK decks, split as evenly as possible across the
     --deck_ratio temperatures (100 over 0.7/0.8/0.9 -> 34/33/33)
  2. every deck plays --games-per-deck games, piloted on BOTH sides by the
     same checkpoint, so only the decks differ
  3. the top --survivors decks stay, the rest are eliminated and deleted
  4. the pool is refilled back to each temperature's target

With --player_temp_mix=1 the pilot temperature is drawn 1:1:1 (or whatever
the length of --player_ratio implies) and assigned PER PAIR, so both seats
of a seat-swapped pair play at the same temperature and every deck sees each
temperature equally often. With --player_temp_mix=0 (default) it is 0.0
throughout, i.e. pure greedy argmax, matching the old script.

    python3 deck_evolution_v3a.py \
        --builder ../deck_evolution/builder_tf_deck.pt \
        --player  ../seq_trial/best_v3a.pt \
        --engine  ../submission_r2_t07_torch \
        --NUM_DECK 100 --NUM_CYCLE 8 --games-per-deck 1000 \
        --player_temp_mix 1 --workers 126
"""

import os

# MUST come before numpy/torch are imported, and must be at module level:
# with spawn, every child re-imports THIS module, so a per-worker setting
# lands too late and each child opens 64 BLAS threads -> pthread_create
# failures ("thread explosion") at high worker counts.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import argparse
import collections
import csv
import multiprocessing as mp
import pathlib
import queue
import random
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

DEFAULT_TF_DIR = HERE.parent / "deck_evolution"     # builder side lives here
DEFAULT_SEQ_DIR = HERE.parent / "seq_trial"         # v3a Policy lives here
MAX_SELECTIONS_PER_GAME = 2000


def parse_temps(spec, what):
    try:
        vals = [float(x) for x in str(spec).split(",") if x.strip() != ""]
    except ValueError:
        sys.exit(f"ERROR: {what} must be comma-separated numbers, got {spec!r}")
    if not vals:
        sys.exit(f"ERROR: {what} is empty")
    return tuple(vals)


def split_counts(total, k):
    """total split across k buckets as evenly as possible, front-loaded."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


# ---------------------------------------------------------------------------
# deck sampling (builder side)
# ---------------------------------------------------------------------------
def read_deck_csv(path):
    return [int(x) for x in pathlib.Path(path).read_text().split()]


def check_prefix_legal(vocab, prefix):
    """The prefix has to obey the same rules the sampler enforces, or
    legality_mask would eventually mask every token and the sampler would
    fail deep inside a worker."""
    problems = []
    counts = collections.Counter(vocab.name[c] for c in prefix)
    for cid in set(prefix):
        if not vocab.is_basic_energy[cid] and counts[vocab.name[cid]] > 4:
            problems.append(f">4 x {vocab.name[cid]}")
    if sum(vocab.is_ace_spec[c] for c in prefix) > 1:
        problems.append(">1 ACE SPEC")
    unknown = [c for c in prefix if c not in vocab.name]
    if unknown:
        problems.append(f"unknown card ids {sorted(set(unknown))[:5]}")
    return sorted(set(problems))


def sample_deck_from(model, vocab, vocab_size, temperature, device, rng,
                     prefix):
    """build_deck.sample_deck, but seeded with `prefix` already in the deck.

    The builder is a decoder over [BOS, BUILD, card, card, ...], so
    continuing a partial deck is just starting the loop with those cards
    already appended. legality_mask is recomputed from `deck` every step,
    so the 4-copy cap and the single ACE SPEC count the prefix too and the
    completion cannot be illegal.

    With an empty prefix this is byte-identical to bd.sample_deck -- the
    same rng draws in the same order -- but the no-prefix path calls
    bd.sample_deck directly anyway, so from-scratch behaviour is not
    merely equivalent, it is the original function.
    """
    import torch
    import build_deck as bd
    from card_vocab import BOS, BUILD, PAD, PLAY, CARD_OFFSET

    deck = list(prefix)
    tokens = [BOS, BUILD] + [vocab.token(c) for c in deck]
    with torch.no_grad():
        while len(deck) < 60:
            t = torch.tensor([tokens[-model.max_len:]], dtype=torch.int64,
                             device=device)
            logits = model(t)[0, -1].float().cpu().numpy()
            logits[[PAD, BOS, BUILD, PLAY]] = -np.inf
            logits[~bd.legality_mask(vocab, deck, vocab_size)] = -np.inf
            logits = logits / max(temperature, 1e-4)
            logits -= logits.max()
            pr = np.exp(logits)
            pr /= pr.sum()
            tok = int(rng.choice(vocab_size, p=pr))
            deck.append(tok - CARD_OFFSET)
            if len(tokens) < model.max_len:
                tokens.append(tok)
    return deck


def sample_one_deck(model, vocab, vocab_size, temperature, seed, device,
                    prefix=()):
    """Reproducible single-deck sample: same seed -> same 60 cards."""
    import build_deck as bd
    rng = np.random.default_rng(seed)
    if not prefix:
        return bd.sample_deck(model, vocab, vocab_size, temperature, device,
                              rng)
    return sample_deck_from(model, vocab, vocab_size, temperature, device,
                            rng, prefix)


# ---------------------------------------------------------------------------
# tournament worker: deck-vs-deck, one v3a checkpoint on both seats
# ---------------------------------------------------------------------------
def worker_loop(wid, engine_dir, seq_dir, player_ckpt, decks, jobs, out_q):
    sys.path.insert(0, str(pathlib.Path(seq_dir) / "eval"))
    sys.path.insert(0, str(engine_dir))
    import torch
    torch.set_num_threads(1)
    from cg.game import battle_start, battle_select, battle_finish
    from tournament_seq import Policy

    # one policy per SEAT: a trajectory model's memory is per game and is
    # conditioned on ITS OWN deck, so the seats cannot share a chain.
    pols = [Policy(player_ckpt, 0.0, seed=0x5EED + wid * 2 + s)
            for s in (0, 1)]
    # the two seats differ only in their memory chain -- share the weights
    # rather than holding a second 23 MB copy per worker.
    if (pols[0].arch == pols[1].arch == "sequence_v3"
            and pols[1].runner is not None):
        pols[1].model = pols[0].model
        pols[1].runner = type(pols[1].runner)(pols[0].model)

    for (ia, ib, a_seat, temp) in jobs:
        da, db = decks[ia], decks[ib]
        seat = (da, db) if a_seat == 0 else (db, da)
        for s in (0, 1):
            pols[s].temp = float(temp)
        try:
            pols[0].reset(seat[0])
            pols[1].reset(seat[1])
            obs, _ = battle_start(seat[0], seat[1])
            steps = 0
            while (obs["current"]["result"] == -1
                   and steps < MAX_SELECTIONS_PER_GAME):
                mover = obs["current"]["yourIndex"]
                obs = battle_select(pols[mover].select(obs))
                steps += 1
            res = obs["current"]["result"]
            battle_finish()
        except Exception:
            try:
                battle_finish()
            except Exception:
                pass
            out_q.put((ia, ib, "error"))
            continue
        if res == 2 or res not in (0, 1):
            out_q.put((ia, ib, "draw"))
        else:
            out_q.put((ia, ib, "A" if res == a_seat else "B"))


def build_schedule(n_decks, games_per_deck, rng, play_temps):
    """Random perfect matchings: every deck gets exactly games_per_deck
    games, opponents spread ~uniformly, seats swapped in each pair.

    The pilot temperature is assigned PER ROUND and cycles through
    play_temps. Every deck plays exactly one pair per round, so cycling by
    round -- rather than by pair -- gives each deck the same number of games
    at each temperature, not just the same number pool-wide. Both seats of a
    seat-swap always share a temperature."""
    jobs = []
    rounds = max(1, games_per_deck // 2)
    idx = list(range(n_decks))
    for r in range(rounds):
        rng.shuffle(idx)
        t = play_temps[r % len(play_temps)]
        for k in range(0, n_decks - 1, 2):
            a, b = idx[k], idx[k + 1]
            jobs.append((a, b, 0, t))
            jobs.append((a, b, 1, t))
    rng.shuffle(jobs)
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="evolutionary deck search, v3a pilot")
    ap.add_argument("--builder", required=True, help="builder checkpoint .pt")
    ap.add_argument("--player", required=True,
                    help="v3a player checkpoint .pt (arch sequence_v3)")
    ap.add_argument("--engine", required=True,
                    help="folder containing the cg/ package")
    ap.add_argument("--NUM_DECK", type=int, default=100,
                    help="pool size per cycle (default 100)")
    ap.add_argument("--NUM_CYCLE", type=int, default=8,
                    help="number of cycles (default 8)")
    ap.add_argument("--deck_ratio", default="0.7,0.8,0.9",
                    help="builder temperatures, pool split evenly across them")
    ap.add_argument("--player_temp_mix", type=int, default=0, choices=(0, 1),
                    help="0 = pilot always at 0.0 (default), 1 = mix")
    ap.add_argument("--player_ratio", default="0.0,0.1,0.2",
                    help="pilot temperatures, equal share each; "
                         "only used when --player_temp_mix=1")
    ap.add_argument("--base_deck", default=None,
                    help="csv of an existing deck; every sampled deck keeps "
                         "its first --start_at-1 cards and the builder "
                         "generates the rest. Needs --start_at.")
    ap.add_argument("--start_at", type=int, default=None,
                    help="1..60 -- the position the builder starts building "
                         "at, so cards 1..start_at-1 are kept from "
                         "--base_deck and that card itself is NOT kept. "
                         "Needs --base_deck.")
    ap.add_argument("--cards", default=None, help="EN_Card_Data.csv")
    ap.add_argument("--games-per-deck", type=int, default=1000)
    ap.add_argument("--survivors", type=int, default=None,
                    help="default NUM_DECK//4, the old 9-of-36 fraction")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260803,
                    help="master seed: everything below derives from it")
    ap.add_argument("--tf-dir", default=str(DEFAULT_TF_DIR),
                    help="folder holding build_deck.py / builder_model.py")
    ap.add_argument("--seq-dir", default=str(DEFAULT_SEQ_DIR),
                    help="seq_trial root, for eval/tournament_seq.py")
    ap.add_argument("--out", default=str(HERE / "evolution_v3a"))
    args = ap.parse_args()

    DECK_TEMPS = parse_temps(args.deck_ratio, "--deck_ratio")
    if args.player_temp_mix == 1:
        PLAY_TEMPS = parse_temps(args.player_ratio, "--player_ratio")
    else:
        PLAY_TEMPS = (0.0,)

    # --base_deck and --start_at only do anything together; either one on
    # its own is almost certainly a mistake, so say so instead of silently
    # building from scratch.
    if (args.base_deck is None) != (args.start_at is None):
        missing = "--start_at" if args.start_at is None else "--base_deck"
        sys.exit(f"ERROR: --base_deck and --start_at go together; {missing} "
                 f"is missing. Give both, or neither to build from scratch.")
    if args.start_at is not None and not (1 <= args.start_at <= 60):
        sys.exit(f"ERROR: --start_at must be between 1 and 60, got "
                 f"{args.start_at}")

    if args.NUM_DECK < 2:
        sys.exit("ERROR: --NUM_DECK must be at least 2")
    survivors = args.survivors if args.survivors else max(1, args.NUM_DECK // 4)
    if survivors >= args.NUM_DECK:
        sys.exit(f"ERROR: --survivors {survivors} must be < "
                 f"--NUM_DECK {args.NUM_DECK}")

    tf_dir = pathlib.Path(args.tf_dir).resolve()
    seq_dir = pathlib.Path(args.seq_dir).resolve()
    for p, what in ((tf_dir / "build_deck.py", "--tf-dir"),
                    (seq_dir / "eval" / "tournament_seq.py", "--seq-dir")):
        if not p.exists():
            sys.exit(f"ERROR: {what} looks wrong, no {p}")

    mp.set_start_method("spawn", force=True)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "manifest.csv"
    decks_dir = outdir / "decks"
    decks_dir.mkdir(exist_ok=True)

    # builder side only: keep this path OUT of the workers, whose
    # tournament_seq imports a same-named card_vocab from seq_trial/common.
    sys.path.insert(0, str(tf_dir))
    sys.path.insert(0, str(pathlib.Path(args.engine).resolve()))
    import torch
    import build_deck as bd
    from builder_model import DeckDecoder
    from card_vocab import Vocab

    device = torch.device("cpu")
    vocab = Vocab(args.cards) if args.cards else Vocab()
    ck = torch.load(args.builder, map_location="cpu")
    builder = DeckDecoder(ck["vocab_size"], max_len=ck["max_len"]).to(device)
    builder.load_state_dict(ck["model"])
    builder.eval()
    vocab_size = ck["vocab_size"]

    # ---- base deck prefix ------------------------------------------------
    PREFIX = ()
    if args.base_deck is not None:
        base = read_deck_csv(args.base_deck)
        if len(base) < args.start_at - 1:
            sys.exit(f"ERROR: --base_deck has {len(base)} cards but "
                     f"--start_at {args.start_at} needs at least "
                     f"{args.start_at - 1}")
        PREFIX = tuple(base[:args.start_at - 1])
        problems = check_prefix_legal(vocab, PREFIX)
        if problems:
            sys.exit(f"ERROR: the kept prefix is already illegal: "
                     f"{', '.join(problems)}")
        # the seed alone no longer reproduces a deck, so keep the prefix
        # beside the manifest
        (outdir / "base_prefix.csv").write_text(
            "\n".join(str(c) for c in PREFIX) + "\n" if PREFIX else "")

    # fail before spending an hour sampling decks if the pilot is not v3a
    pck = torch.load(args.player, map_location="cpu", weights_only=False)
    parch = pck.get("arch", "<none>")
    if parch != "sequence_v3":
        print(f"WARNING: --player arch is {parch!r}, not 'sequence_v3'. "
              f"Policy will still try to drive it.", flush=True)
    del pck

    n_workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    master = np.random.default_rng(args.seed)
    targets = dict(zip(DECK_TEMPS,
                       split_counts(args.NUM_DECK, len(DECK_TEMPS))))

    print(f"evolution v3a: {args.NUM_CYCLE} cycles | pool {args.NUM_DECK} "
          f"({', '.join(f'{t}:{targets[t]}' for t in DECK_TEMPS)}) | "
          f"{args.games_per_deck} games/deck | survivors {survivors} | "
          f"workers {n_workers}", flush=True)
    print(f"builder {pathlib.Path(args.builder).name} | "
          f"player {pathlib.Path(args.player).name} (arch {parch}) | "
          f"master seed {args.seed}", flush=True)
    if PREFIX:
        print(f"base deck: {pathlib.Path(args.base_deck).name} | "
              f"keeping cards 1..{args.start_at - 1} ({len(PREFIX)} cards), "
              f"builder generates {60 - len(PREFIX)} from position "
              f"{args.start_at}", flush=True)
    else:
        print("base deck: none -- building every deck from scratch",
              flush=True)
    print(f"pilot temperature: "
          f"{'mix ' + ':'.join('1' for _ in PLAY_TEMPS) + ' over ' + str(PLAY_TEMPS) if args.player_temp_mix else 'fixed 0.0 (greedy)'}",
          flush=True)

    if not manifest.exists():
        with open(manifest, "w", newline="") as f:
            csv.writer(f).writerow(
                ["generation", "deck_uid", "temperature", "seed", "born_gen",
                 "games", "wins", "losses", "draws", "win_rate", "rank",
                 "survived"])

    pool = []
    uid_counter = 0

    def new_deck(temp, born_gen):
        global uid_counter
        seed = int(master.integers(0, 2**31 - 1))
        cards = sample_one_deck(builder, vocab, vocab_size, temp, seed,
                                device, PREFIX)
        uid = f"g{born_gen}_{uid_counter:04d}"
        uid_counter += 1
        path = decks_dir / f"{uid}_t{int(temp*100)}.csv"
        path.write_text("\n".join(str(c) for c in cards) + "\n")
        return {"uid": uid, "temp": temp, "seed": seed,
                "cards": cards, "born": born_gen}

    t_start = time.time()
    for gen in range(1, args.NUM_CYCLE + 1):
        # ---- refill so every temperature is back to its target ----
        alive = {t: sum(1 for d in pool if d["temp"] == t) for t in DECK_TEMPS}
        need = {t: max(0, targets[t] - alive[t]) for t in DECK_TEMPS}
        t0 = time.time()
        for t in DECK_TEMPS:
            for _ in range(need[t]):
                pool.append(new_deck(t, gen))
        print(f"\n=== cycle {gen} === survivors by T: "
              f"{{{', '.join(f'{t}:{alive[t]}' for t in DECK_TEMPS)}}} "
              f"-> sampled {{{', '.join(f'{t}:{need[t]}' for t in DECK_TEMPS)}}} "
              f"in {time.time()-t0:.0f}s", flush=True)

        decks = [d["cards"] for d in pool]
        sched_rng = random.Random(int(master.integers(0, 2**31 - 1)))
        jobs = build_schedule(len(pool), args.games_per_deck, sched_rng,
                              PLAY_TEMPS)
        chunks = [jobs[i::n_workers] for i in range(n_workers)]

        out_q = mp.Queue()
        procs = []
        for w, ch in enumerate(chunks):
            if not ch:
                continue
            p = mp.Process(target=worker_loop,
                           args=(w, str(pathlib.Path(args.engine).resolve()),
                                 str(seq_dir), str(pathlib.Path(
                                     args.player).resolve()),
                                 decks, ch, out_q))
            p.start()
            procs.append(p)

        wins = [0] * len(pool)
        losses = [0] * len(pool)
        draws = [0] * len(pool)
        errors = 0
        done = 0
        total = len(jobs)
        t0 = time.time()
        while done < total:
            try:
                ia, ib, r = out_q.get(timeout=3600)
            except queue.Empty:
                print("  !! tournament timed out", flush=True)
                break
            done += 1
            if r == "A":
                wins[ia] += 1; losses[ib] += 1
            elif r == "B":
                wins[ib] += 1; losses[ia] += 1
            elif r == "draw":
                draws[ia] += 1; draws[ib] += 1
            else:
                errors += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"  {done}/{total} games | {done/el:.0f} g/s | "
                      f"eta {(total-done)/max(done/el,1e-9)/60:.1f} min",
                      flush=True)
        for p in procs:
            p.terminate()

        played = [wins[i] + losses[i] + draws[i] for i in range(len(pool))]
        wr = [(wins[i] + 0.5 * draws[i]) / played[i] if played[i] else 0.0
              for i in range(len(pool))]
        order = sorted(range(len(pool)), key=lambda i: -wr[i])
        keep = set(order[:survivors])

        with open(manifest, "a", newline="") as f:
            w = csv.writer(f)
            for rank, i in enumerate(order, 1):
                d = pool[i]
                w.writerow([gen, d["uid"], d["temp"], d["seed"], d["born"],
                            played[i], wins[i], losses[i], draws[i],
                            f"{wr[i]:.4f}", rank, int(i in keep)])

        # eliminated decks are DELETED from disk; the manifest keeps their
        # seed, which is all that is needed to rebuild them.
        removed = 0
        for i, d in enumerate(pool):
            if i not in keep:
                p = decks_dir / f"{d['uid']}_t{int(d['temp']*100)}.csv"
                if p.exists():
                    p.unlink()
                    removed += 1

        print(f"  tournament done in {(time.time()-t0)/60:.1f} min "
              f"({total} games, {errors} errored) | deleted {removed} "
              f"eliminated decks", flush=True)
        for rank, i in enumerate(order[:survivors], 1):
            d = pool[i]
            print(f"   #{rank} {d['uid']} T{d['temp']} wr {wr[i]:.3f} "
                  f"({wins[i]}W/{losses[i]}L) born g{d['born']}", flush=True)

        pool = [pool[i] for i in order[:survivors]]

    print(f"\nfinished {args.NUM_CYCLE} cycles in "
          f"{(time.time()-t_start)/60:.0f} min")
    print(f"manifest: {manifest}")
    print(f"surviving decks: {decks_dir}")
