"""Evolutionary deck search: sample -> tournament -> eliminate -> refill.

One generation:
  1. the pool holds POOL_PER_TEMP decks at each temperature (default
     12 x 0.7, 12 x 0.8, 12 x 0.9 = 36 decks)
  2. every deck plays --games-per-deck games, piloted on BOTH sides by
     the same player checkpoint, so only the decks differ
  3. the top --survivors decks stay, the rest are eliminated
  4. the pool is refilled so each temperature is back to POOL_PER_TEMP:
     if 4/2/3 survivors are t0.7/t0.8/t0.9, the new decks are 8/10/9

Every deck carries its own seed, so any deck in the manifest can be
rebuilt exactly:
    python3 build_deck.py --ckpt <builder> --n 1 --temperature <t> --seed <seed>

manifest.csv columns:
    generation, deck_uid, temperature, seed, born_gen, games, wins,
    losses, draws, win_rate, rank, survived

Usage (on the server, inside tmux):
    python3 deck_evolution.py --builder builder_tf_deck.pt \
        --player player_v2_r2.pt --engine ../submission_SD \
        --generations 10 --games-per-deck 1000 --workers 126
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

# torch emits two harmless UserWarnings per worker (nested-tensor fallback
# and the float-vs-bool mask deprecation). At 120 workers that is hundreds
# of lines. Silencing here covers the children too, since spawn re-imports
# this module in each of them.
import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import multiprocessing as mp
import pathlib
import queue
import random
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent

POOL_PER_TEMP = 12
TEMPERATURES = (0.7, 0.8, 0.9)
MAX_SELECTIONS_PER_GAME = 2000


# ---------------------------------------------------------------------------
# deck sampling (builder side)
# ---------------------------------------------------------------------------
def sample_one_deck(model, vocab, vocab_size, temperature, seed, device):
    """Reproducible single-deck sample: same seed -> same 60 cards."""
    import build_deck as bd
    rng = np.random.default_rng(seed)
    return bd.sample_deck(model, vocab, vocab_size, temperature, device, rng)


# ---------------------------------------------------------------------------
# tournament worker: plays deck-vs-deck games with ONE shared pilot
# ---------------------------------------------------------------------------
def worker_loop(wid, engine_dir, tf_dir, player_ckpt, decks, jobs, out_q):
    # thread limits are set at module import time (see top of file) — by the
    # time this runs, numpy is already loaded in the child.
    sys.path.insert(0, str(tf_dir))
    sys.path.insert(0, str(engine_dir))
    import torch
    torch.set_num_threads(1)
    from cg.game import battle_start, battle_select, battle_finish
    from player_vocab import PlayerVocab, tokenize_decision
    from player_model import PlayerDecoder

    ck = torch.load(player_ckpt, map_location="cpu")
    model = PlayerDecoder(ck["vocab_size"], max_len=ck["max_len"])
    model.load_state_dict(ck["model"])
    model.eval()
    pv = PlayerVocab()
    rng = random.Random(wid * 7919 + 13)

    def pick(obs):
        cur, sel = obs["current"], obs["select"]
        n = len(sel.get("option") or [])
        min_c = sel.get("minCount") or 0
        max_c = sel.get("maxCount") or 0
        if n == 0:
            return []
        if n == 1 and min_c >= 1:
            return [0]
        out = tokenize_decision(pv, cur, sel)
        if out is None:
            k = min(max(min_c, 1), max(max_c, 1), n)
            return rng.sample(range(n), k)
        toks, opt_pos = out
        with torch.no_grad():
            t = torch.tensor([toks], dtype=torch.int64)
            pad = torch.zeros_like(t, dtype=torch.bool)
            p = torch.tensor([opt_pos], dtype=torch.int64)
            om = torch.ones_like(p, dtype=torch.bool)
            scores = model(t, pad, p, om)[0].tolist()
        dec = n if min_c == 0 else None
        real = sorted(range(n), key=lambda i: scores[i], reverse=True)
        if dec is not None and scores[dec] > scores[real[0]]:
            return []
        picked = real[:max(min_c, 1)]
        for i in real[max(min_c, 1):max_c]:
            if dec is not None and scores[i] <= scores[dec]:
                break
            picked.append(i)
        return sorted(picked)

    for (ia, ib, a_seat) in jobs:
        da, db = decks[ia], decks[ib]
        seat = (da, db) if a_seat == 0 else (db, da)
        try:
            obs, _ = battle_start(seat[0], seat[1])
            steps = 0
            while obs["current"]["result"] == -1 and steps < MAX_SELECTIONS_PER_GAME:
                obs = battle_select(pick(obs))
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


def build_schedule(n_decks, games_per_deck, rng):
    """Random perfect matchings: every deck gets exactly games_per_deck
    games, opponents spread ~uniformly, seats swapped in each pair."""
    jobs = []
    rounds = max(1, games_per_deck // 2)
    idx = list(range(n_decks))
    for _ in range(rounds):
        rng.shuffle(idx)
        for k in range(0, n_decks - 1, 2):
            a, b = idx[k], idx[k + 1]
            jobs.append((a, b, 0))
            jobs.append((a, b, 1))
    rng.shuffle(jobs)
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="evolutionary deck search")
    ap.add_argument("--builder", required=True, help="builder checkpoint .pt")
    ap.add_argument("--player", required=True, help="player checkpoint .pt")
    ap.add_argument("--engine", required=True,
                    help="folder containing the cg/ package")
    ap.add_argument("--cards", default=None, help="EN_Card_Data.csv")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--games-per-deck", type=int, default=1000)
    ap.add_argument("--survivors", type=int, default=9)
    ap.add_argument("--pool-per-temp", type=int, default=POOL_PER_TEMP)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260803,
                    help="master seed: everything below derives from it")
    ap.add_argument("--out", default=str(HERE / "evolution"))
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "manifest.csv"
    decks_dir = outdir / "decks"
    decks_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(HERE))
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

    n_workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    master = np.random.default_rng(args.seed)
    per_temp = args.pool_per_temp
    n_decks = per_temp * len(TEMPERATURES)

    print(f"evolution: {args.generations} generations | pool {n_decks} "
          f"({per_temp} per T in {TEMPERATURES}) | {args.games_per_deck} "
          f"games/deck | survivors {args.survivors} | workers {n_workers}",
          flush=True)
    print(f"builder {pathlib.Path(args.builder).name} | "
          f"player {pathlib.Path(args.player).name} | master seed {args.seed}",
          flush=True)

    if not manifest.exists():
        with open(manifest, "w", newline="") as f:
            csv.writer(f).writerow(
                ["generation", "deck_uid", "temperature", "seed", "born_gen",
                 "games", "wins", "losses", "draws", "win_rate", "rank",
                 "survived"])

    # pool entries: dict(uid, temp, seed, cards, born)
    pool = []
    uid_counter = 0

    def new_deck(temp, born_gen):
        global uid_counter
        seed = int(master.integers(0, 2**31 - 1))
        cards = sample_one_deck(builder, vocab, vocab_size, temp, seed, device)
        uid = f"g{born_gen}_{uid_counter:04d}"
        uid_counter += 1
        path = decks_dir / f"{uid}_t{int(temp*10)}.csv"
        path.write_text("\n".join(str(c) for c in cards) + "\n")
        return {"uid": uid, "temp": temp, "seed": seed,
                "cards": cards, "born": born_gen}

    t_start = time.time()
    for gen in range(1, args.generations + 1):
        # ---- refill so every temperature is back to per_temp ----
        alive = {t: sum(1 for d in pool if d["temp"] == t)
                 for t in TEMPERATURES}
        need = {t: per_temp - alive[t] for t in TEMPERATURES}
        t0 = time.time()
        for t in TEMPERATURES:
            for _ in range(need[t]):
                pool.append(new_deck(t, gen))
        print(f"\n=== generation {gen} === survivors by T: "
              f"{{{', '.join(f'{t}:{alive[t]}' for t in TEMPERATURES)}}} "
              f"-> sampled {{{', '.join(f'{t}:{need[t]}' for t in TEMPERATURES)}}} "
              f"in {time.time()-t0:.0f}s", flush=True)

        decks = [d["cards"] for d in pool]
        sched_rng = random.Random(int(master.integers(0, 2**31 - 1)))
        jobs = build_schedule(len(pool), args.games_per_deck, sched_rng)
        chunks = [jobs[i::n_workers] for i in range(n_workers)]

        out_q = mp.Queue()
        procs = []
        for w, ch in enumerate(chunks):
            if not ch:
                continue
            p = mp.Process(target=worker_loop,
                           args=(w, str(pathlib.Path(args.engine).resolve()),
                                 str(HERE), args.player, decks, ch, out_q))
            p.start()
            procs.append(p)

        wins = [0] * len(pool)
        losses = [0] * len(pool)
        draws = [0] * len(pool)
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
        keep = set(order[:args.survivors])

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
                p = decks_dir / f"{d['uid']}_t{int(d['temp']*10)}.csv"
                if p.exists():
                    p.unlink()
                    removed += 1

        print(f"  tournament done in {(time.time()-t0)/60:.1f} min "
              f"({total} games) | deleted {removed} eliminated decks",
              flush=True)
        for rank, i in enumerate(order[:args.survivors], 1):
            d = pool[i]
            print(f"   #{rank} {d['uid']} T{d['temp']} wr {wr[i]:.3f} "
                  f"({wins[i]}W/{losses[i]}L) born g{d['born']}", flush=True)

        pool = [pool[i] for i in order[:args.survivors]]

    print(f"\nfinished {args.generations} generations in "
          f"{(time.time()-t_start)/60:.0f} min")
    print(f"manifest: {manifest}")
    print(f"surviving decks: {decks_dir}")
