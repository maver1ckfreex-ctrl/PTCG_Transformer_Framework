"""Head-to-head with a DIFFERENT deck on each side.

tournament_seq.py deliberately plays one deck on both sides, so the pilot is
the only variable. This variant lets each side bring its own 60 cards, for
the opposite question: pilot+deck as a package, e.g. an evolved deck driven
by v3a against the R2 submission on R2's stock deck.

Seats still swap every game, so first-player advantage cancels. Everything
else -- the Policy wrapper, per-game memory reset, the z-test -- is imported
from tournament_seq.py unchanged.

    python3 eval/tournament_decks.py \
        --engine ../submission_r2_t07_torch \
        --a ../submission_r2_t07_torch/player.pt \
        --deck-a ../submission_r2_t07_torch/deck.csv \
        --b best_v3a.pt --deck-b evolved.csv \
        --games 2000
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import math
import multiprocessing as mp
import pathlib
import queue
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tournament_seq import Policy, read_deck, _paths, MAX_SELECTIONS_PER_GAME


def worker_loop(worker_id, deck_a, deck_b, n_games, result_queue, engine_dir,
                ckpt_a, ckpt_b, temp_a, temp_b):
    _paths()
    sys.path.insert(0, str(engine_dir))
    from cg.game import battle_start, battle_select, battle_finish

    pol_a = Policy(ckpt_a, temp_a, seed=0xA0000 + worker_id)
    pol_b = Policy(ckpt_b, temp_b, seed=0xB0000 + worker_id)

    for g in range(n_games):
        a_seat = (worker_id + g) % 2
        # each pilot is reset with ITS OWN deck, and that deck is dealt to
        # whichever seat the pilot is sitting in this game
        pol_a.reset(deck_a)
        pol_b.reset(deck_b)
        seats = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)
        try:
            obs, _ = battle_start(seats[0], seats[1])
            if obs is None:
                result_queue.put(("error", 0))
                continue
            steps = 0
            while (obs["current"]["result"] == -1
                   and steps < MAX_SELECTIONS_PER_GAME):
                mover = obs["current"]["yourIndex"]
                pol = pol_a if mover == a_seat else pol_b
                obs = battle_select(pol.select(obs))
                steps += 1
            result = obs["current"]["result"]
            battle_finish()
        except Exception:
            try:
                battle_finish()
            except Exception:
                pass
            result_queue.put(("error", 0))
            continue

        if result in (0, 1):
            result_queue.put(("A" if result == a_seat else "B", steps))
        else:
            result_queue.put(("draw", steps))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser(
        description="head-to-head, a different deck on each side")
    ap.add_argument("--engine", required=True,
                    help="directory containing the cg/ engine package")
    ap.add_argument("--a", required=True, help="side A checkpoint")
    ap.add_argument("--b", required=True, help="side B checkpoint")
    ap.add_argument("--deck-a", required=True, help="60-card csv for side A")
    ap.add_argument("--deck-b", required=True, help="60-card csv for side B")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--temp-a", type=float, default=0.0)
    ap.add_argument("--temp-b", type=float, default=0.0)
    args = ap.parse_args()

    engine = pathlib.Path(args.engine).resolve()
    if not (engine / "cg").exists():
        sys.exit(f"ERROR: no cg/ package in {engine}")
    deck_a = read_deck(args.deck_a)
    deck_b = read_deck(args.deck_b)
    workers_n = args.workers or max(1, (os.cpu_count() or 1) - 2)

    same = "SAME deck" if deck_a == deck_b else "different decks"
    print(f"=== {args.games} games | {same} | seats swap ===")
    print(f"  A = {pathlib.Path(args.a).name}  "
          f"on {pathlib.Path(args.deck_a).name}")
    print(f"  B = {pathlib.Path(args.b).name}  "
          f"on {pathlib.Path(args.deck_b).name}")
    print(f"  workers {workers_n}\n")

    result_queue = mp.Queue()
    per = [args.games // workers_n] * workers_n
    for i in range(args.games % workers_n):
        per[i] += 1

    procs = []
    for i, n in enumerate(per):
        if n == 0:
            continue
        p = mp.Process(target=worker_loop,
                       args=(i, deck_a, deck_b, n, result_queue, str(engine),
                             str(pathlib.Path(args.a).resolve()),
                             str(pathlib.Path(args.b).resolve()),
                             args.temp_a, args.temp_b))
        p.start()
        procs.append(p)

    a_wins = b_wins = draws = errors = finished = total_steps = 0
    t0 = time.time()
    while finished < args.games:
        try:
            kind, steps = result_queue.get(timeout=1800)
        except queue.Empty:
            print("  !! timed out; aborting")
            break
        finished += 1
        total_steps += steps
        if kind == "A":
            a_wins += 1
        elif kind == "B":
            b_wins += 1
        elif kind == "draw":
            draws += 1
        else:
            errors += 1
        if finished % 50 == 0 or finished == args.games:
            print(f"  {finished}/{args.games} | A {a_wins}  B {b_wins}  "
                  f"draws {draws} | {time.time()-t0:.0f}s", end="\r")

    for p in procs:
        p.terminate()
    for p in procs:
        p.join()

    played = a_wins + b_wins + draws
    print("\n\n=== RESULT ===")
    print(f"  A {pathlib.Path(args.a).name} on "
          f"{pathlib.Path(args.deck_a).name}: {a_wins} "
          f"({100.0*a_wins/played if played else 0:.1f}%)")
    print(f"  B {pathlib.Path(args.b).name} on "
          f"{pathlib.Path(args.deck_b).name}: {b_wins} "
          f"({100.0*b_wins/played if played else 0:.1f}%)")
    print(f"  draws {draws} | errors {errors} | "
          f"avg len {total_steps/max(finished,1):.0f} | {time.time()-t0:.0f}s")

    decided = a_wins + b_wins
    if decided:
        z = (a_wins - b_wins) / math.sqrt(decided)
        verdict = ("SIGNIFICANT (p<0.05)" if abs(z) >= 1.96
                   else "NOT significant")
        print(f"  z = {z:+.2f} over {decided} decided games -> {verdict}")
    print(f"  --> stronger side: "
          f"{'A' if a_wins > b_wins else 'B' if b_wins > a_wins else 'TIE'}")
