"""Head-to-head: BASELINE arm vs SEQUENCE arm. Same deck both sides.

Both pilots play the identical 60-card deck and seats swap every game, so
the only thing under test is the pilot. Ends with a two-proportion z-test.

The sequence pilot carries trajectory memory, so its policy is RESET at the
start of every game and fed that game's decklist. The baseline pilot is
stateless, exactly as it is today.

Usage:
    python3 tournament_seq.py \
        --engine ../../submission_r2_t07_torch \
        --deck   ../../submission_r2_t07_torch/deck.csv \
        --a ../baseline/baseline.pt \
        --b ../sequence/sequence.pt \
        --games 1000
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
import random
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_SELECTIONS_PER_GAME = 2000


def read_deck(path):
    with open(path) as f:
        rows = f.read().split()
    return [int(rows[i]) for i in range(60)]


def _paths():
    sys.path.insert(0, str(ROOT / "common"))
    sys.path.insert(0, str(ROOT / "baseline"))
    sys.path.insert(0, str(ROOT / "sequence"))


class Policy:
    """Wraps either arm behind one interface: reset(deck) + select(obs)."""

    def __init__(self, ckpt_path, temp=0.0, seed=0xC0FFEE):
        _paths()
        import torch
        from player_vocab import PlayerVocab
        torch.set_num_threads(1)
        self.torch = torch
        self.temp = float(temp)
        self.rng = random.Random(seed)

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # an R2-era checkpoint has no "arch" tag and uses the old
        # PlayerDecoder parameter names -> remap it onto the baseline arm
        # so R2 itself can be one side of the tournament.
        is_r2 = "arch" not in ck and "embed.weight" in ck["model"]
        self.arch = "r2" if is_r2 else ck.get("arch", "baseline")

        if self.arch == "sequence_v3b":
            from seq_model_v3b import SeqPlayerV3B, SeqPlayerV3BRunner, WIN
            self.model = SeqPlayerV3B(ck["vocab_size"], max_len=ck["max_len"],
                                      mem_layers=ck.get("mem_layers", 2),
                                      max_traj=ck.get("max_traj", 256))
            self.model.load_state_dict(ck["model"])
            self.model.eval()
            # play is always conditioned on WIN
            self.runner = SeqPlayerV3BRunner(self.model, condition=WIN)
        elif self.arch == "sequence_v3":
            from seq_model_v3 import SeqPlayerV3, SeqPlayerV3Runner
            self.model = SeqPlayerV3(ck["vocab_size"], max_len=ck["max_len"],
                                     mem_layers=ck.get("mem_layers", 2),
                                     max_traj=ck.get("max_traj", 256))
            self.model.load_state_dict(ck["model"])
            self.model.eval()
            self.runner = SeqPlayerV3Runner(self.model)
        elif self.arch == "sequence":
            from seq_model import SeqPlayer, SeqPlayerRunner
            self.model = SeqPlayer(ck["vocab_size"], max_len=ck["max_len"],
                                   mem_layers=ck.get("mem_layers", 2),
                                   max_traj=ck.get("max_traj", 256))
            self.model.load_state_dict(ck["model"])
            self.model.eval()
            self.runner = SeqPlayerRunner(self.model)
        else:
            from baseline_model import BaselinePlayer
            self.model = BaselinePlayer(ck["vocab_size"],
                                        max_len=ck["max_len"])
            if is_r2:
                sys.path.insert(0, str(ROOT / "tools"))
                from warm_start import remap_r2
                _, unexp = self.model.load_state_dict(
                    remap_r2(ck["model"]), strict=False)
                if unexp:
                    raise RuntimeError(f"bad R2 remap: {unexp}")
            else:
                self.model.load_state_dict(ck["model"])
            self.model.eval()
            self.runner = None
        self.pv = PlayerVocab()

    def reset(self, deck_card_ids):
        if self.runner is not None:
            from card_vocab import CARD_OFFSET
            self.runner.reset([int(c) + CARD_OFFSET for c in deck_card_ids])

    def _scores(self, toks, opt_pos):
        if self.runner is not None:
            return self.runner.scores(toks, opt_pos)
        t = self.torch
        with t.no_grad():
            tokens = t.tensor([list(toks)], dtype=t.int64)
            pos = t.tensor([list(opt_pos)], dtype=t.int64)
            omask = t.ones_like(pos, dtype=t.bool)
            return self.model(tokens, pos, omask)[0].tolist()

    def select(self, obs):
        from player_vocab import tokenize_decision
        cur, sel = obs["current"], obs["select"]
        opts = sel.get("option") or []
        n = len(opts)
        min_c = sel.get("minCount") or 0
        max_c = sel.get("maxCount") or 0
        if n == 0:
            return []
        if n == 1 and min_c >= 1:
            return [0]

        out = tokenize_decision(self.pv, cur, sel)
        if out is None:
            k = min(max(min_c, 1), max(max_c, 1), n)
            return self.rng.sample(range(n), k)
        toks, opt_pos = out
        scores = self._scores(toks, opt_pos)

        decline_idx = n if min_c == 0 else None
        if self.temp > 0:
            z = np.asarray(scores, dtype=np.float64) / max(self.temp, 1e-6)
            g = np.random.default_rng(self.rng.randrange(1 << 30)).gumbel(
                size=z.shape)
            order = [int(i) for i in np.argsort(-(z + g))]
        else:
            order = sorted(range(len(scores)), key=lambda i: scores[i],
                           reverse=True)
        real = [i for i in order if i < n]
        if decline_idx is not None and order[0] == decline_idx:
            return []
        picked = real[:max(min_c, 1)]
        for i in real[max(min_c, 1):max_c]:
            if decline_idx is not None and scores[i] <= scores[decline_idx]:
                break
            picked.append(i)
        return sorted(picked)


def worker_loop(worker_id, deck, n_games, result_queue, engine_dir,
                ckpt_a, ckpt_b, temp_a, temp_b):
    _paths()
    sys.path.insert(0, str(engine_dir))
    from cg.game import battle_start, battle_select, battle_finish

    pol_a = Policy(ckpt_a, temp_a, seed=0xA0000 + worker_id)
    pol_b = Policy(ckpt_b, temp_b, seed=0xB0000 + worker_id)

    for g in range(n_games):
        a_seat = (worker_id + g) % 2
        pol_a.reset(deck)
        pol_b.reset(deck)
        try:
            obs, _ = battle_start(deck, deck)
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
    ap = argparse.ArgumentParser(description="baseline arm vs sequence arm")
    ap.add_argument("--engine", required=True,
                    help="directory containing the cg/ engine package")
    ap.add_argument("--deck", required=True,
                    help="60-card csv played by BOTH sides")
    ap.add_argument("--a", required=True, help="side A checkpoint")
    ap.add_argument("--b", required=True, help="side B checkpoint")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--temp-a", type=float, default=0.0)
    ap.add_argument("--temp-b", type=float, default=0.0)
    args = ap.parse_args()

    engine = pathlib.Path(args.engine).resolve()
    if not (engine / "cg").exists():
        sys.exit(f"ERROR: no cg/ package in {engine}")
    deck = read_deck(args.deck)
    workers_n = args.workers or max(1, (os.cpu_count() or 1) - 2)

    print(f"=== {args.games} games | same deck both sides | seats swap ===")
    print(f"  A = {pathlib.Path(args.a).name}")
    print(f"  B = {pathlib.Path(args.b).name}")
    print(f"  deck = {pathlib.Path(args.deck).name} | workers {workers_n}\n")

    result_queue = mp.Queue()
    per = [args.games // workers_n] * workers_n
    for i in range(args.games % workers_n):
        per[i] += 1

    procs = []
    for i, n in enumerate(per):
        if n == 0:
            continue
        p = mp.Process(target=worker_loop,
                       args=(i, deck, n, result_queue, str(engine),
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
    print(f"  A {pathlib.Path(args.a).name}: {a_wins} "
          f"({100.0*a_wins/played if played else 0:.1f}%)")
    print(f"  B {pathlib.Path(args.b).name}: {b_wins} "
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
