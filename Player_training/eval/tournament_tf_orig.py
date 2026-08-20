"""
tournament_tf.py — TRANSFORMER (builder deck + player pilot) vs submission_s.

Side A = the V2 transformer system:
          deck  built by the deck builder (build_deck.py output csv)
          pilot player_v2.pt scoring the engine's options directly
Side B = the previous best submission (its own MLP + its own deck),
          loaded from the submission folder untouched.

Seats swap every game so first-player advantage cancels. Ends with a
two-proportion z-test on the result.

Prepare side A's deck first:
    cd deck_builder_tf_v2 && python3 build_deck.py --ckpt builder_v2.pt --n 1

Usage:
    python3 tournament_tf.py                  # 1000 games
    python3 tournament_tf.py --games 2000 --workers 8
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import math
import pathlib
import queue
import random
import sys
import time
import multiprocessing as mp

import numpy as np

ROOT = pathlib.Path(__file__).parent
TF_DIR = ROOT / "deck_builder_tf_v2"

# ============================================================
#  FILL THESE IN  (leave "ROOT /" and put your path after it)
# ============================================================
PLAYER_CKPT = TF_DIR / "player_v2s.pt"         # trained player decoder
DECK_A      = TF_DIR / "deck_tf30_1.csv"       # deck built by the builder

OPP_DIR     = ROOT / "submission_SD"           # baseline shallow RL
# ============================================================

OPP_WEIGHTS = OPP_DIR / "model_weights.npz"
OPP_DECK    = OPP_DIR / "deck.csv"
MAX_SELECTIONS_PER_GAME = 2000


def read_deck(path):
    with open(path) as f:
        rows = f.read().split()
    return [int(rows[i]) for i in range(60)]


class NumpyPolicy:
    """Greedy inference with an exported MLP npz (same math as main.py)."""

    def __init__(self, npz_path):
        w = np.load(npz_path)
        self.layers = []
        i = 1
        while f"w{i}" in w:
            self.layers.append((w[f"w{i}"].astype(np.float32),
                                w[f"b{i}"].astype(np.float32)))
            i += 1

    def scores(self, candidates):
        x = np.asarray(candidates, dtype=np.float32)
        for k, (wm, bm) in enumerate(self.layers):
            x = x @ wm.T + bm
            if k < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return (1.0 / (1.0 + np.exp(-x)))[:, 0].tolist()


class TransformerPolicy:
    """player_v2 decoder scoring the engine's options directly."""

    def __init__(self, ckpt_path, temp=0.0):
        self.temp = float(temp)
        import torch
        from player_model import PlayerDecoder
        from player_vocab import PlayerVocab
        torch.set_num_threads(1)
        self.torch = torch
        ck = torch.load(ckpt_path, map_location="cpu")
        self.model = PlayerDecoder(ck["vocab_size"], max_len=ck["max_len"])
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.pv = PlayerVocab()
        self.rng = random.Random(0xC0FFEE)

    def select(self, obs):
        """Score every move the engine offers — including 'take nothing',
        which is legal whenever minCount == 0 and is scored as an extra
        DECLINE span at index n."""
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
        if out is None:      # oversized/unusual selection: random legal pick
            k = min(max(min_c, 1), max(max_c, 1), n)
            return self.rng.sample(range(n), k)
        toks, opt_pos = out
        t = self.torch
        with t.no_grad():
            tokens = t.tensor([toks], dtype=t.int64)
            pad = t.zeros_like(tokens, dtype=t.bool)
            pos = t.tensor([opt_pos], dtype=t.int64)
            omask = t.ones_like(pos, dtype=t.bool)
            scores = self.model(tokens, pad, pos, omask)[0].tolist()

        decline_idx = n if min_c == 0 else None
        if self.temp > 0:
            # Gumbel-top-k == sampling without replacement from
            # softmax(scores/T). int() is required: the engine rejects
            # np.int64 in the selection list.
            z = np.asarray(scores, dtype=np.float64) / max(self.temp, 1e-6)
            g = np.random.default_rng(self.rng.randrange(1 << 30)).gumbel(
                size=z.shape)
            order = [int(i) for i in np.argsort(-(z + g))]
        else:
            order = sorted(range(len(scores)), key=lambda i: scores[i],
                           reverse=True)
        real = [i for i in order if i < n]
        if decline_idx is not None and order[0] == decline_idx:
            return []                          # the model says: take nothing
        picked = real[:max(min_c, 1)]
        for i in real[max(min_c, 1):max_c]:    # keep adding while it wants to
            if decline_idx is not None and scores[i] <= scores[decline_idx]:
                break
            picked.append(i)
        return sorted(picked)


# --- WORKER: side A (transformer) vs side B (submission OR transformer) --
def worker_loop(worker_id, deck_a, deck_b, n_games, result_queue,
                opp_dir=None, player_ckpt=None, opp_player=None,
                temp_a=0.0, temp_b=0.0):
    opp_dir = pathlib.Path(opp_dir) if opp_dir else OPP_DIR
    sys.path.insert(0, str(TF_DIR))            # player model + vocab
    sys.path.insert(0, str(opp_dir))           # the submission's cg + features
    from cg.game import battle_start, battle_select, battle_finish

    pol_a = TransformerPolicy(str(player_ckpt or PLAYER_CKPT), temp_a)

    if opp_player:                      # transformer vs transformer
        pol_b = TransformerPolicy(str(opp_player), temp_b)
        b_select = pol_b.select
    else:                               # transformer vs an MLP submission
        from features import build_candidate_inputs, choose_by_scores
        pol_b = NumpyPolicy(str(opp_dir / "model_weights.npz"))

        def b_select(obs):
            candidates, _ = build_candidate_inputs(obs)
            return choose_by_scores(pol_b.scores(candidates), obs["select"])

    for g in range(n_games):
        a_seat = (worker_id + g) % 2
        seat_deck = (deck_a, deck_b) if a_seat == 0 else (deck_b, deck_a)

        try:
            obs, sd = battle_start(seat_deck[0], seat_deck[1])
            if obs is None:
                result_queue.put(("error", 0))
                continue
            steps = 0
            while obs["current"]["result"] == -1 and steps < MAX_SELECTIONS_PER_GAME:
                mover = obs["current"]["yourIndex"]
                if mover == a_seat:
                    obs = battle_select(pol_a.select(obs))
                else:
                    obs = battle_select(b_select(obs))
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
    parser = argparse.ArgumentParser(
        description="transformer (deck+player) vs submission_s")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--deck", default=None,
                        help="override side A deck csv")
    parser.add_argument("--opp", default=None,
                        help="override side B submission folder")
    parser.add_argument("--player", default=None,
                        help="override side A player checkpoint")
    parser.add_argument("--opp-player", default=None,
                        help="side B is ANOTHER transformer checkpoint "
                             "(both sides then play --deck)")
    parser.add_argument("--play-temp", type=float, default=0.0,
                        help="side A move temperature (0 = argmax)")
    parser.add_argument("--opp-play-temp", type=float, default=0.0,
                        help="side B move temperature (0 = argmax)")
    parser.add_argument("--opp-deck", default=None,
                        help="side B's deck in --opp-player mode "
                             "(default: same deck as side A)")
    args = parser.parse_args()
    if args.deck:
        DECK_A = pathlib.Path(args.deck)
    if args.player:
        PLAYER_CKPT = pathlib.Path(args.player)
    if args.opp:
        OPP_DIR = pathlib.Path(args.opp)
        OPP_WEIGHTS = OPP_DIR / "model_weights.npz"
        OPP_DECK = OPP_DIR / "deck.csv"

    for label, p in {"PLAYER_CKPT": PLAYER_CKPT, "DECK_A": DECK_A,
                     "OPP_WEIGHTS": OPP_WEIGHTS, "OPP_DECK": OPP_DECK}.items():
        if not pathlib.Path(p).exists():
            sys.exit(f"ERROR: {label} not found: '{p}'")

    deck_a = read_deck(DECK_A)
    # in transformer-vs-transformer mode both sides play the SAME deck
    if args.opp_player:
        # transformer vs transformer: side B plays --opp-deck if given,
        # otherwise the same deck as side A (pure pilot comparison)
        deck_b_path = pathlib.Path(args.opp_deck) if args.opp_deck else DECK_A
    else:
        deck_b_path = OPP_DECK
    deck_b = read_deck(deck_b_path)
    num_workers = args.workers or max(1, (os.cpu_count() or 1) - 2)

    print(f"=== TRANSFORMER tournament | {args.games} games ===")
    print(f"  A = player_v2 ({PLAYER_CKPT.name}) + {pathlib.Path(DECK_A).name}")
    if args.opp_player:
        print(f"  B = player {pathlib.Path(args.opp_player).name} + "
              f"{deck_b_path.name}"
              + ("  (SAME deck both sides)"
                 if deck_b_path == DECK_A else "  (different deck)"))
    else:
        print(f"  B = {OPP_WEIGHTS.name} + {OPP_DECK.name}  (from {OPP_DIR.name}/)")
    print(f"  (seats swapped every game)\n")

    result_queue = mp.Queue()
    per = [args.games // num_workers] * num_workers
    for i in range(args.games % num_workers):
        per[i] += 1

    workers = []
    for i, n in enumerate(per):
        if n == 0:
            continue
        p = mp.Process(target=worker_loop,
                       args=(i, deck_a, deck_b, n, result_queue,
                             str(OPP_DIR), str(PLAYER_CKPT),
                             args.opp_player, args.play_temp,
                             args.opp_play_temp))
        p.start()
        workers.append(p)

    a_wins = b_wins = draws = errors = finished = 0
    total_steps = 0
    t0 = time.time()
    while finished < args.games:
        try:
            kind, steps = result_queue.get(timeout=1800)
        except queue.Empty:
            print("  !! timed out; aborting"); break
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
            print(f"  {finished}/{args.games} | A: {a_wins}  B: {b_wins}  "
                  f"draws: {draws} | {time.time()-t0:.0f}s", end="\r")

    for p in workers:
        p.terminate()
    for p in workers:
        p.join()

    played = a_wins + b_wins + draws
    wr_a = 100.0 * a_wins / played if played else 0.0
    wr_b = 100.0 * b_wins / played if played else 0.0
    print("\n\n=== RESULT ===")
    print(f"  A (transformer + {pathlib.Path(DECK_A).name}) : "
          f"{a_wins}  ({wr_a:.1f}%)")
    b_label = (f"player {pathlib.Path(args.opp_player).name} + "
               f"{deck_b_path.name}" if args.opp_player
               else f"{OPP_DIR.name}")
    print(f"  B ({b_label}) : {b_wins}  ({wr_b:.1f}%)")
    print(f"  draws {draws} | errors {errors} | "
          f"avg len {total_steps/max(finished,1):.0f} | {time.time()-t0:.0f}s")

    # two-proportion z-test on decided games (A vs B, null = 50/50)
    decided = a_wins + b_wins
    if decided:
        z = (a_wins - b_wins) / math.sqrt(decided)
        verdict = ("SIGNIFICANT (p<0.05)" if abs(z) >= 1.96
                   else "NOT significant — more games or same strength")
        print(f"  z = {z:+.2f} over {decided} decided games -> {verdict}")
    winner = "A" if a_wins > b_wins else "B" if b_wins > a_wins else "TIE"
    print(f"  --> stronger side: {winner}")
