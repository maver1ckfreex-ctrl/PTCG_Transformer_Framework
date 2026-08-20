"""SEQUENCE design: convert replays into per-game TRAJECTORIES.

One sample per (replay, agent) = one full trajectory:

    DECK -> obs_1 -> act_1 -> obs_2 -> act_2 -> ... -> obs_N -> act_N -> win/lose

Decision order inside a trajectory is preserved, and the acting agent's
60-card decklist is stored with it. Nothing is flattened across games.

The per-decision tokenization is `player_vocab.tokenize_decision`, byte for
byte the same call the baseline converter makes, and the same skip rules
apply (forced moves, oversized selects). The ONLY difference between this
file and baseline/replay_to_decisions.py is that decisions keep their game
grouping and their order. That is deliberate: the tournament must compare
two designs, not two datasets.

Usage:
    python3 replay_to_trajectories.py --replays <dir> --out traj.npz
    python3 replay_to_trajectories.py --replays <dir> --out traj.npz --limit 500
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

import numpy as np

from card_vocab import CARD_OFFSET
from player_vocab import PlayerVocab, tokenize_decision, MAX_OPTIONS

DECK_SIZE = 60
MAX_DECISIONS = 256      # observed max ~130 per agent; guard only

_PV = None
_WINNERS_ONLY = False


def _init_worker(cards_csv, winners_only=False):
    global _PV, _WINNERS_ONLY
    _PV = PlayerVocab(cards_csv) if cards_csv else PlayerVocab()
    _WINNERS_ONLY = winners_only


def process_file(path_str):
    """One replay -> (filename, [trajectory, ...], usable).

    trajectory = (deck_tokens, [(toks, opt_pos, chosen), ...], reward)

    KAGGLE EPISODE FORMAT (same as the baseline converter):
      steps[i][agent]["action"] is the action that CAUSED the transition
      into step i, so the pair is
          state  = steps[i][agent].observation   (status ACTIVE)
          action = steps[i+1][agent].action
      The 60-card decklist each agent submitted is steps[1][agent]["action"].
    """
    path = pathlib.Path(path_str)
    try:
        ep = json.loads(path.read_text())
        rw = ep.get("rewards") if isinstance(ep, dict) else None
        if not rw or len(rw) != 2 or set(rw) != {1, -1}:
            return path.name, [], False
        steps = ep.get("steps") or []
        if len(steps) < 2:
            return path.name, [], False

        # ---- the deck each agent brought (front of the trajectory) ----
        # Kaggle replays: the 60-card list is the action at step 1.
        # Self-play replays (selfplay_replay.py) start recording AFTER the
        # decks are handed to battle_start, so they carry an explicit
        # "decks" field instead. Accept either.
        decks = {}
        explicit = ep.get("decks")
        if isinstance(explicit, list) and len(explicit) == 2:
            for ai in (0, 1):
                d = explicit[ai]
                if isinstance(d, list) and len(d) == DECK_SIZE:
                    decks[ai] = [int(c) + CARD_OFFSET for c in d]
        if not decks:
            for ai in (0, 1):
                act = steps[1][ai].get("action") if ai < len(steps[1]) else None
                if isinstance(act, list) and len(act) == DECK_SIZE:
                    decks[ai] = [int(c) + CARD_OFFSET for c in act]

        per_agent = {0: [], 1: []}
        for i in range(len(steps) - 1):
            for ai, entry in enumerate(steps[i]):
                if entry.get("status") != "ACTIVE":
                    continue
                obs = entry.get("observation") or {}
                cur, sel = obs.get("current"), obs.get("select")
                if not cur or not sel:
                    continue
                if ai >= len(steps[i + 1]):
                    continue
                action = steps[i + 1][ai].get("action")
                if not isinstance(action, list):
                    continue
                n_opt = len(sel.get("option") or [])
                if n_opt > MAX_OPTIONS:
                    continue
                chosen = [a for a in action
                          if isinstance(a, int) and 0 <= a < n_opt]
                can_decline = (sel.get("minCount") or 0) == 0
                if not chosen:
                    if not can_decline:
                        continue
                    chosen = [n_opt]
                out = tokenize_decision(_PV, cur, sel)
                if out is None:
                    continue
                toks, opt_pos = out
                per_agent[ai].append((toks, opt_pos, chosen))

        trajs = []
        for ai in (0, 1):
            if _WINNERS_ONLY and rw[ai] < 0:
                continue                 # V3a: drop the loser's side entirely
            dec = per_agent[ai][:MAX_DECISIONS]
            if not dec or ai not in decks:
                continue
            trajs.append((decks[ai], dec, float(rw[ai])))
        return path.name, trajs, True
    except Exception:
        return path.name, [], False


def default_workers():
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:      # macOS
        cores = os.cpu_count() or 1
    return max(1, cores - 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replays -> trajectories")
    parser.add_argument("--replays", required=True, nargs="+",
                        help="one or more directories of replay json; all "
                             "are globbed together (kaggle + self-play)")
    parser.add_argument("--cards", default=None)
    parser.add_argument("--out", default="trajectories.npz")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--winners-only", action="store_true",
                        help="V3a: keep ONLY the winning side of each replay. "
                             "The loser's trajectory is dropped at conversion "
                             "time, so training never sees it at all.")
    args = parser.parse_args()

    files = []
    for d in args.replays:
        files.extend(pathlib.Path(d).rglob("*.json"))
    files = sorted({f for f in files if not f.name.startswith("_")})
    if args.limit:
        files = files[:args.limit]

    workers = args.workers or default_workers()
    vocab_size = (PlayerVocab(args.cards) if args.cards
                  else PlayerVocab()).vocab_size
    if len(args.replays) > 1:
        for d in args.replays:
            n = len(list(pathlib.Path(d).rglob("*.json")))
            print(f"  source: {n:>7} json in {d}", flush=True)
    print(f"scanning {len(files)} json files | vocab {vocab_size} "
          f"| workers {workers}"
          + ("  | WINNERS ONLY (V3a)" if args.winners_only else ""),
          flush=True)

    tok_flat, tok_off = [], [0]
    pos_flat, pos_off = [], [0]
    chosen_flat, chosen_off = [], [0]
    deck_flat, deck_off = [], [0]
    traj_off = [0]              # index into the DECISION arrays
    rewards, sources = [], []
    used = skipped = 0

    if files:
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(args.cards, args.winners_only)) as pool:
            for i, (name, trajs, usable) in enumerate(
                    pool.imap_unordered(process_file,
                                        (str(f) for f in files),
                                        chunksize=8), 1):
                sources.append(name)
                used, skipped = (used + 1, skipped) if usable else (used, skipped + 1)
                for deck, decisions, reward in trajs:
                    deck_flat.extend(deck)
                    deck_off.append(len(deck_flat))
                    for toks, opt_pos, chosen in decisions:
                        tok_flat.extend(toks)
                        tok_off.append(len(tok_flat))
                        pos_flat.extend(opt_pos)
                        pos_off.append(len(pos_flat))
                        chosen_flat.extend(chosen)
                        chosen_off.append(len(chosen_flat))
                    traj_off.append(len(tok_off) - 1)
                    rewards.append(reward)
                if i % 500 == 0:
                    print(f"  {i}/{len(files)} files | {len(rewards)} trajectories"
                          f" | {len(tok_off) - 1} decisions", flush=True)

    if not rewards:
        raise SystemExit("no usable trajectories found")

    np.savez_compressed(
        args.out,
        vocab_size=np.int64(vocab_size),
        tok_flat=np.asarray(tok_flat, dtype=np.int16),
        tok_off=np.asarray(tok_off, dtype=np.int64),
        pos_flat=np.asarray(pos_flat, dtype=np.int16),
        pos_off=np.asarray(pos_off, dtype=np.int64),
        chosen_flat=np.asarray(chosen_flat, dtype=np.int16),
        chosen_off=np.asarray(chosen_off, dtype=np.int64),
        deck_flat=np.asarray(deck_flat, dtype=np.int16),
        deck_off=np.asarray(deck_off, dtype=np.int64),
        traj_off=np.asarray(traj_off, dtype=np.int64),
        reward=np.asarray(rewards, dtype=np.float32),
        sources=np.asarray(sorted(set(sources)), dtype="U40"),
    )

    n_tr, n_dec = len(rewards), len(tok_off) - 1
    lens = np.diff(np.asarray(traj_off))
    print(f"done: {used} replays used, {skipped} skipped -> "
          f"{n_tr} trajectories / {n_dec} decisions "
          f"({int((np.asarray(rewards) > 0).sum())} winning trajectories)")
    print(f"decisions per trajectory: mean {lens.mean():.1f} "
          f"median {int(np.median(lens))} max {int(lens.max())}")
    print(f"avg tokens per decision {len(tok_flat) / n_dec:.0f} | saved {args.out}")
