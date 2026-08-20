"""Convert Kaggle replays into the deck-builder training dataset.

For EVERY side of every replay (winner AND loser — no wasted data):

  reward       +1 if that side won, -1 if it lost (draws skipped).
                The reward is stored per sequence and applied by the
                trainer AFTER the whole deck is read — never leaked
                into the tokens.

  build seq    the deck read FROM THE END OF THE GAME BACKWARD:
                 1. board at game end (active stack, bench stacks,
                    own stadium)  — the win condition made visible
                 2. discarded cards, most recently consumed first
                 3. cards dead in hand / revealed own prizes, last
                [BOS, BUILD, cards...] padded to BUILD_LEN.

  play seq     the game read forward: cards in the order they were
                first put into action (left hand -> play/discard).
                [BOS, PLAY, cards...] padded to PLAY_LEN. (Phase 2.)

MULTI-CORE: json parsing is spread over (available cores - 2) worker
processes (BLAS threads pinned to 1 per process — required, >2 threads
per process crashes on the many-core server). The 2 spare cores are
left for the pipeline's download/unzip/delete work.

Usage:
    python3 replay_to_dataset.py --replays <dir> --out dataset.npz
    # incremental folders (resume-safe; already-converted files skipped):
    python3 replay_to_dataset.py --replays folder_0 --out dataset.npz
    python3 replay_to_dataset.py --replays folder_1 --out dataset.npz --append
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

import numpy as np

from card_vocab import Vocab, BOS, BUILD, PLAY

BUILD_LEN = 64    # BOS + BUILD + up to 60 revealed cards
PLAY_LEN = 128


# ---------------------------------------------------------------------------
# replay walking
# ---------------------------------------------------------------------------
def iter_observations(episode):
    """Yield (step_idx, current) for every observation in step order."""
    for step_idx, step in enumerate(episode.get("steps") or []):
        for entry in step:
            cur = (entry.get("observation") or {}).get("current")
            if cur and cur.get("players"):
                yield step_idx, cur


def stack_cards(pk):
    """All card dicts in one in-play pokemon stack."""
    out = [pk]
    for sub in ("energyCards", "tools", "preEvolution"):
        out.extend(c for c in (pk.get(sub) or []) if c)
    return out


def side_zone_cards(cur, side):
    """Yield (serial, card_id, zone) for the side's visible cards."""
    players = cur.get("players") or []
    if side >= len(players):
        return
    ps = players[side]
    for zone in ("hand", "discard", "prize"):
        for c in (ps.get(zone) or []):
            if c and c.get("id") and c.get("serial") is not None:
                yield c["serial"], c["id"], zone
    for grp in ("active", "bench"):
        for pk in (ps.get(grp) or []):
            if not pk:
                continue
            for c in stack_cards(pk):
                if c.get("id") and c.get("serial") is not None:
                    yield c["serial"], c["id"], "play"
    for c in (cur.get("stadium") or []):
        if c and c.get("playerIndex") == side and c.get("id"):
            yield c["serial"], c["id"], "play"


def track_side(episode, side):
    """Follow every revealed card of one side through the game.

    Returns (cards, last_cur) where cards maps
    serial -> {id, zone, zone_step, first_active_step}.
    """
    cards = {}
    last_cur = None
    for step_idx, cur in iter_observations(episode):
        last_cur = cur
        for serial, cid, zone in side_zone_cards(cur, side):
            info = cards.get(serial)
            if info is None:
                info = {"id": cid, "zone": zone, "zone_step": step_idx,
                        "first_active_step": None}
                cards[serial] = info
            elif zone != info["zone"]:
                info["zone"] = zone
                info["zone_step"] = step_idx
            if zone in ("play", "discard") and info["first_active_step"] is None:
                info["first_active_step"] = step_idx
    return cards, last_cur


def board_order(last_cur, side):
    """Serials in play at game end, in stack order: active, bench, stadium."""
    order = []
    players = last_cur.get("players") or []
    if side < len(players):
        ps = players[side]
        for grp in ("active", "bench"):
            for pk in (ps.get(grp) or []):
                if not pk:
                    continue
                for c in stack_cards(pk):
                    if c.get("serial") is not None:
                        order.append(c["serial"])
    for c in (last_cur.get("stadium") or []):
        if c and c.get("playerIndex") == side and c.get("serial") is not None:
            order.append(c["serial"])
    return order


def build_sequence(cards, last_cur, side):
    """The deck read from the game's end backward (list of card ids)."""
    seen = set()
    seq = []

    def take(serial):
        if serial in cards and serial not in seen:
            seen.add(serial)
            seq.append(cards[serial]["id"])

    for serial in board_order(last_cur, side):          # 1. final board
        take(serial)
    for serial in sorted(                               # 2. consumed, latest first
            (s for s, c in cards.items() if c["zone"] == "discard"),
            key=lambda s: -cards[s]["zone_step"]):
        take(serial)
    for serial in sorted(                               # 3. dead cards last
            (s for s, c in cards.items() if c["zone"] in ("hand", "prize")),
            key=lambda s: -cards[s]["zone_step"]):
        take(serial)
    for serial in cards:                                # any stragglers
        take(serial)
    return seq


def play_sequence(cards):
    """The game read forward: cards ordered by first activation."""
    active = [(c["first_active_step"], c["id"])
              for c in cards.values() if c["first_active_step"] is not None]
    active.sort(key=lambda t: t[0])
    return [cid for _, cid in active]


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------
def pack(prefix, card_ids, vocab, length):
    toks = [BOS, prefix] + [vocab.token(c) for c in card_ids]
    toks = toks[:length]
    n = len(toks)
    return np.pad(np.asarray(toks, dtype=np.int16), (0, length - n)), n


_VOCAB = None


def _init_worker(cards_csv):
    global _VOCAB
    _VOCAB = Vocab(cards_csv) if cards_csv else Vocab()


def process_file(path_str):
    """One replay -> (filename, rows, usable).

    rows: list of (build_tokens, build_len, play_tokens, play_len, reward).
    """
    path = pathlib.Path(path_str)
    rows = []
    try:
        ep = json.loads(path.read_text())
        rw = ep.get("rewards") if isinstance(ep, dict) else None
        if not rw or len(rw) != 2 or set(rw) != {1, -1}:
            return path.name, rows, False   # draw, error game, or not a replay
        for side in (0, 1):
            cards, last_cur = track_side(ep, side)
            if last_cur is None or len(cards) < 20:
                continue                    # too little revealed to learn from
            bseq = build_sequence(cards, last_cur, side)
            pseq = play_sequence(cards)
            bt, bl = pack(BUILD, bseq, _VOCAB, BUILD_LEN)
            pt, pl = pack(PLAY, pseq, _VOCAB, PLAY_LEN)
            rows.append((bt, bl, pt, pl, float(rw[side])))
        return path.name, rows, True
    except Exception:
        return path.name, [], False


def default_workers():
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:      # macOS
        cores = os.cpu_count() or 1
    return max(1, cores - 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replays -> builder dataset")
    parser.add_argument("--replays", required=True,
                        help="directory scanned recursively for *.json")
    parser.add_argument("--cards", default=None,
                        help="path to EN_Card_Data.csv")
    parser.add_argument("--out", default="dataset.npz")
    parser.add_argument("--append", action="store_true",
                        help="add to an existing --out npz, skipping "
                             "replay files it already contains")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: cores - 2)")
    args = parser.parse_args()

    vocab = Vocab(args.cards) if args.cards else Vocab()
    files = sorted(pathlib.Path(args.replays).rglob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]

    prior, seen = None, set()
    if args.append and pathlib.Path(args.out).exists():
        prior = dict(np.load(args.out, allow_pickle=False))
        seen = set(prior.get("sources", np.empty(0, dtype="U40")).tolist())
        before = len(files)
        files = [f for f in files if f.name not in seen]
        print(f"append mode: {len(prior['reward'])} sequences already in "
              f"{args.out}; skipping {before - len(files)} converted files",
              flush=True)
    workers = args.workers or default_workers()
    print(f"scanning {len(files)} json files | workers {workers}", flush=True)
    sources = []

    build_toks, build_len = [], []
    play_toks, play_len = [], []
    rewards = []
    used = skipped = 0

    if files:
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(args.cards,)) as pool:
            for i, (name, rows, usable) in enumerate(
                    pool.imap_unordered(process_file,
                                        (str(f) for f in files),
                                        chunksize=8), 1):
                sources.append(name)
                if usable:
                    used += 1
                else:
                    skipped += 1
                for bt, bl, pt, pl, r in rows:
                    build_toks.append(bt)
                    build_len.append(bl)
                    play_toks.append(pt)
                    play_len.append(pl)
                    rewards.append(r)
                if i % 500 == 0:
                    print(f"  {i}/{len(files)} files | "
                          f"{len(rewards)} sequences", flush=True)

    if not rewards and prior is None:
        raise SystemExit("no usable replays found")

    build_toks = (np.stack(build_toks) if rewards
                  else np.empty((0, BUILD_LEN), dtype=np.int16))
    play_toks = (np.stack(play_toks) if rewards
                 else np.empty((0, PLAY_LEN), dtype=np.int16))
    rewards = np.asarray(rewards, dtype=np.float32)
    build_len = np.asarray(build_len, dtype=np.int16)
    play_len = np.asarray(play_len, dtype=np.int16)
    src_arr = np.asarray(sorted(seen | set(sources)), dtype="U40")
    if prior is not None:
        build_toks = np.concatenate([prior["build_tokens"], build_toks])
        play_toks = np.concatenate([prior["play_tokens"], play_toks])
        build_len = np.concatenate([prior["build_len"], build_len])
        play_len = np.concatenate([prior["play_len"], play_len])
        rewards = np.concatenate([prior["reward"], rewards])
    np.savez_compressed(
        args.out,
        build_tokens=build_toks,
        build_len=build_len,
        play_tokens=play_toks,
        play_len=play_len,
        reward=rewards,
        vocab_size=np.int64(vocab.vocab_size),
        sources=src_arr,
    )
    avg_b = float(np.mean(build_len)) - 2   # minus BOS/BUILD
    avg_p = float(np.mean(play_len)) - 2
    print(f"done: {used} replays used, {skipped} skipped this run -> "
          f"TOTAL {len(rewards)} sequences ({int((rewards > 0).sum())} win / "
          f"{int((rewards < 0).sum())} loss)")
    print(f"avg revealed cards per deck: {avg_b:.1f} | "
          f"avg played cards: {avg_p:.1f}")
    print(f"saved {args.out}")
