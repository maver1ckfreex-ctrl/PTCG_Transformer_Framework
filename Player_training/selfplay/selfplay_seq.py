"""Self-play that can pilot ANY arch, including the trajectory models.

selfplay_replay.py loads only R2-era PlayerDecoder checkpoints, so it
cannot drive a sequence_v3 model -- which makes iterative self-play
impossible past the first cycle. This uses the same Policy wrapper the
tournament uses, so r2 / baseline / sequence / sequence_v3 / sequence_v3b
all work.

The important difference from selfplay_replay.py: TWO policies, one per
seat. A trajectory model carries per-game memory conditioned on ITS OWN
deck and ITS OWN earlier decisions. Sharing one runner across both seats
would interleave two games into one memory chain and corrupt both sides.

Output is Kaggle episode format plus an explicit "decks" field, so
replay_to_trajectories.py reads it directly.

    python3 selfplay/selfplay_seq.py --decks pool.npy --player model.pt \\
        --engine ../submission_r2_t07_torch --out replays_selfplay \\
        --games 10000 --workers 96 --play-temps 0.0,0.1,0.2
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import pathlib
import random
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MAX_SELECTIONS_PER_GAME = 2000


def play_one(battle_start, battle_select, battle_finish, deck0, deck1, pols):
    """One game, both seats driven by their own policy. -> replay dict."""
    obs, _ = battle_start(deck0, deck1)
    if obs is None:
        return None
    frames = []
    steps = 0
    while obs["current"]["result"] == -1 and steps < MAX_SELECTIONS_PER_GAME:
        mover = obs["current"]["yourIndex"]
        action = pols[mover].select(obs)
        frames.append((mover, obs, action))
        obs = battle_select(action)
        steps += 1
    result = obs["current"]["result"]
    frames.append((obs["current"]["yourIndex"], obs, []))
    battle_finish()

    if result == 0:
        rewards = [1, -1]
    elif result == 1:
        rewards = [-1, 1]
    else:
        return None                       # draw / step cap: no +/-1 label

    kaggle_steps = []
    for i, (mover, o, _a) in enumerate(frames):
        entries = []
        for a in (0, 1):
            prev_mover, _po, prev_act = (frames[i - 1] if i > 0
                                         else (None, None, []))
            entries.append({
                "action": list(prev_act) if prev_mover == a else [],
                "reward": rewards[a],
                "info": {},
                "observation": {
                    "current": o["current"],
                    "select": o.get("select") if a == mover else None,
                },
                "status": ("DONE" if o["current"]["result"] != -1
                           else ("ACTIVE" if a == mover else "INACTIVE")),
            })
        kaggle_steps.append(entries)

    return {"steps": kaggle_steps, "rewards": rewards,
            "decks": [list(deck0), list(deck1)],
            "info": {"TeamNames": ["p1", "p2"]}}


def worker(wid, engine_dir, ckpt, jobs, deck_table, out_dir, q,
           play_temps, seed):
    sys.path.insert(0, str(ROOT / "eval"))
    sys.path.insert(0, str(engine_dir))
    from tournament_seq import Policy           # handles every arch
    from cg.game import battle_start, battle_select, battle_finish

    # one policy per SEAT: trajectory memory is per game and per deck
    pols = [Policy(ckpt, 0.0, seed=0x5EED + wid * 2 + s) for s in (0, 1)]
    rng = random.Random(seed + wid)
    out_dir = pathlib.Path(out_dir)

    for gid, i, j in jobs:
        d0, d1 = deck_table[i], deck_table[j]
        t = rng.choice(play_temps)
        for s in (0, 1):
            pols[s].temp = float(t)
        pols[0].reset(d0)                       # memory reset + deck prefix
        pols[1].reset(d1)
        try:
            rep = play_one(battle_start, battle_select, battle_finish,
                           d0, d1, pols)
        except Exception:
            try:
                battle_finish()
            except Exception:
                pass
            q.put(("error", 0))
            continue
        if rep is None:
            q.put(("skip", 0))
            continue
        with open(out_dir / f"sp_{gid:08d}.json", "w") as f:
            json.dump(rep, f, separators=(",", ":"))
        q.put(("ok", len(rep["steps"])))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser(description="self-play, any arch")
    ap.add_argument("--decks", required=True, help="deck pool .npy or dir")
    ap.add_argument("--player", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--play-temps", default="0.0")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(args.decks)
    if src.is_file() and src.suffix == ".npy":
        decks = [[int(c) for c in row] for row in np.load(src)]
    else:
        files = sorted(src.glob("*.csv")) if src.is_dir() else []
        if not files:
            sys.exit(f"no deck pool at {args.decks}")
        decks = [[int(x) for x in p.read_text().split()] for p in files]

    temps = tuple(float(x) for x in args.play_temps.split(","))
    nw = args.workers or (os.cpu_count() or 2)
    rng = random.Random(args.seed)

    jobs = []
    order = list(range(len(decks)))
    for g in range(args.games):
        if g % max(1, len(order) // 2) == 0:
            rng.shuffle(order)
        i = order[(2 * g) % len(order)]
        j = order[(2 * g + 1) % len(order)]
        if i == j:
            j = (j + 1) % len(order)
        jobs.append((g, i, j))

    print(f"self-play: {args.games} games | {len(decks)} decks | "
          f"player {pathlib.Path(args.player).name} | workers {nw}", flush=True)
    print(f"play temperatures: {temps} (one drawn per game)", flush=True)

    q = mp.Queue()
    chunks = [jobs[k::nw] for k in range(nw)]
    procs = []
    for k, ch in enumerate(chunks):
        if not ch:
            continue
        need = {i for _, i, j in ch} | {j for _, i, j in ch}
        sub = {i: decks[i] for i in need}       # only this worker's decks
        p = mp.Process(target=worker,
                       args=(k, str(pathlib.Path(args.engine).resolve()),
                             str(pathlib.Path(args.player).resolve()),
                             ch, sub, str(out_dir), q, temps, args.seed))
        p.start()
        procs.append(p)

    ok = skip = err = 0
    t0 = time.time()
    for _ in range(len(jobs)):
        kind, _n = q.get()
        if kind == "ok":
            ok += 1
        elif kind == "skip":
            skip += 1
        else:
            err += 1
        done = ok + skip + err
        if done % 200 == 0:
            print(f"  {done}/{len(jobs)} | ok {ok} skip {skip} err {err} "
                  f"| {time.time()-t0:.0f}s", flush=True)
    for p in procs:
        p.join()
    print(f"done: {ok} replays written, {skip} draws skipped, {err} errors "
          f"| {time.time()-t0:.0f}s -> {out_dir}")
