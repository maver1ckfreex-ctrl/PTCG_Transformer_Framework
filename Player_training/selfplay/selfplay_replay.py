"""Generate replay files from engine self-play, in Kaggle replay format.

TEMPERATURE VARIANT: the player can SAMPLE its move from a softmax over
the option scores instead of always taking the argmax. T=0 keeps the
old deterministic behaviour; higher T means more exploration, so the
two seats of a game diverge and the data contains moves the greedy
policy would not have made. --play-temps accepts several values and
draws one per game, so a single run mixes deterministic and exploratory
play.

Every field comes from the engine itself — `current` (state), `select`
(options offered) and the actions actually taken. Nothing about the game
is invented here; this only wraps genuine engine output in the same
envelope the Kaggle CLI downloads, so replay_to_dataset.py and
replay_to_decisions.py read these files unchanged.

Kaggle envelope (verified against real downloaded replays):
    {"steps": [[agent0_entry, agent1_entry], ...],
     "rewards": [1, -1],
     "info": {"TeamNames": [...]}}
  entry = {"observation": {"current":..., "select":...},
           "action": <the action answering the PREVIOUS step's
                      observation for this agent — the engine uses the
                      same one-step shift, verified 360/360>,
           "status": "ACTIVE"/"INACTIVE"/"DONE",
           "reward": <final reward>}

Usage (normally driven by selfplay_loop.py):
    python3 selfplay_replay.py --decks decks_dir --player player_r2.pt \
        --engine ../submission_SD --games 1000 --out replays/ --workers 120
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
import json
import multiprocessing as mp
import pathlib
import queue
import random
import sys
import time

import numpy as np

MAX_SELECTIONS_PER_GAME = 2000


def make_player(tf_dir, player_ckpt, play_temps=(0.0,), seed=0xBEEF):
    """Returns a pick(obs)->list[int] closure using the trained player."""
    sys.path.insert(0, str(tf_dir))
    import torch
    torch.set_num_threads(1)
    from player_vocab import PlayerVocab, tokenize_decision
    from player_model import PlayerDecoder

    ck = torch.load(player_ckpt, map_location="cpu")
    model = PlayerDecoder(ck["vocab_size"], max_len=ck["max_len"])
    model.load_state_dict(ck["model"])
    model.eval()
    pv = PlayerVocab()
    rng = random.Random(seed)
    play_temps = tuple(play_temps) or (0.0,)

    state = {"T": play_temps[0]}

    def new_game():
        """called once per game: pick this game's play temperature"""
        state["T"] = rng.choice(play_temps)
        return state["T"]

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

        T = state["T"]
        if T > 0:
            # Gumbel-top-k: adding Gumbel noise to the logits and sorting
            # gives exactly a sample-without-replacement from
            # softmax(scores/T). No normalisation, so it cannot fail when
            # softmax underflows (np.choice(replace=False, p=...) raises
            # "Fewer non-zero entries in p than size" on large gaps).
            z = np.asarray(scores, dtype=np.float64) / max(T, 1e-6)
            g = np.random.default_rng(rng.randrange(1 << 30)).gumbel(
                size=z.shape)
            # int() is mandatory: the engine rejects np.int64
            # (`all(isinstance(i, int) ...)` in cg/game.py)
            order = [int(i) for i in np.argsort(-(z + g))]
        else:                           # argmax, as before
            order = sorted(range(len(scores)), key=lambda i: scores[i],
                           reverse=True)
        dec = n if min_c == 0 else None
        real = [i for i in order if i < n]
        if dec is not None and order[0] == dec:
            return []
        picked = real[:max(min_c, 1)]
        for i in real[max(min_c, 1):max_c]:
            if dec is not None and scores[i] <= scores[dec]:
                break
            picked.append(i)
        return sorted(picked)

    return pick, new_game


def play_and_record(battle_start, battle_select, battle_finish,
                    deck0, deck1, pick, names):
    """Play one game; return the Kaggle-format replay dict (or None)."""
    obs, _ = battle_start(deck0, deck1)
    if obs is None:
        return None
    frames = []          # (mover, observation, action)
    steps = 0
    while obs["current"]["result"] == -1 and steps < MAX_SELECTIONS_PER_GAME:
        mover = obs["current"]["yourIndex"]
        action = pick(obs)
        frames.append((mover, obs, action))
        obs = battle_select(action)
        steps += 1
    result = obs["current"]["result"]
    frames.append((obs["current"]["yourIndex"], obs, []))   # terminal frame
    battle_finish()

    if result == 0:
        rewards = [1, -1]
    elif result == 1:
        rewards = [-1, 1]
    else:
        return None                      # draw / step-cap: no ±1 label

    kaggle_steps = []
    for i, (mover, o, _act) in enumerate(frames):
        entries = []
        for a in (0, 1):
            # the action recorded at step i is this agent's answer to the
            # observation it saw at step i-1 (engine + Kaggle convention)
            prev_mover, _po, prev_act = frames[i - 1] if i > 0 else (None, None, [])
            entries.append({
                "action": list(prev_act) if prev_mover == a else [],
                "reward": rewards[a],
                "info": {},
                "observation": {
                    "current": o["current"],
                    # only the agent to move is offered a selection
                    "select": o.get("select") if a == mover else None,
                },
                "status": ("DONE" if o["current"]["result"] != -1
                           else ("ACTIVE" if a == mover else "INACTIVE")),
            })
        kaggle_steps.append(entries)

    # Kaggle records the two competitors' user ids here. In self-play both
    # seats are the same model, so they are simply registered as p1 / p2.
    # The trajectory converter needs each side's 60-card list. Kaggle
    # replays carry it as the step-1 action; self-play recording starts
    # after battle_start, so record it explicitly here instead.
    return {"steps": kaggle_steps, "rewards": rewards,
            "decks": [list(deck0), list(deck1)],
            "info": {"TeamNames": ["p1", "p2"]}}


def worker(wid, engine_dir, tf_dir, player_ckpt, jobs, deck_table, out_dir, q,
           play_temps=(0.0,)):
    """jobs are (gid, i, j) index pairs; deck_table holds only the decks
    this worker needs — passing every deck to every worker would ship
    hundreds of MB per process at 200k decks."""
    sys.path.insert(0, str(tf_dir))
    sys.path.insert(0, str(engine_dir))
    from cg.game import battle_start, battle_select, battle_finish
    pick, new_game = make_player(tf_dir, player_ckpt, play_temps,
                                 seed=wid * 7919 + 23)

    for (gid, i, j) in jobs:
        new_game()                     # this game's play temperature
        try:
            rep = play_and_record(battle_start, battle_select, battle_finish,
                                  deck_table[i], deck_table[j], pick, None)
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
        path = pathlib.Path(out_dir) / f"sp_{gid:08d}.json"
        with open(path, "w") as f:
            json.dump(rep, f, separators=(",", ":"))
        q.put(("ok", len(rep["steps"])))


def generate(deck_files, player_ckpt, engine_dir, tf_dir, out_dir,
             n_games, n_workers, seed=0, tag="sp", play_temps=(0.0,)):
    """Play n_games; each game draws two decks from deck_files."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (len(deck_files) == 1
            and str(deck_files[0]).endswith(".npy")):     # GPU-sampled pool
        decks = [[int(c) for c in row] for row in np.load(deck_files[0])]
    else:
        decks = [[int(x) for x in pathlib.Path(p).read_text().split()]
                 for p in deck_files]
    rng = random.Random(seed)

    jobs = []
    order = list(range(len(decks)))
    for g in range(n_games):
        if g % max(1, len(order) // 2) == 0:
            rng.shuffle(order)
        i = order[(2 * g) % len(order)]
        j = order[(2 * g + 1) % len(order)]
        if i == j:
            j = (j + 1) % len(order)
        jobs.append((g, i, j))

    chunks = [jobs[i::n_workers] for i in range(n_workers)]
    q = mp.Queue()
    procs = []
    for w, ch in enumerate(chunks):
        if not ch:
            continue
        need = {i for (_g, i, j) in ch} | {j for (_g, i, j) in ch}
        sub = {i: decks[i] for i in need}     # only this worker's decks
        p = mp.Process(target=worker,
                       args=(w, str(engine_dir), str(tf_dir), player_ckpt,
                             ch, sub, str(out_dir), q, play_temps))
        p.start()
        procs.append(p)

    ok = skip = err = 0
    t0 = time.time()
    total = len(jobs)
    done = 0
    while done < total:
        try:
            kind, _ = q.get(timeout=300)
        except queue.Empty:
            # nothing for 5 min: if every worker has exited, stop waiting
            # instead of blocking for an hour on a crashed run
            if not any(p.is_alive() for p in procs):
                print(f"  !! all workers exited early at {done}/{total} "
                      f"games — continuing with what was written",
                      flush=True)
                break
            continue
        done += 1
        ok += kind == "ok"
        skip += kind == "skip"
        err += kind == "error"
        if done % 5000 == 0:
            el = time.time() - t0
            print(f"  {done}/{total} games | {done/el:.0f} g/s | "
                  f"eta {(total-done)/max(done/el,1e-9)/60:.1f} min",
                  flush=True)
    for p in procs:
        p.terminate()
    print(f"  replays written: {ok} | skipped(draw): {skip} | errors: {err} "
          f"| {(time.time()-t0)/60:.1f} min", flush=True)
    return ok


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser(description="self-play -> Kaggle replays")
    ap.add_argument("--decks", required=True, help="folder of deck csvs")
    ap.add_argument("--player", required=True)
    ap.add_argument("--engine", default=str(pathlib.Path(__file__).parent))
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--play-temps", default="0",
                    help="comma-separated player temperatures; one is drawn "
                         "per game (0 = argmax). e.g. 0,0.3,0.6,1.0")
    args = ap.parse_args()

    here = pathlib.Path(__file__).parent
    src = pathlib.Path(args.decks)
    if src.is_file() and src.suffix == ".npy":          # GPU-sampled pool
        files = [src]
    elif src.is_dir() and (src.with_suffix(".npy")).exists():
        files = [src.with_suffix(".npy")]
    else:
        files = sorted(src.glob("*.csv")) if src.is_dir() else []
        if not files:
            sys.exit(f"no deck csvs or .npy pool at {args.decks}")
    nw = args.workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"self-play: {args.games} games from {len(files)} decks "
          f"| player {pathlib.Path(args.player).name} | workers {nw}",
          flush=True)
    ptemps = tuple(float(x) for x in args.play_temps.split(","))
    print(f"player temperatures: {ptemps} (one drawn per game)", flush=True)
    generate(files, args.player, pathlib.Path(args.engine).resolve(), here,
             args.out, args.games, nw, args.seed, play_temps=ptemps)
