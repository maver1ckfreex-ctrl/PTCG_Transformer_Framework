"""Randomly sample N whole trajectories out of a trajectory npz.

Used by mix-ratio training: after a self-play cycle produces n winning
trajectories, this pulls a matching count out of the kaggle dataset so the
two can be merged at a chosen ratio.

Sampling is at TRAJECTORY granularity -- a trajectory is taken whole or not
at all, so decision order and the deck prefix stay intact.

    python3 tools/sample_npz.py --in kaggle.npz --n 12000 --out sub.npz
    python3 tools/sample_npz.py --in kaggle.npz --n 12000 --out sub.npz --seed 7
"""

import argparse
import pathlib
import sys

import numpy as np

FLAT_PAIRS = [("tok_flat", "tok_off"),
              ("pos_flat", "pos_off"),
              ("chosen_flat", "chosen_off")]


def sample(src, n, out, seed=0):
    d = np.load(src, allow_pickle=False)
    reward = d["reward"]
    traj_off = d["traj_off"]
    deck_off = d["deck_off"]
    n_traj = len(reward)

    if n >= n_traj:
        print(f"  requested {n} >= {n_traj} available: copying all")
        pick = np.arange(n_traj)
    else:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(n_traj, size=n, replace=False))

    tok_flat, tok_off = [], [0]
    pos_flat, pos_off = [], [0]
    cho_flat, cho_off = [], [0]
    deck_flat, deck_off_new = [], [0]
    new_traj_off = [0]
    rewards = []

    tf, pf, cf, df = (d["tok_flat"], d["pos_flat"],
                      d["chosen_flat"], d["deck_flat"])
    to, po, co = d["tok_off"], d["pos_off"], d["chosen_off"]

    for g in pick:
        a, b = deck_off[g], deck_off[g + 1]
        deck_flat.append(df[a:b])
        deck_off_new.append(deck_off_new[-1] + (b - a))
        for j in range(traj_off[g], traj_off[g + 1]):
            tok_flat.append(tf[to[j]:to[j + 1]])
            tok_off.append(tok_off[-1] + (to[j + 1] - to[j]))
            pos_flat.append(pf[po[j]:po[j + 1]])
            pos_off.append(pos_off[-1] + (po[j + 1] - po[j]))
            cho_flat.append(cf[co[j]:co[j + 1]])
            cho_off.append(cho_off[-1] + (co[j + 1] - co[j]))
        new_traj_off.append(len(tok_off) - 1)
        rewards.append(reward[g])

    cat = lambda parts, dt: (np.concatenate(parts) if parts
                             else np.empty(0, dtype=dt))
    np.savez_compressed(
        out,
        vocab_size=d["vocab_size"],
        tok_flat=cat(tok_flat, np.int16),
        tok_off=np.asarray(tok_off, dtype=np.int64),
        pos_flat=cat(pos_flat, np.int16),
        pos_off=np.asarray(pos_off, dtype=np.int64),
        chosen_flat=cat(cho_flat, np.int16),
        chosen_off=np.asarray(cho_off, dtype=np.int64),
        deck_flat=cat(deck_flat, np.int16),
        deck_off=np.asarray(deck_off_new, dtype=np.int64),
        traj_off=np.asarray(new_traj_off, dtype=np.int64),
        reward=np.asarray(rewards, dtype=np.float32),
        sources=d["sources"],
    )
    n_dec = len(tok_off) - 1
    print(f"sampled {len(pick)}/{n_traj} trajectories ({n_dec} decisions) "
          f"-> {out}")
    return len(pick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="random trajectory subset")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not pathlib.Path(args.src).exists():
        sys.exit(f"missing {args.src}")
    sample(args.src, args.n, args.out, args.seed)
