"""Verify a trajectories npz is COMPLETE and self-consistent.

Dual-GPU mode runs this on every trainer's dataset before a single GPU
starts. A truncated or half-written npz that still loads would otherwise
train silently on partial data, and you would not find out until the
tournament.

Checks:
  * file exists, loads, and carries every required array
  * offset arrays are monotonic, start at 0, and end at the flat length
  * traj_off spans exactly the decision count
  * one 60-card deck per trajectory
  * reward is one per trajectory and strictly +/-1
  * nothing is empty
  * --winners-only: every reward is +1

    python3 tools/verify_data.py <npz> [--winners-only] [--min-trajectories N]

Exit 0 = usable. Non-zero = do not train on it.
"""

import argparse
import pathlib
import sys

import numpy as np

REQUIRED = ["vocab_size", "tok_flat", "tok_off", "pos_flat", "pos_off",
            "chosen_flat", "chosen_off", "deck_flat", "deck_off",
            "traj_off", "reward", "sources"]
DECK_SIZE = 60


def check(path, winners_only=False, min_traj=1):
    errs, notes = [], []
    p = pathlib.Path(path)
    if not p.exists():
        return [f"missing file: {p}"], []
    if p.stat().st_size == 0:
        return [f"zero-byte file: {p}"], []

    try:
        d = np.load(p)
    except Exception as e:
        return [f"cannot load ({type(e).__name__}: {e})"], []

    missing = [k for k in REQUIRED if k not in d.files]
    if missing:
        return [f"missing arrays: {missing}"], []

    traj_off = d["traj_off"]
    reward = d["reward"]
    deck_off = d["deck_off"]
    n_traj = len(reward)
    n_dec = len(d["tok_off"]) - 1

    def mono(name, off, flat_len):
        a = d[off]
        if a[0] != 0:
            errs.append(f"{off}[0] = {a[0]}, expected 0")
        if np.any(np.diff(a) < 0):
            errs.append(f"{off} is not monotonic")
        if a[-1] != flat_len:
            errs.append(f"{off}[-1] = {a[-1]} but {name} has {flat_len}")

    mono("tok_flat", "tok_off", len(d["tok_flat"]))
    mono("pos_flat", "pos_off", len(d["pos_flat"]))
    mono("chosen_flat", "chosen_off", len(d["chosen_flat"]))
    mono("deck_flat", "deck_off", len(d["deck_flat"]))

    if len(d["pos_off"]) - 1 != n_dec:
        errs.append(f"pos_off implies {len(d['pos_off'])-1} decisions, "
                    f"tok_off implies {n_dec}")
    if len(d["chosen_off"]) - 1 != n_dec:
        errs.append(f"chosen_off implies {len(d['chosen_off'])-1} decisions, "
                    f"tok_off implies {n_dec}")

    if len(traj_off) != n_traj + 1:
        errs.append(f"traj_off has {len(traj_off)} entries, expected "
                    f"{n_traj + 1} for {n_traj} trajectories")
    elif traj_off[-1] != n_dec:
        errs.append(f"traj_off ends at {traj_off[-1]} but there are "
                    f"{n_dec} decisions")
    if len(deck_off) != n_traj + 1:
        errs.append(f"deck_off has {len(deck_off)} entries, expected "
                    f"{n_traj + 1}")
    else:
        sizes = np.diff(deck_off)
        bad = int((sizes != DECK_SIZE).sum())
        if bad:
            errs.append(f"{bad} trajectories do not have exactly "
                        f"{DECK_SIZE} deck cards")

    if n_traj < min_traj:
        errs.append(f"only {n_traj} trajectories (need >= {min_traj})")
    if n_dec == 0:
        errs.append("zero decisions")

    uniq = np.unique(reward)
    if not np.all(np.isin(uniq, [-1.0, 1.0])):
        errs.append(f"reward has values outside +/-1: {uniq[:8]}")

    empty = int((np.diff(traj_off) == 0).sum()) if len(traj_off) > 1 else 0
    if empty:
        errs.append(f"{empty} trajectories contain zero decisions")

    n_win = int((reward > 0).sum())
    if winners_only and n_win != n_traj:
        errs.append(f"--winners-only expected all {n_traj} trajectories to "
                    f"be wins, found {n_traj - n_win} losses")

    lens = np.diff(traj_off)
    notes.append(f"trajectories {n_traj} ({100*n_win/max(n_traj,1):.1f}% won)")
    notes.append(f"decisions    {n_dec}")
    notes.append(f"per traj     mean {lens.mean():.1f} median "
                 f"{int(np.median(lens))} max {int(lens.max())}")
    notes.append(f"tokens/dec   {len(d['tok_flat'])/max(n_dec,1):.0f}")
    notes.append(f"vocab        {int(d['vocab_size'])}")
    notes.append(f"source files {len(d['sources'])}")
    notes.append(f"size on disk {p.stat().st_size/1e9:.2f} GB")
    return errs, notes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="verify a trajectories npz")
    ap.add_argument("npz")
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--min-trajectories", type=int, default=1)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    tag = args.label or pathlib.Path(args.npz).name
    errs, notes = check(args.npz, args.winners_only, args.min_trajectories)
    print(f"=== verify {tag}: {args.npz}")
    for n in notes:
        print(f"    {n}")
    if errs:
        print(f"    FAIL ({len(errs)} problem(s)):")
        for e in errs:
            print(f"      * {e}")
        sys.exit(1)
    print("    OK -- complete and self-consistent")
