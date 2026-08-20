"""Merge trajectory npz shards into one dataset.

Chunked parsing writes one npz per chunk of days. This concatenates them
into the single file the trainers expect.

The format is flat arrays plus offset arrays, so merging is: concatenate
every flat array, and shift each subsequent offset array past the length of
what came before (dropping its leading 0). traj_off indexes into DECISION
space rather than token space, so it shifts by the decision count.

    python3 tools/merge_npz.py --out final.npz part_000.npz part_001.npz ...
    python3 tools/merge_npz.py --out final.npz --glob 'parts/part_*.npz'
"""

import argparse
import glob as globmod
import pathlib
import sys

import numpy as np

FLAT_PAIRS = [("tok_flat", "tok_off"),
              ("pos_flat", "pos_off"),
              ("chosen_flat", "chosen_off"),
              ("deck_flat", "deck_off")]


def merge(paths, out):
    paths = [str(p) for p in paths]
    if not paths:
        raise SystemExit("no shards to merge")

    acc = None
    vocab = None
    sources = set()
    n_traj = n_dec = 0

    for i, p in enumerate(paths):
        d = np.load(p, allow_pickle=False)
        v = int(d["vocab_size"])
        if vocab is None:
            vocab = v
        elif v != vocab:
            raise SystemExit(f"vocab mismatch: {paths[0]} has {vocab}, "
                             f"{p} has {v}")
        sources.update(d["sources"].tolist())

        if acc is None:
            acc = {k: d[k].copy() for k in d.files if k != "sources"}
        else:
            for flat, off in FLAT_PAIRS:
                base = acc[off][-1]
                acc[off] = np.concatenate([acc[off], d[off][1:] + base])
                acc[flat] = np.concatenate([acc[flat], d[flat]])
            # traj_off points into the DECISION arrays, not the flat ones
            dec_base = acc["traj_off"][-1]
            acc["traj_off"] = np.concatenate(
                [acc["traj_off"], d["traj_off"][1:] + dec_base])
            acc["reward"] = np.concatenate([acc["reward"], d["reward"]])

        n_traj = len(acc["reward"])
        n_dec = len(acc["tok_off"]) - 1
        print(f"  [{i+1}/{len(paths)}] {pathlib.Path(p).name}: "
              f"+{len(d['reward'])} traj -> {n_traj} total, {n_dec} decisions",
              flush=True)

    acc["vocab_size"] = np.int64(vocab)
    acc["sources"] = np.asarray(sorted(sources), dtype="U40")

    # cheap internal consistency check before writing
    if acc["traj_off"][-1] != n_dec:
        raise SystemExit(f"merge bug: traj_off ends {acc['traj_off'][-1]}, "
                         f"expected {n_dec}")
    for flat, off in FLAT_PAIRS:
        if acc[off][-1] != len(acc[flat]):
            raise SystemExit(f"merge bug: {off}[-1]={acc[off][-1]} but "
                             f"{flat} has {len(acc[flat])}")

    np.savez_compressed(out, **acc)
    lens = np.diff(acc["traj_off"])
    print(f"merged {len(paths)} shards -> {out}")
    print(f"  {n_traj} trajectories ({int((acc['reward']>0).sum())} winning) "
          f"| {n_dec} decisions")
    print(f"  per traj: mean {lens.mean():.1f} max {int(lens.max())} "
          f"| {len(sources)} source files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="merge trajectory npz shards")
    ap.add_argument("shards", nargs="*")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = list(args.shards)
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    paths = sorted(set(paths))
    if not paths:
        sys.exit("no shards given")
    print(f"merging {len(paths)} shards")
    merge(paths, args.out)
