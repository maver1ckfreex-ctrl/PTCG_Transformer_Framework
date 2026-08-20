"""Merge staged builder datasets into one dataset.npz.

Each stage file was written by replay_to_dataset.py, so they all share the
same schema and the same fixed token widths (BUILD_LEN=64, PLAY_LEN=128).
Merging is a concatenation along the sequence axis; `sources` is the union
of the episode ids each stage consumed.

Nothing about the data is transformed -- the merged file is exactly what a
single --append run over the same days would have produced, and
train_builder.py reads it unchanged.

    python3 merge_dataset.py --out dataset.npz stages/stage_*.npz
"""

import argparse
import pathlib
import sys

import numpy as np

KEYS = ("build_tokens", "build_len", "play_tokens", "play_len", "reward")


def load_stage(path):
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as e:
        sys.exit(f"ERROR: cannot load {path} ({type(e).__name__}: {e})")
    missing = [k for k in KEYS + ("vocab_size",) if k not in z.files]
    if missing:
        sys.exit(f"ERROR: {path} is missing {missing}")
    n = len(z["reward"])
    for k in KEYS:
        if len(z[k]) != n:
            sys.exit(f"ERROR: {path}: {k} has {len(z[k])} rows, reward has "
                     f"{n} -- file is inconsistent")
    return z, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="merge staged builder datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-overlap", action="store_true",
                    help="permit stages that consumed the same episode ids")
    ap.add_argument("parts", nargs="+", help="stage npz files, in order")
    args = ap.parse_args()

    parts = [pathlib.Path(p) for p in args.parts]
    for p in parts:
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")

    acc = {k: [] for k in KEYS}
    sources = []
    seen = set()
    vocab_size = None
    widths = {}
    total = 0

    for p in parts:
        z, n = load_stage(p)
        vs = int(z["vocab_size"])
        if vocab_size is None:
            vocab_size = vs
        elif vs != vocab_size:
            sys.exit(f"ERROR: {p} has vocab_size {vs}, earlier stages have "
                     f"{vocab_size} -- these were built against different "
                     f"card data and must not be merged")
        for k in ("build_tokens", "play_tokens"):
            w = z[k].shape[1] if z[k].ndim == 2 else None
            if k in widths and w != widths[k]:
                sys.exit(f"ERROR: {p}: {k} width {w} != {widths[k]}")
            widths[k] = w

        src = set(z["sources"].tolist()) if "sources" in z.files else set()
        dup = src & seen
        if dup and not args.allow_overlap:
            sys.exit(f"ERROR: {p} re-uses {len(dup)} episode ids already "
                     f"consumed by an earlier stage (e.g. "
                     f"{sorted(dup)[:3]}). Merging would double-count them; "
                     f"pass --allow-overlap only if that is intended.")
        seen |= src
        sources.append(z["sources"] if "sources" in z.files
                       else np.empty(0, dtype="U40"))

        for k in KEYS:
            acc[k].append(z[k])
        total += n
        wins = int((z["reward"] > 0).sum())
        print(f"  {p.name}: {n} sequences ({wins} win / {n - wins} loss)")

    merged = {k: np.concatenate(acc[k]) for k in KEYS}
    src_arr = np.asarray(sorted(seen), dtype="U40")

    np.savez_compressed(
        args.out,
        build_tokens=merged["build_tokens"],
        build_len=merged["build_len"],
        play_tokens=merged["play_tokens"],
        play_len=merged["play_len"],
        reward=merged["reward"],
        vocab_size=np.int64(vocab_size),
        sources=src_arr,
    )
    r = merged["reward"]
    print(f"\nmerged {len(parts)} stages -> {total} sequences "
          f"({int((r > 0).sum())} win / {int((r < 0).sum())} loss) | "
          f"vocab {vocab_size} | {len(src_arr)} distinct episodes")
    print(f"saved {args.out}")
