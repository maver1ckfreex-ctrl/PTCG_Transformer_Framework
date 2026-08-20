"""Staged download + parse: N days at a time, checkpoint, delete raw, repeat.

Without staging, pipeline.py appends every day into one growing
dataset.npz. That file is rewritten in full on each append, so the cost per
day climbs with the archive, and a crash late in a long run leaves one
half-written file.

With --stage_pursing=1 the manifest is cut into blocks of --split_days
(default 20). Each block is downloaded, parsed into its OWN stage file,
verified, and only then counted as done -- and pipeline.py deletes each
day's raw folder and zip as soon as that day is converted, so disk never
holds more than a block's worth. When every block is done the stage files
are merged into one dataset.npz for training.

    stage 001  days 1-20   -> stages/stage_001.npz   raw deleted
    stage 002  days 21-40  -> stages/stage_002.npz   raw deleted
    ...
    merge_dataset.py stages/stage_*.npz -> dataset.npz

Resume is free: a stage whose npz exists and verifies is skipped, and
inside a block pipeline_state.json still tracks finished days, so an
interrupted block restarts at the day it stopped on.

Nothing here changes how replays are parsed or how the builder trains --
pipeline.py, replay_to_dataset.py and train_builder.py are untouched.

    python3 stage_pursing.py --manifest manifest_01Aug.csv \\
        --work /local/ptcg_work --split_days 20
"""

import argparse
import csv
import json
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_days(manifest, start, end):
    with open(manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("date")]
    days = [r["date"] for r in rows]
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    return sorted(days)


def verify_stage(path, label):
    """A stage counts as done only if it loads and is self-consistent.
    A truncated npz raises here rather than at merge time."""
    try:
        z = np.load(path, allow_pickle=False)
        n = len(z["reward"])
        for k in ("build_tokens", "build_len", "play_tokens", "play_len"):
            if len(z[k]) != n:
                log(f"  {label}: {k} has {len(z[k])} rows vs reward {n}")
                return False
        if n == 0:
            log(f"  {label}: 0 sequences")
            return False
        wins = int((z["reward"] > 0).sum())
        log(f"  {label}: OK -- {n} sequences ({wins} win / {n - wins} loss), "
            f"vocab {int(z['vocab_size'])}")
        return True
    except Exception as e:
        log(f"  {label}: cannot load ({type(e).__name__}: {e})")
        return False


def sweep_raw(work):
    """Belt and braces: pipeline deletes each day as it converts it, but a
    block interrupted mid-download can leave a partial folder or zip."""
    freed = 0
    for sub in ("raw", "zips"):
        d = work / sub
        if not d.exists():
            continue
        for child in d.iterdir():
            try:
                if child.is_dir():
                    freed += sum(f.stat().st_size for f in child.rglob("*")
                                 if f.is_file())
                    shutil.rmtree(child)
                else:
                    freed += child.stat().st_size
                    child.unlink()
            except Exception:
                pass
    if freed:
        log(f"  swept {freed / 1e9:.1f} GB of leftover raw data")


def forget_days(work, block):
    """Drop a block's days from pipeline_state.json.

    The state file marks a day done as soon as it is converted. If a stage
    has to be rebuilt, its days are already marked, and pipeline.py would
    find nothing pending and write no dataset at all. Rebuilding therefore
    has to forget them first."""
    state_path = work / "pipeline_state.json"
    if not state_path.exists():
        return
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return
    done = [d for d in state.get("done", []) if d not in set(block)]
    if len(done) != len(state.get("done", [])):
        state["done"] = done
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(state_path)
        log(f"  forgot {len(block)} days in pipeline_state.json so they "
            f"re-download")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="staged download+parse for the builder dataset")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--split_days", type=int, default=20,
                    help="days per stage (default 20)")
    ap.add_argument("--dataset-out", default=str(HERE / "dataset.npz"),
                    help="final merged dataset")
    ap.add_argument("--stages-dir", default=None,
                    help="where stage npz files live (default <work>/stages)")
    ap.add_argument("--cards", default=None)
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--keep-stages", action="store_true",
                    help="keep the per-stage npz files after merging")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip downloading; just merge existing stages")
    args = ap.parse_args()

    if args.split_days < 1:
        sys.exit("ERROR: --split_days must be >= 1")

    work = pathlib.Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    stages_dir = pathlib.Path(args.stages_dir) if args.stages_dir \
        else work / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    days = read_days(args.manifest, args.start_date, args.end_date)
    if not days:
        sys.exit("ERROR: no days in range")
    blocks = [days[i:i + args.split_days]
              for i in range(0, len(days), args.split_days)]

    log(f"staged parse: {len(days)} days in {len(blocks)} stages of "
        f"{args.split_days}")
    log(f"  work      : {work}")
    log(f"  stages    : {stages_dir}")
    log(f"  final     : {args.dataset_out}")

    stage_files = []
    t_start = time.time()
    for i, block in enumerate(blocks, 1):
        tag = f"stage_{i:03d}"
        out = stages_dir / f"{tag}.npz"
        stage_files.append(out)

        if out.exists() and verify_stage(out, tag):
            log(f"=== {tag} ({block[0]}..{block[-1]}) already done, skipping")
            continue
        if args.merge_only:
            sys.exit(f"ERROR: --merge-only but {tag} is missing or bad")
        if out.exists():
            log(f"=== {tag} exists but failed verification -- rebuilding")
            out.unlink()
            forget_days(work, block)

        log(f"=== {tag}: days {block[0]}..{block[-1]} ({len(block)} days)")
        t0 = time.time()
        cmd = [sys.executable, str(HERE / "pipeline.py"),
               "--manifest", args.manifest,
               "--work", str(work),
               "--convert", "dataset",
               "--dataset-out", str(out),
               "--start-date", block[0],
               "--end-date", block[-1],
               "--prefetch", str(args.prefetch),
               "--retries", str(args.retries)]
        if args.cards:
            cmd += ["--cards", args.cards]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"{tag} failed (exit {r.returncode}). Nothing was "
                     f"discarded -- re-run the same command to resume.")

        if not out.exists() or not verify_stage(out, tag):
            sys.exit(f"{tag} produced no usable dataset -- raw data for this "
                     f"block was already deleted per day by pipeline.py, so "
                     f"re-running will re-download it.")
        sweep_raw(work)
        log(f"=== {tag} done in {(time.time() - t0) / 60:.1f} min "
            f"({(time.time() - t_start) / 60:.0f} min total)")

    log(f"all {len(blocks)} stages present -- merging")
    r = subprocess.run([sys.executable, str(HERE / "merge_dataset.py"),
                        "--out", args.dataset_out]
                       + [str(p) for p in stage_files])
    if r.returncode != 0:
        sys.exit("merge failed; the stage files are intact, fix and re-run "
                 "with --merge-only")

    if not args.keep_stages:
        for p in stage_files:
            p.unlink(missing_ok=True)
        log(f"deleted {len(stage_files)} stage files (--keep-stages to keep)")

    log(f"finished in {(time.time() - t_start) / 60:.0f} min")
    log(f"dataset: {args.dataset_out}")
    log(f"train:   python3 train_builder.py --data {args.dataset_out} "
        f"--epochs 30")
