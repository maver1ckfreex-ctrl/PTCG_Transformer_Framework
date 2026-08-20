"""Daily-dataset pipeline: download -> unzip -> convert -> delete, overlapped.

Implements the streaming flow for the full episode archive (manifest csv):

  [prefetch thread]  download day N+1, N+2 zips and unzip them
  [main thread]      convert day N (decisions + builder datasets, --append)
                     then DELETE day N's folder and zip
  repeat until the manifest is exhausted.

Disk high-water mark ~= (1 + prefetch) days unzipped + prefetch zips
(~21 GB each), instead of ~1 TB for the whole archive.

Resume-safe at three levels:
  - pipeline_state.json in --work records fully finished days
  - an existing zip is reused (re-downloaded only if it fails to unzip)
  - the converters themselves skip already-converted episode files

manifest.csv inside each daily folder is ignored by construction: the
converters only glob *.json, and non-replay jsons are skipped anyway.

Usage:
    python3 pipeline.py --manifest manifest_01Aug.csv --work /local/data/inb808/ptcg_work
    python3 pipeline.py ... --start-date 2026-07-01 --end-date 2026-07-15
    python3 pipeline.py ... --convert decisions        # skip builder dataset
"""

import argparse
import csv
import json
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
import zipfile

HERE = pathlib.Path(__file__).parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"done": []}


def save_state(path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


def download(slug, zip_path, retries):
    """kaggle datasets download kaggle/<slug> -> zip_path (reuses existing)."""
    if zip_path.exists():
        log(f"  zip already present: {zip_path.name}")
        return
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", f"kaggle/{slug}",
                 "-p", str(zip_path.parent)],
                check=True, stdout=subprocess.DEVNULL)
            if zip_path.exists():
                return
            raise RuntimeError(f"kaggle finished but {zip_path.name} missing")
        except Exception as e:
            log(f"  download attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            time.sleep(30 * attempt)


def unzip(zip_path, dest):
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    n = sum(1 for _ in dest.rglob("*.json"))
    log(f"  unzipped {zip_path.name}: {n} json files")
    return n


def fetch_day(date, slug, zips_dir, raw_dir, retries):
    """Download + unzip one day; returns the unzipped folder."""
    zip_path = zips_dir / f"{slug}.zip"
    dest = raw_dir / date
    log(f"fetch {date}: downloading {slug}")
    download(slug, zip_path, retries)
    try:
        unzip(zip_path, dest)
    except zipfile.BadZipFile:
        log(f"  {zip_path.name} corrupt -> re-downloading once")
        zip_path.unlink(missing_ok=True)
        download(slug, zip_path, retries)
        unzip(zip_path, dest)
    return dest, zip_path


def convert_day(day_dir, args):
    """Run the converters with --append; raises on failure."""
    jobs = []
    if args.convert in ("both", "decisions"):
        jobs.append(("decisions", HERE / "replay_to_decisions.py",
                     args.decisions_out))
    if args.convert in ("both", "dataset"):
        jobs.append(("builder dataset", HERE / "replay_to_dataset.py",
                     args.dataset_out))
    for name, script, out in jobs:
        log(f"  converting -> {name} ({pathlib.Path(out).name})")
        cmd = [sys.executable, str(script), "--replays", str(day_dir),
               "--out", str(out), "--append"]
        if args.cards:
            cmd += ["--cards", args.cards]
        t0 = time.time()
        subprocess.run(cmd, check=True)
        log(f"  {name} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming download/convert pipeline for daily episodes")
    parser.add_argument("--manifest", required=True,
                        help="csv with date,daily_dataset_slug,... rows")
    parser.add_argument("--work", required=True,
                        help="working dir for zips/raw data/state")
    parser.add_argument("--decisions-out", default=str(HERE / "decisions.npz"))
    parser.add_argument("--dataset-out", default=str(HERE / "dataset.npz"))
    parser.add_argument("--convert", choices=["both", "decisions", "dataset"],
                        default="both")
    parser.add_argument("--cards", default=None,
                        help="EN_Card_Data.csv path (passed to converters)")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--prefetch", type=int, default=2,
                        help="days downloaded/unzipped ahead (disk!)")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--keep-raw", action="store_true",
                        help="do not delete folders/zips after converting")
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found — pip install kaggle, and put your "
                 "API token in ~/.kaggle/kaggle.json")

    work = pathlib.Path(args.work)
    zips_dir = work / "zips"
    raw_dir = work / "raw"
    zips_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = work / "pipeline_state.json"
    state = load_state(state_path)

    with open(args.manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("date")]
    days = [(r["date"], r["daily_dataset_slug"]) for r in rows]
    if args.start_date:
        days = [d for d in days if d[0] >= args.start_date]
    if args.end_date:
        days = [d for d in days if d[0] <= args.end_date]
    pending = [d for d in days if d[0] not in state["done"]]
    log(f"manifest: {len(days)} days in range, {len(pending)} to do "
        f"(state: {state_path})")
    if not pending:
        sys.exit("nothing to do")

    # ---- prefetch thread: keeps up to --prefetch days ready ---------------
    ready = queue.Queue(maxsize=max(1, args.prefetch))
    fetch_error = []

    def fetcher():
        for date, slug in pending:
            try:
                item = fetch_day(date, slug, zips_dir, raw_dir, args.retries)
                ready.put((date,) + item)     # blocks when queue is full
            except Exception as e:
                fetch_error.append((date, e))
                ready.put(None)
                return
        ready.put(None)

    threading.Thread(target=fetcher, daemon=True).start()

    done_this_run = 0
    t_start = time.time()
    try:
        while True:
            item = ready.get()
            if item is None:
                break
            date, day_dir, zip_path = item
            log(f"=== {date}: converting ===")
            convert_day(day_dir, args)
            if not args.keep_raw:
                shutil.rmtree(day_dir, ignore_errors=True)
                zip_path.unlink(missing_ok=True)
                log(f"  deleted raw folder + zip for {date}")
            state["done"].append(date)
            save_state(state_path, state)
            done_this_run += 1
            log(f"=== {date} complete ({done_this_run}/{len(pending)}, "
                f"{(time.time() - t_start) / 60:.0f} min elapsed) ===")
    except KeyboardInterrupt:
        log("interrupted — state saved, re-run the same command to resume")
    if fetch_error:
        date, e = fetch_error[0]
        sys.exit(f"FETCH FAILED for {date}: {e}\n"
                 f"re-run the same command to resume from there")
    log(f"all done: {done_this_run} days this run, "
        f"{len(state['done'])} total in {state_path}")
