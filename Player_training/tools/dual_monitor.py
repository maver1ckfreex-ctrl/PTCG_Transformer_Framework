"""Combined progress line for dual-GPU mode.

Each trainer writes its own log. This tails both and prints ONE line
whenever either side advances:

    GPU0 v3a | step 120/709 ce 1.7421 lr 3.0e-04 6912/s  ||  GPU1 v3b | step 98/1418 ce 1.9002 ...

Whichever trainer updates first triggers the new line; the other side is
shown at whatever its latest status was. Neither waits for the other -- the
monitor only reads files, so a stalled or finished trainer never blocks the
one still running.

    python3 tools/dual_monitor.py <log0> <label0> <log1> <label1> [--poll 5]
"""

import argparse
import os
import pathlib
import re
import sys
import time

# progress lines the trainers emit
STEP = re.compile(r"^\s*step\s+(\d+)/(\d+)\s+(.*?)\s*\(([\d.]+) dec/s\)")
EPOCH = re.compile(r"^\[(\w+)\]\s+epoch\s+(\d+):\s*(.*)$")
DONE = re.compile(r"^\[(\w+)\]\s+done\.")


def compress(metrics):
    """Shorten the metric blob so two fit on one terminal line."""
    m = metrics
    m = m.replace("action_ce", "ce").replace("winner_ce", "wce")
    m = m.replace("outcome_bce", "obce").replace("outcome", "out")
    m = re.sub(r"\s+", " ", m).strip()
    return m


def tail_status(path, prev):
    """-> (status string, changed?) from the last progress line in a log."""
    p = pathlib.Path(path)
    if not p.exists():
        return prev or "waiting...", False
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except Exception:
        return prev or "waiting...", False

    status = None
    for ln in reversed(lines):
        d = DONE.match(ln)
        if d:
            status = "DONE"
            break
        e = EPOCH.match(ln)
        if e:
            status = f"epoch {e.group(2)} {compress(e.group(3))}"
            break
        s = STEP.match(ln)
        if s:
            status = (f"step {s.group(1)}/{s.group(2)} "
                      f"{compress(s.group(3))} {float(s.group(4)):.0f}/s")
            break
    if status is None:
        status = prev or "starting..."
    return status, status != prev


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="dual-GPU progress monitor")
    ap.add_argument("log0")
    ap.add_argument("label0")
    ap.add_argument("log1")
    ap.add_argument("label1")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--stop-file", default=None,
                    help="exit once this file exists and both sides are DONE")
    args = ap.parse_args()

    s0 = s1 = None
    t0 = time.time()
    while True:
        n0, c0 = tail_status(args.log0, s0)
        n1, c1 = tail_status(args.log1, s1)
        s0, s1 = n0, n1
        if c0 or c1:
            el = int(time.time() - t0)
            print(f"[{el//3600:02d}:{el%3600//60:02d}:{el%60:02d}] "
                  f"GPU0 {args.label0} | {s0}"
                  f"   ||   GPU1 {args.label1} | {s1}", flush=True)
        if s0 == "DONE" and s1 == "DONE":
            print("[monitor] both trainers finished", flush=True)
            break
        try:
            time.sleep(args.poll)
        except KeyboardInterrupt:
            sys.exit(0)
