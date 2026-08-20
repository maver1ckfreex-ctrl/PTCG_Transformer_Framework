"""Prove the warm start is exact: R2 and the warm-started trial models must
produce IDENTICAL option scores on real observations.

If this fails, the remap is wrong and every downstream number is garbage.

Usage:
    python3 tools/verify_r2.py --r2 ../submission_r2_t07_torch/player.pt \
        --replay /path/to/some_kaggle_replay.json
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "sequence"))

import torch

from player_vocab import PlayerVocab, tokenize_decision
from warm_start import warm_start


def old_model(r2_path):
    """The ORIGINAL PlayerDecoder, unmodified, as R2 shipped it."""
    sys.path.insert(0, str(ROOT / "common"))
    from player_model import PlayerDecoder
    ck = torch.load(r2_path, map_location="cpu", weights_only=False)
    m = PlayerDecoder(ck["vocab_size"], max_len=ck["max_len"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def collect(replay, pv, limit=60):
    ep = json.load(open(replay))
    st = ep["steps"]
    out = []
    for i in range(len(st) - 1):
        for ai, e in enumerate(st[i]):
            if e.get("status") != "ACTIVE":
                continue
            o = e.get("observation") or {}
            cur, sel = o.get("current"), o.get("select")
            if not cur or not sel:
                continue
            t = tokenize_decision(pv, cur, sel)
            if t is not None:
                out.append(t)
            if len(out) >= limit:
                return out
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", required=True)
    ap.add_argument("--replay", required=True)
    args = ap.parse_args()

    pv = PlayerVocab()
    samples = collect(args.replay, pv)
    print(f"comparing on {len(samples)} real decisions\n")

    ref = old_model(args.r2)
    base, _, nb, fb, _ = warm_start(args.r2, "baseline")
    seq, _, ns, fs, _ = warm_start(args.r2, "sequence")
    base.eval()
    seq.eval()
    print(f"baseline: carried {nb} tensors, fresh {len(fb)}")
    print(f"sequence: carried {ns} tensors, fresh {len(fs)} -> {fs}\n")

    from seq_model import SeqPlayerRunner
    runner = SeqPlayerRunner(seq)
    runner.reset()

    d_base = d_seq = 0.0
    for toks, opt_pos in samples:
        tk = torch.tensor([toks], dtype=torch.int64)
        pos = torch.tensor([opt_pos], dtype=torch.int64)
        om = torch.ones_like(pos, dtype=torch.bool)
        with torch.no_grad():
            pad = torch.zeros_like(tk, dtype=torch.bool)
            r = ref(tk, pad, pos, om)[0]
            b = base(tk, pos, om)[0]
        s = torch.tensor(runner.scores(toks, opt_pos))
        d_base = max(d_base, (r - b).abs().max().item())
        d_seq = max(d_seq, (r - s).abs().max().item())

    print(f"max |R2 - baseline arm| = {d_base:.3e}")
    print(f"max |R2 - sequence arm| = {d_seq:.3e}   (mem_out zero-init)")
    ok = d_base < 1e-4 and d_seq < 1e-4
    print("\n" + ("PASS - warm start is exact, both arms start AS R2"
                  if ok else "FAIL - remap is wrong"))
    sys.exit(0 if ok else 1)
