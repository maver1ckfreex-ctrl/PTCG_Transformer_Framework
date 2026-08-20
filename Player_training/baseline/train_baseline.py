"""BASELINE arm trainer: decisions broken to pieces and shuffled.

Reads the SAME trajectories.npz the sequence arm reads, then throws the
grouping away -- every decision becomes an independent sample, shuffled
across all games, exactly like the current design.

Train/val is split BY GAME in both arms (a decision-level split leaks,
since consecutive states inside one game are near-identical; that would
break the metric, not the design under test).

Usage:
    python3 train_baseline.py --data ../trajectories.npz --out baseline.pt
"""

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

import numpy as np
import torch

from baseline_model import BaselinePlayer
from encoder import player_loss
from player_vocab import MAX_SEQ

HERE = pathlib.Path(__file__).parent


class Decisions:
    """Trajectory npz -> flat, order-free decisions."""

    def __init__(self, path):
        d = np.load(path)
        self.tok_flat = d["tok_flat"]
        self.tok_off = d["tok_off"]
        self.pos_flat = d["pos_flat"]
        self.pos_off = d["pos_off"]
        self.chosen_flat = d["chosen_flat"]
        self.chosen_off = d["chosen_off"]
        self.traj_off = d["traj_off"]
        self.traj_reward = d["reward"]
        self.vocab_size = int(d["vocab_size"])
        self.n_traj = len(self.traj_reward)
        self.n = len(self.tok_off) - 1

        # every decision inherits its game's +/-1 outcome
        self.reward = np.zeros(self.n, dtype=np.float32)
        self.game_of = np.zeros(self.n, dtype=np.int64)
        for g in range(self.n_traj):
            a, b = self.traj_off[g], self.traj_off[g + 1]
            self.reward[a:b] = self.traj_reward[g]
            self.game_of[a:b] = g

    def lengths(self):
        return self.tok_off[1:] - self.tok_off[:-1]

    def decisions_of_games(self, games):
        keep = np.zeros(self.n, dtype=bool)
        for g in games:
            keep[self.traj_off[g]:self.traj_off[g + 1]] = True
        return np.flatnonzero(keep)

    def batch(self, idx, device):
        seqs = [self.tok_flat[self.tok_off[i]:self.tok_off[i + 1]] for i in idx]
        opts = [self.pos_flat[self.pos_off[i]:self.pos_off[i + 1]] for i in idx]
        chos = [self.chosen_flat[self.chosen_off[i]:self.chosen_off[i + 1]]
                for i in idx]
        b = len(idx)
        max_l = max(len(s) for s in seqs)
        max_o = max(len(o) for o in opts)
        tokens = np.zeros((b, max_l), dtype=np.int64)
        opt_pos = np.zeros((b, max_o), dtype=np.int64)
        opt_mask = np.zeros((b, max_o), dtype=bool)
        chosen_mask = np.zeros((b, max_o), dtype=bool)
        for j, (s, o, c) in enumerate(zip(seqs, opts, chos)):
            tokens[j, :len(s)] = s          # RIGHT padded (causal-safe)
            opt_pos[j, :len(o)] = o
            opt_mask[j, :len(o)] = True
            chosen_mask[j, c] = True
        t = lambda a, dt: torch.from_numpy(a.astype(dt)).to(device)
        return (t(tokens, np.int64), t(opt_pos, np.int64), t(opt_mask, bool),
                t(chosen_mask, bool),
                torch.from_numpy(self.reward[idx]).to(device))


def main():
    ap = argparse.ArgumentParser(description="baseline (decision-level) arm")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "baseline.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--decisions-per-batch", type=int, default=256,
                    help="MUST match the sequence arm for a fair comparison")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--init", default=None,
                    help="checkpoint to warm-start from: an R2-era player.pt "
                         "(remapped) or a previous baseline.pt")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.set_float32_matmul_precision("high")

    dec = Decisions(args.data)
    print(f"[baseline] {dec.n_traj} games | {dec.n} decisions "
          f"| vocab {dec.vocab_size} | device {device} "
          f"| bf16 {'on' if use_amp else 'off'}", flush=True)

    rng = np.random.default_rng(args.seed)
    gperm = rng.permutation(dec.n_traj)
    n_val = max(1, int(dec.n_traj * args.val_frac))
    val_games, train_games = gperm[:n_val], gperm[n_val:]
    train_idx = dec.decisions_of_games(train_games)
    val_idx = dec.decisions_of_games(val_games)
    print(f"[baseline] split by game: {len(train_games)} train games "
          f"({len(train_idx)} decisions) | {len(val_games)} val games "
          f"({len(val_idx)} decisions)", flush=True)

    model = BaselinePlayer(dec.vocab_size, max_len=MAX_SEQ)
    if args.init:
        sys.path.insert(0, str(HERE.parent / "tools"))
        from warm_start import remap_r2
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck.get("arch") == "baseline":
            model.load_state_dict(ck["model"])
            print(f"[baseline] resumed from {args.init}", flush=True)
        else:
            miss, unexp = model.load_state_dict(remap_r2(ck["model"]),
                                                strict=False)
            if unexp:
                raise SystemExit(f"unexpected keys in {args.init}: {unexp}")
            print(f"[baseline] warm-started from R2 {args.init} "
                  f"({len(miss)} fresh tensors)", flush=True)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[baseline] parameters: {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    lens = dec.lengths()

    def length_batches(idx, size, shuffle):
        """Group similar lengths so padding stays small, then shuffle the
        BATCH order (samples still mix across epochs)."""
        idx = np.asarray(idx)
        if shuffle:
            idx = rng.permutation(idx)
        idx = idx[np.argsort(lens[idx], kind="stable")]
        chunks = [idx[i:i + size] for i in range(0, len(idx), size)]
        if shuffle:
            rng.shuffle(chunks)
        return chunks

    def run_epoch(idx, train):
        model.train(train)
        tot = cnt = 0.0
        chunks = length_batches(idx, args.decisions_per_batch, train)
        t0 = time.time()
        for si, b in enumerate(chunks, 1):
            tokens, pos, omask, cmask, rew = dec.batch(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                scores = model(tokens, pos, omask)
            loss = player_loss(scores.float(), omask, cmask, rew).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += float(loss.detach()) * len(b)
            cnt += len(b)
            if train and si % 200 == 0:
                print(f"    step {si}/{len(chunks)} loss {tot/cnt:.4f} "
                      f"({cnt/(time.time()-t0):.0f} dec/s)", flush=True)
        return tot / max(cnt, 1)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_idx, True)
        with torch.no_grad():
            va = run_epoch(val_idx, False)
        print(f"[baseline] epoch {ep}: train {tr:.4f} | val {va:.4f}",
              flush=True)
        if va < best:
            best = va
            torch.save({"model": model.state_dict(),
                        "vocab_size": dec.vocab_size,
                        "max_len": MAX_SEQ,
                        "arch": "baseline",
                        "val_loss": va}, args.out)
            print(f"[baseline] saved {args.out} (val {va:.4f})", flush=True)
    print(f"[baseline] done. best val {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
