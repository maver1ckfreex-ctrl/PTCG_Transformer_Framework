"""Phase 1: train the deck BUILDER on replay deck sequences.

The builder reads each deck from the game's end backward; the win/lose
reward (+1/-1) is applied to the whole sequence only after it is read.

Usage:
    python3 train_builder.py --data dataset.npz
    python3 train_builder.py --data dataset.npz --epochs 50 --batch 256
"""

import argparse
import csv
import math
import pathlib
import time

import numpy as np
import torch

from builder_model import DeckDecoder, reward_weighted_loss

HERE = pathlib.Path(__file__).parent


def batches(idx, size, rng=None):
    if rng is not None:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), size):
        yield idx[i:i + size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the deck builder")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default=str(HERE / "builder_tf.pt"))
    parser.add_argument("--no-amp", action="store_true",
                        help="disable bf16 mixed precision (fp32 baseline)")
    args = parser.parse_args()

    data = np.load(args.data)
    tokens = data["build_tokens"].astype(np.int64)
    reward = data["reward"].astype(np.float32)
    vocab_size = int(data["vocab_size"])
    n, seq_len = tokens.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.set_float32_matmul_precision("high")
    print(f"dataset: {n} sequences (len {seq_len}) | vocab {vocab_size} "
          f"| {int((reward > 0).sum())} win / {int((reward < 0).sum())} loss "
          f"| device {device} | bf16 {'ON' if use_amp else 'off'}", flush=True)

    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    n_val = max(1, n // 20)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    tokens_t = torch.from_numpy(tokens)
    reward_t = torch.from_numpy(reward)

    model = DeckDecoder(vocab_size, max_len=seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(train_idx) / args.batch) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)

    def run(idx_batch, train):
        toks = tokens_t[idx_batch].to(device)
        rew = reward_t[idx_batch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(toks[:, :-1])
            loss = reward_weighted_loss(logits, toks[:, 1:], rew)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        return loss.item()

    log_path = HERE / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = [run(b, True) for b in batches(train_idx, args.batch, rng)]
        model.eval()
        with torch.no_grad():
            va = [run(b, False) for b in batches(val_idx, args.batch)]
        tr_loss, va_loss = float(np.mean(tr)), float(np.mean(va))
        marker = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(),
                        "vocab_size": vocab_size,
                        "max_len": seq_len}, args.out)
            marker = "  <- saved"
        print(f"epoch {epoch:3d}/{args.epochs} | train {tr_loss:.4f} "
              f"| val {va_loss:.4f} | lr {sched.get_last_lr()[0]:.2e} "
              f"| {time.time()-t0:.0f}s{marker}", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tr_loss:.5f}", f"{va_loss:.5f}",
                 f"{sched.get_last_lr()[0]:.2e}"])

    print(f"best val {best_val:.4f} | checkpoint: {args.out}")
