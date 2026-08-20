"""SEQUENCE arm trainer, v2 -- side-by-side with train_seq.py, which is
left untouched so you can run both and compare.

Three changes from v1, each independently switchable so you can bisect
which one (if any) actually matters:

1. GRADIENT ACCUMULATION  --games-per-step (default 64)
   THE fix. Every decision in a trajectory shares one +/-1 label, so
   batching by game collapsed the reward-signal batch size:

       v1: 256 decisions = 4 games  ->   4 independent +/-1 draws/step
                                          12.45% of steps 100% one sign
       original train_v2.py: 64 shuffled decisions -> ~64 draws/step
                                          0.00% of steps one sign

   The losing branch's gradient is p/(1-p), ~1e6 near p=1. In the
   original those spikes were always counterbalanced inside the same
   update. At 4 games/step, one step in eight has nothing opposing them,
   which is the slow monotone rise. v2 keeps 4 games per forward pass
   (same memory, same speed) but accumulates ~16 of them before stepping,
   so an optimizer step again sees ~64 independent games.

   Trajectories are still read whole: batching runs them in parallel, it
   does not concatenate them. Each game keeps its own deck and its own
   causal memory chain.

2. UNLIKELIHOOD CLAMP  --p-max (default 0.95)
   Caps the losing branch's gradient at p_max/(1-p_max) = 19 instead of
   ~1e6. --p-max 0.999999 reproduces v1 exactly.

3. LR SCHEDULE  --warmup-steps (500) --lr-schedule cosine
   v1 ran a constant 3e-4 for ~50k steps with no warmup and no decay.
   --lr-schedule none reproduces v1.

Reproduce v1 exactly (sanity check that nothing else drifted):
    --games-per-step 4 --p-max 0.999999 --lr-schedule none --warmup-steps 0

Usage:
    python3 train_seq_v2.py --data traj.npz --out seq_v2.pt --epochs 4
"""

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

import numpy as np
import torch

from player_vocab import MAX_SEQ
from seq_model import SeqPlayer, MAX_TRAJ
from train_seq import Trajectories          # identical data path as v1

HERE = pathlib.Path(__file__).parent


def player_loss_v2(scores, opt_mask, chosen_mask, reward, p_max):
    """Same loss as common/encoder.player_loss, with the unlikelihood
    branch's probability clamped so its gradient p/(1-p) stays finite."""
    logp = torch.log_softmax(scores, dim=-1)
    logp = torch.where(opt_mask, logp, torch.zeros_like(logp))
    chosen = chosen_mask.float()
    n_chosen = chosen.sum(-1).clamp(min=1.0)
    chosen_ll = (logp * chosen).sum(-1) / n_chosen

    pos_loss = -chosen_ll
    p = chosen_ll.exp().clamp(max=p_max)
    neg_loss = -torch.log1p(-p)
    return torch.where(reward > 0, pos_loss, neg_loss)


def lr_at(step, total, base_lr, warmup, kind):
    if kind == "none":
        return base_lr
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return base_lr
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def main():
    ap = argparse.ArgumentParser(description="sequence arm, v2")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "sequence_v2.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--decisions-per-batch", type=int, default=256,
                    help="decisions per FORWARD pass (memory knob)")
    ap.add_argument("--games-per-step", type=int, default=64,
                    help="games per OPTIMIZER step; micro-batches are "
                         "accumulated until this many games are covered. "
                         "Set to 4 to reproduce v1.")
    ap.add_argument("--p-max", type=float, default=0.95,
                    help="clamp on p in the unlikelihood branch; "
                         "0.999999 reproduces v1")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-schedule", choices=["cosine", "none"],
                    default="cosine")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--mem-layers", type=int, default=2)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epoch-ckpt-dir", default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.set_float32_matmul_precision("high")

    tr = Trajectories(args.data)
    print(f"[v2] {tr.n} trajectories | {tr.n_dec} decisions "
          f"| decisions/traj mean {tr.traj_len.mean():.1f} "
          f"max {tr.traj_len.max()} | vocab {tr.vocab_size} "
          f"| device {device} | bf16 {'on' if use_amp else 'off'}", flush=True)
    print(f"[v2] games/optimizer-step {args.games_per_step} "
          f"| p_max {args.p_max} | lr {args.lr} {args.lr_schedule} "
          f"warmup {args.warmup_steps}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(tr.n)
    n_val = max(1, int(tr.n * args.val_frac))
    val_games, train_games = perm[:n_val], perm[n_val:]
    print(f"[v2] split by game: {len(train_games)} train "
          f"| {len(val_games)} val", flush=True)

    model = SeqPlayer(tr.vocab_size, max_len=MAX_SEQ,
                      mem_layers=args.mem_layers, max_traj=MAX_TRAJ)
    if args.init:
        sys.path.insert(0, str(HERE.parent / "tools"))
        from warm_start import remap_r2
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck.get("arch") == "sequence":
            model.load_state_dict(ck["model"])
            print(f"[v2] resumed from {args.init}", flush=True)
        else:
            torch.nn.init.zeros_(model.mem_out.weight)
            torch.nn.init.zeros_(model.mem_out.bias)
            miss, unexp = model.load_state_dict(remap_r2(ck["model"]),
                                                strict=False)
            if unexp:
                raise SystemExit(f"unexpected keys: {unexp}")
            print(f"[v2] warm-started from R2 {args.init} "
                  f"({len(miss)} fresh tensors)", flush=True)
    model = model.to(device)
    print(f"[v2] parameters: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def game_batches(games, shuffle):
        games = np.asarray(games)
        if shuffle:
            games = rng.permutation(games)
        games = games[np.argsort(tr.traj_len[games], kind="stable")]
        out, i = [], 0
        while i < len(games):
            t = int(tr.traj_len[games[i]])
            g = max(1, args.decisions_per_batch // max(t, 1))
            out.append(games[i:i + g])
            i += g
        if shuffle:
            rng.shuffle(out)
        return out

    def accum_groups(batches):
        """Group forward-pass micro-batches into optimizer steps."""
        groups, cur, n = [], [], 0
        for b in batches:
            cur.append(b)
            n += len(b)
            if n >= args.games_per_step:
                groups.append(cur)
                cur, n = [], 0
        if cur:
            groups.append(cur)
        return groups

    def forward(b):
        (tokens, last_idx, pos, omask, cmask,
         deck, dmask, rew) = tr.batch(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            scores = model(tokens, last_idx, pos, omask, deck, dmask)
        g, t, o = scores.shape
        per = player_loss_v2(scores.reshape(g * t, o).float(),
                             omask.reshape(g * t, o),
                             cmask.reshape(g * t, o),
                             rew.unsqueeze(1).expand(g, t).reshape(g * t),
                             args.p_max)
        fd = dmask.reshape(g * t).float()
        return (per * fd).sum(), fd.sum()

    total_steps = args.epochs * max(1, len(accum_groups(
        game_batches(train_games, False))))
    step = 0

    def run_epoch(games, train):
        nonlocal step
        model.train(train)
        tot = cnt = 0.0
        t0 = time.time()
        if not train:
            for b in game_batches(games, False):
                s, n = forward(b)
                tot += float(s.detach())
                cnt += float(n)
            return tot / max(cnt, 1)

        groups = accum_groups(game_batches(games, True))
        for gi, group in enumerate(groups, 1):
            # exact denominator for this optimizer step, known up front
            n_group = float(sum(int(tr.traj_len[b].sum()) for b in group))
            for lr_g in opt.param_groups:
                lr_g["lr"] = lr_at(step, total_steps, args.lr,
                                   args.warmup_steps, args.lr_schedule)
            opt.zero_grad(set_to_none=True)
            for b in group:
                s, n = forward(b)
                (s / n_group).backward()
                tot += float(s.detach())
                cnt += float(n)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if gi % 20 == 0:
                print(f"    step {gi}/{len(groups)} "
                      f"({sum(len(b) for b in group)} games) "
                      f"loss {tot/cnt:.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} "
                      f"({cnt/(time.time()-t0):.0f} dec/s)", flush=True)
        return tot / max(cnt, 1)

    def payload(va):
        return {"model": model.state_dict(), "vocab_size": tr.vocab_size,
                "max_len": MAX_SEQ, "mem_layers": args.mem_layers,
                "max_traj": MAX_TRAJ, "arch": "sequence", "val_loss": va}

    ep_dir = pathlib.Path(args.epoch_ckpt_dir) if args.epoch_ckpt_dir else None
    if ep_dir:
        ep_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        trl = run_epoch(train_games, True)
        with torch.no_grad():
            va = run_epoch(val_games, False)
        print(f"[v2] epoch {ep}: train {trl:.4f} | val {va:.4f}", flush=True)
        if va < best:
            best = va
            torch.save(payload(va), args.out)
            print(f"[v2] saved {args.out} (val {va:.4f})", flush=True)
        if ep_dir:
            final = ep_dir / f"epoch_{ep:03d}.pt"
            tmp = ep_dir / f".epoch_{ep:03d}.pt.tmp"
            torch.save(payload(va), tmp)
            tmp.rename(final)
            print(f"[v2] snapshot {final}", flush=True)
    if ep_dir:
        (ep_dir / "TRAINING_DONE").write_text(
            f"epochs={args.epochs} best_val={best:.6f}\n")
    print(f"[v2] done. best val {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
