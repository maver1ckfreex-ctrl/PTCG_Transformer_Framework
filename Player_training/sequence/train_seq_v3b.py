"""v3b trainer -- outcome token at the trajectory head, and the action loss
is PUNISHED on losing trajectories.

Two mechanisms, both as specified:

1. CONDITIONING. The true outcome sits at level-2 position 0, ahead of the
   deck, so every decision is scored knowing whether this game was won or
   lost. At play time the token is pinned to WIN.

2. SIGN FLIP, per decision, dense.
       winner : -log p(action)        raise what the winner played
       loser  : -log(1 - p(action))   push down what the loser played
   Unlike v1 this is applied to the ACTION cross-entropy at every step
   rather than to a single trajectory-level scalar, so the supervision is
   still ~150 bits per game instead of 1.

   The push-down branch has gradient p/(1-p), which reaches ~1e6 at p=1
   and is what destabilised v1. --p-max clamps it: 0.95 caps the gradient
   at 19. Raise it toward 1.0 to reproduce v1's unbounded behaviour.

There is no outcome-prediction head: the outcome is an input here, so
predicting it would be degenerate.

Usage:
    python3 train_seq_v3b.py --data traj.npz --out seq_v3b.pt --epochs 8
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
from seq_model_v3b import SeqPlayerV3B, MAX_TRAJ
from train_seq import Trajectories          # identical data path

HERE = pathlib.Path(__file__).parent


def action_ce(scores, opt_mask, chosen_mask):
    logp = torch.log_softmax(scores, dim=-1)
    logp = torch.where(opt_mask, logp, torch.zeros_like(logp))
    chosen = chosen_mask.float()
    n_chosen = chosen.sum(-1).clamp(min=1.0)
    return -(logp * chosen).sum(-1) / n_chosen


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
    ap = argparse.ArgumentParser(description="sequence arm, v3b")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "sequence_v3b.pt"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--decisions-per-batch", type=int, default=256)
    ap.add_argument("--games-per-step", type=int, default=64)
    ap.add_argument("--p-max", type=float, default=0.95,
                    help="clamp on p in the push-down branch; caps its "
                         "gradient at p/(1-p). 0.95 -> 19")
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
    n_opt = np.diff(tr.pos_off).astype(np.float64)
    frac_win = float((tr.reward > 0).mean())
    print(f"[v3b] {tr.n} trajectories ({100*frac_win:.1f}% winning) "
          f"| {tr.n_dec} decisions | vocab {tr.vocab_size} "
          f"| device {device} | bf16 {'on' if use_amp else 'off'}", flush=True)
    print(f"[v3b] winner-branch chance level = ln(n_options) = "
          f"{np.log(n_opt).mean():.4f}", flush=True)
    print(f"[v3b] p_max {args.p_max} (push-down gradient capped at "
          f"{args.p_max/(1-args.p_max):.0f}) | games/step "
          f"{args.games_per_step} | lr {args.lr} {args.lr_schedule}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(tr.n)
    n_val = max(1, int(tr.n * args.val_frac))
    val_games, train_games = perm[:n_val], perm[n_val:]
    print(f"[v3b] split by game: {len(train_games)} train "
          f"| {len(val_games)} val", flush=True)

    model = SeqPlayerV3B(tr.vocab_size, max_len=MAX_SEQ,
                         mem_layers=args.mem_layers, max_traj=MAX_TRAJ)
    if args.init:
        sys.path.insert(0, str(HERE.parent / "tools"))
        from warm_start import remap_r2
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck.get("arch") == "sequence_v3b":
            model.load_state_dict(ck["model"])
            print(f"[v3b] resumed from {args.init}", flush=True)
        else:
            torch.nn.init.zeros_(model.mem_out.weight)
            torch.nn.init.zeros_(model.mem_out.bias)
            miss, unexp = model.load_state_dict(remap_r2(ck["model"]),
                                                strict=False)
            if unexp:
                raise SystemExit(f"unexpected keys: {unexp}")
            print(f"[v3b] warm-started from R2 {args.init} "
                  f"({len(miss)} fresh tensors)", flush=True)
    model = model.to(device)
    print(f"[v3b] parameters: "
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
        """-> (loss sum, decisions, winner-CE sum, winner decisions)"""
        (tokens, last_idx, pos, omask, cmask,
         deck, dmask, rew) = tr.batch(b, device)
        win = (rew > 0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            scores = model(tokens, last_idx, pos, omask, deck, dmask,
                           win.long())
        g, t, o = scores.shape
        ce = action_ce(scores.reshape(g * t, o).float(),
                       omask.reshape(g * t, o),
                       cmask.reshape(g * t, o)).view(g, t)

        # winner: raise p     loser: push p down, gradient bounded by p_max
        p = (-ce).exp().clamp(max=args.p_max)
        push_down = -torch.log1p(-p)
        per = torch.where(win.unsqueeze(1), ce, push_down)

        fd = dmask.float()
        wmask = fd * win.unsqueeze(1).float()
        return ((per * fd).sum(), fd.sum(),
                (ce * wmask).sum(), wmask.sum())

    total_steps = args.epochs * max(1, len(accum_groups(
        game_batches(train_games, False))))
    step = 0

    def run_epoch(games, train):
        nonlocal step
        model.train(train)
        l_tot = d_tot = w_tot = wd_tot = 0.0
        t0 = time.time()

        if not train:
            for b in game_batches(games, False):
                l, n, w, wn = forward(b)
                l_tot += float(l); d_tot += float(n)
                w_tot += float(w); wd_tot += float(wn)
            return (l_tot / max(d_tot, 1),
                    w_tot / wd_tot if wd_tot > 0 else float("nan"))

        groups = accum_groups(game_batches(games, True))
        for gi, group in enumerate(groups, 1):
            n_group = float(sum(int(tr.traj_len[b].sum()) for b in group))
            for lg in opt.param_groups:
                lg["lr"] = lr_at(step, total_steps, args.lr,
                                 args.warmup_steps, args.lr_schedule)
            opt.zero_grad(set_to_none=True)
            for b in group:
                l, n, w, wn = forward(b)
                (l / n_group).backward()
                l_tot += float(l.detach()); d_tot += float(n)
                w_tot += float(w.detach()); wd_tot += float(wn)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if gi % 20 == 0:
                print(f"    step {gi}/{len(groups)} "
                      f"loss {l_tot/max(d_tot,1):.4f} "
                      f"winner_ce {w_tot/max(wd_tot,1):.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} "
                      f"({d_tot/(time.time()-t0):.0f} dec/s)", flush=True)
        return (l_tot / max(d_tot, 1),
                w_tot / wd_tot if wd_tot > 0 else float("nan"))

    def payload(va):
        return {"model": model.state_dict(), "vocab_size": tr.vocab_size,
                "max_len": MAX_SEQ, "mem_layers": args.mem_layers,
                "max_traj": MAX_TRAJ, "arch": "sequence_v3b", "val_loss": va}

    ep_dir = pathlib.Path(args.epoch_ckpt_dir) if args.epoch_ckpt_dir else None
    if ep_dir:
        ep_dir.mkdir(parents=True, exist_ok=True)

    chance = float(np.log(n_opt).mean())
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_l, tr_w = run_epoch(train_games, True)
        with torch.no_grad():
            va_l, va_w = run_epoch(val_games, False)
        print(f"[v3b] epoch {ep}: train loss {tr_l:.4f} winner_ce {tr_w:.4f}"
              f" | val loss {va_l:.4f} winner_ce {va_w:.4f}"
              f" | chance {chance:.4f}", flush=True)
        # Selected on the WINNER branch -- that is the conditional play
        # actually uses. If a val split somehow contains no winning
        # trajectory the winner CE is undefined, so fall back to total
        # loss rather than letting a degenerate 0.0 win every comparison.
        sel = va_w if va_w == va_w else va_l
        if sel < best:
            best = sel
            torch.save(payload(sel), args.out)
            print(f"[v3b] saved {args.out} (val {sel:.4f})", flush=True)
        if ep_dir:
            final = ep_dir / f"epoch_{ep:03d}.pt"
            tmp = ep_dir / f".epoch_{ep:03d}.pt.tmp"
            torch.save(payload(sel), tmp)
            tmp.rename(final)
            print(f"[v3b] snapshot {final}", flush=True)
    if ep_dir:
        (ep_dir / "TRAINING_DONE").write_text(
            f"epochs={args.epochs} best_val_winner_ce={best:.6f}\n")
    print(f"[v3b] done. best val winner_ce {best:.4f} "
          f"(chance {chance:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
