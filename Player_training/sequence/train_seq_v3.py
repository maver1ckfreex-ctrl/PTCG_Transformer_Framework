"""v3 trainer -- dense per-step action prediction, outcome only at the end.

v1/v2 stamped the game's +/-1 onto all ~57 decisions and flipped the loss
on it. That gives 1 bit of supervision per game, and its optimum is the
binary entropy of the winning FRACTION of the data: with a 50/50 corpus
the floor is H(0.5) = 0.6931 and nothing below it can be reached. Measured
the real data:

    +/-1 flip  : 1 bit per game,   floor 0.6931, observed run sat at 0.7526
    v3         : 150 bits per game, chance ln(n) = 1.8240, floor 0

v3 uses two losses:

  ACTION  (dense, one per decision)
      cross-entropy on the option that was ACTUALLY taken, conditioned on
      the deck and every earlier decision of that game. No outcome
      weighting -- at step t the target is simply the action played at
      step t. Starts at ln(n_options) ~ 1.82 and has real room to fall.

  OUTCOME (sparse, one per trajectory)
      win/lose predicted ONCE, from the level-2 position that has read
      every decision in the game. Never applied to intermediate steps.

    loss = action_ce + --outcome-weight * outcome_bce

--loser-weight controls how much a losing player's actions are imitated:
    1.0  learn from both sides equally (default; pure "what was played")
    0.0  winners only (classic winner behaviour cloning)
    0.5  in between
This is how the outcome informs the ACTION loss -- by weighting whole
trajectories, never by flipping the sign of individual steps.

Usage:
    python3 train_seq_v3.py --data traj.npz --out seq_v3.pt --epochs 8
    python3 train_seq_v3.py --data traj.npz --out seq_v3.pt --loser-weight 0
"""

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

import numpy as np
import torch
import torch.nn.functional as F

from player_vocab import MAX_SEQ
from seq_model_v3 import SeqPlayerV3, MAX_TRAJ
from train_seq import Trajectories          # identical data path as v1/v2

HERE = pathlib.Path(__file__).parent


def action_ce(scores, opt_mask, chosen_mask):
    """Cross-entropy on the option(s) actually taken. (B,) per decision.

    Multi-select decisions get the mean log-prob over everything picked,
    which is the same reduction v1 used -- only the outcome flip is gone.
    """
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
    ap = argparse.ArgumentParser(description="sequence arm, v3")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "sequence_v3.pt"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--decisions-per-batch", type=int, default=256)
    ap.add_argument("--games-per-step", type=int, default=64)
    ap.add_argument("--outcome-weight", type=float, default=0.1,
                    help="weight on the end-of-trajectory win/lose loss; "
                         "0 disables the outcome head entirely")
    ap.add_argument("--loser-weight", type=float, default=1.0,
                    help="how much a LOSING player's actions are imitated. "
                         "1.0 = both sides equally, 0.0 = winners only")
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
    print(f"[v3] {tr.n} trajectories | {tr.n_dec} decisions "
          f"| decisions/traj mean {tr.traj_len.mean():.1f} "
          f"| vocab {tr.vocab_size} | device {device} "
          f"| bf16 {'on' if use_amp else 'off'}", flush=True)
    print(f"[v3] action CE chance level = ln(n_options) = "
          f"{np.log(n_opt).mean():.4f}  (perfect = 0)", flush=True)
    print(f"[v3] outcome-weight {args.outcome_weight} "
          f"| loser-weight {args.loser_weight} "
          f"| games/step {args.games_per_step} "
          f"| lr {args.lr} {args.lr_schedule}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(tr.n)
    n_val = max(1, int(tr.n * args.val_frac))
    val_games, train_games = perm[:n_val], perm[n_val:]
    print(f"[v3] split by game: {len(train_games)} train "
          f"| {len(val_games)} val", flush=True)

    model = SeqPlayerV3(tr.vocab_size, max_len=MAX_SEQ,
                        mem_layers=args.mem_layers, max_traj=MAX_TRAJ)
    if args.init:
        sys.path.insert(0, str(HERE.parent / "tools"))
        from warm_start import remap_r2
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck.get("arch") == "sequence_v3":
            model.load_state_dict(ck["model"])
            print(f"[v3] resumed from {args.init}", flush=True)
        else:
            torch.nn.init.zeros_(model.mem_out.weight)
            torch.nn.init.zeros_(model.mem_out.bias)
            miss, unexp = model.load_state_dict(remap_r2(ck["model"]),
                                                strict=False)
            if unexp:
                raise SystemExit(f"unexpected keys: {unexp}")
            print(f"[v3] warm-started from R2 {args.init} "
                  f"({len(miss)} fresh tensors)", flush=True)
    model = model.to(device)
    print(f"[v3] parameters: "
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
        """-> (weighted action-loss sum, decision count,
                outcome-loss sum, game count, raw CE sum)"""
        (tokens, last_idx, pos, omask, cmask,
         deck, dmask, rew) = tr.batch(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            scores, outcome = model(tokens, last_idx, pos, omask, deck, dmask)
        g, t, o = scores.shape

        ce = action_ce(scores.reshape(g * t, o).float(),
                       omask.reshape(g * t, o),
                       cmask.reshape(g * t, o)).view(g, t)
        fd = dmask.float()

        # whole-trajectory weighting -- never a per-step sign flip
        w = torch.where(rew > 0, torch.ones_like(rew),
                        torch.full_like(rew, args.loser_weight))
        act_sum = (ce * fd * w.unsqueeze(1)).sum()
        raw_sum = (ce * fd).sum()
        n_dec = fd.sum()

        win = (rew > 0).float()
        out_sum = F.binary_cross_entropy_with_logits(
            outcome.float(), win, reduction="sum")
        return act_sum, n_dec, out_sum, torch.tensor(float(g)), raw_sum

    total_steps = args.epochs * max(1, len(accum_groups(
        game_batches(train_games, False))))
    step = 0

    def run_epoch(games, train):
        nonlocal step
        model.train(train)
        a_tot = d_tot = o_tot = g_tot = raw_tot = 0.0
        t0 = time.time()

        if not train:
            for b in game_batches(games, False):
                a, n, o, ng, raw = forward(b)
                a_tot += float(a); d_tot += float(n)
                o_tot += float(o); g_tot += float(ng); raw_tot += float(raw)
            return raw_tot / max(d_tot, 1), o_tot / max(g_tot, 1)

        groups = accum_groups(game_batches(games, True))
        for gi, group in enumerate(groups, 1):
            n_group = float(sum(int(tr.traj_len[b].sum()) for b in group))
            g_group = float(sum(len(b) for b in group))
            for lg in opt.param_groups:
                lg["lr"] = lr_at(step, total_steps, args.lr,
                                 args.warmup_steps, args.lr_schedule)
            opt.zero_grad(set_to_none=True)
            for b in group:
                a, n, o, ng, raw = forward(b)
                loss = a / n_group + args.outcome_weight * o / g_group
                loss.backward()
                a_tot += float(a.detach()); d_tot += float(n)
                o_tot += float(o.detach()); g_tot += float(ng)
                raw_tot += float(raw.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if gi % 20 == 0:
                print(f"    step {gi}/{len(groups)} "
                      f"action_ce {raw_tot/max(d_tot,1):.4f} "
                      f"outcome_bce {o_tot/max(g_tot,1):.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} "
                      f"({d_tot/(time.time()-t0):.0f} dec/s)", flush=True)
        return raw_tot / max(d_tot, 1), o_tot / max(g_tot, 1)

    def payload(va):
        return {"model": model.state_dict(), "vocab_size": tr.vocab_size,
                "max_len": MAX_SEQ, "mem_layers": args.mem_layers,
                "max_traj": MAX_TRAJ, "arch": "sequence_v3", "val_loss": va}

    ep_dir = pathlib.Path(args.epoch_ckpt_dir) if args.epoch_ckpt_dir else None
    if ep_dir:
        ep_dir.mkdir(parents=True, exist_ok=True)

    chance = float(np.log(n_opt).mean())
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_a, tr_o = run_epoch(train_games, True)
        with torch.no_grad():
            va_a, va_o = run_epoch(val_games, False)
        print(f"[v3] epoch {ep}: train action_ce {tr_a:.4f} "
              f"outcome {tr_o:.4f} | val action_ce {va_a:.4f} "
              f"outcome {va_o:.4f} | chance {chance:.4f}", flush=True)
        if va_a < best:
            best = va_a
            torch.save(payload(va_a), args.out)
            print(f"[v3] saved {args.out} (val action_ce {va_a:.4f})",
                  flush=True)
        if ep_dir:
            final = ep_dir / f"epoch_{ep:03d}.pt"
            tmp = ep_dir / f".epoch_{ep:03d}.pt.tmp"
            torch.save(payload(va_a), tmp)
            tmp.rename(final)
            print(f"[v3] snapshot {final}", flush=True)
    if ep_dir:
        (ep_dir / "TRAINING_DONE").write_text(
            f"epochs={args.epochs} best_val_action_ce={best:.6f}\n")
    print(f"[v3] done. best val action_ce {best:.4f} "
          f"(chance {chance:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
