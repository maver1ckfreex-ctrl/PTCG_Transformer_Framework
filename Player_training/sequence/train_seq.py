"""SEQUENCE arm trainer: whole trajectories, in order.

Reads the SAME trajectories.npz the baseline reads. A sample is one full
game from one agent's side:

    DECK -> obs_1 -> act_1 -> ... -> obs_N -> act_N -> win/lose

Every decision is scored knowing the deck and every earlier decision of
that game. Loss, optimiser, lr, epochs and decisions-per-batch match the
baseline arm; the trajectory memory is the only variable.

Usage:
    python3 train_seq.py --data ../trajectories.npz --out sequence.pt
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

import numpy as np
import torch

from encoder import player_loss
from player_vocab import MAX_SEQ
from seq_model import SeqPlayer, MAX_TRAJ

HERE = pathlib.Path(__file__).parent


class Trajectories:
    def __init__(self, path):
        d = np.load(path)
        self.tok_flat = d["tok_flat"]
        self.tok_off = d["tok_off"]
        self.pos_flat = d["pos_flat"]
        self.pos_off = d["pos_off"]
        self.chosen_flat = d["chosen_flat"]
        self.chosen_off = d["chosen_off"]
        self.deck_flat = d["deck_flat"]
        self.deck_off = d["deck_off"]
        self.traj_off = d["traj_off"]
        self.reward = d["reward"]
        self.vocab_size = int(d["vocab_size"])
        self.n = len(self.reward)
        self.n_dec = len(self.tok_off) - 1
        self.traj_len = np.diff(self.traj_off)

    def batch(self, games, device):
        """games: iterable of trajectory indices -> padded tensors."""
        games = list(games)
        g = len(games)
        t = int(max(self.traj_len[i] for i in games))
        dec_ids = [list(range(self.traj_off[i], self.traj_off[i + 1]))
                   for i in games]

        max_l = max(self.tok_off[j + 1] - self.tok_off[j]
                    for ids in dec_ids for j in ids)
        max_o = max(self.pos_off[j + 1] - self.pos_off[j]
                    for ids in dec_ids for j in ids)

        tokens = np.zeros((g, t, max_l), dtype=np.int64)
        last_idx = np.zeros((g, t), dtype=np.int64)
        opt_pos = np.zeros((g, t, max_o), dtype=np.int64)
        opt_mask = np.zeros((g, t, max_o), dtype=bool)
        chosen_mask = np.zeros((g, t, max_o), dtype=bool)
        dec_mask = np.zeros((g, t), dtype=bool)
        deck = np.zeros((g, 60), dtype=np.int64)

        for gi, (gid, ids) in enumerate(zip(games, dec_ids)):
            a, b = self.deck_off[gid], self.deck_off[gid + 1]
            deck[gi, :b - a] = self.deck_flat[a:b]
            for ti, j in enumerate(ids):
                s = self.tok_flat[self.tok_off[j]:self.tok_off[j + 1]]
                o = self.pos_flat[self.pos_off[j]:self.pos_off[j + 1]]
                c = self.chosen_flat[self.chosen_off[j]:self.chosen_off[j + 1]]
                tokens[gi, ti, :len(s)] = s      # RIGHT padded (causal-safe)
                last_idx[gi, ti] = len(s) - 1
                opt_pos[gi, ti, :len(o)] = o
                opt_mask[gi, ti, :len(o)] = True
                chosen_mask[gi, ti, c] = True
                dec_mask[gi, ti] = True

        to = lambda a, dt: torch.from_numpy(a.astype(dt)).to(device)
        return (to(tokens, np.int64), to(last_idx, np.int64),
                to(opt_pos, np.int64), to(opt_mask, bool),
                to(chosen_mask, bool), to(deck, np.int64),
                to(dec_mask, bool),
                torch.from_numpy(self.reward[np.asarray(games)]).to(device))


def main():
    ap = argparse.ArgumentParser(description="sequence (trajectory) arm")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "sequence.pt"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--decisions-per-batch", type=int, default=256,
                    help="MUST match the baseline arm for a fair comparison")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--mem-layers", type=int, default=2)
    ap.add_argument("--init", default=None,
                    help="checkpoint to warm-start from: an R2-era player.pt "
                         "(remapped, memory zero-init) or a previous "
                         "sequence.pt from an earlier round")
    ap.add_argument("--epoch-ckpt-dir", default=None,
                    help="also write an unconditional epoch_NNN.pt snapshot "
                         "here after every epoch, plus a TRAINING_DONE marker "
                         "at the end. Used by bulk mode's evaluator; this "
                         "file never reads the directory back.")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.set_float32_matmul_precision("high")

    tr = Trajectories(args.data)
    print(f"[sequence] {tr.n} trajectories | {tr.n_dec} decisions "
          f"| decisions/traj mean {tr.traj_len.mean():.1f} "
          f"max {tr.traj_len.max()} | vocab {tr.vocab_size} "
          f"| device {device} | bf16 {'on' if use_amp else 'off'}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(tr.n)
    n_val = max(1, int(tr.n * args.val_frac))
    val_games, train_games = perm[:n_val], perm[n_val:]
    print(f"[sequence] split by game: {len(train_games)} train "
          f"| {len(val_games)} val", flush=True)

    model = SeqPlayer(tr.vocab_size, max_len=MAX_SEQ,
                      mem_layers=args.mem_layers, max_traj=MAX_TRAJ)
    if args.init:
        sys.path.insert(0, str(HERE.parent / "tools"))
        from warm_start import remap_r2
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck.get("arch") == "sequence":
            model.load_state_dict(ck["model"])
            print(f"[sequence] resumed from {args.init}", flush=True)
        else:                       # R2-era PlayerDecoder
            torch.nn.init.zeros_(model.mem_out.weight)
            torch.nn.init.zeros_(model.mem_out.bias)
            miss, unexp = model.load_state_dict(remap_r2(ck["model"]),
                                                strict=False)
            if unexp:
                raise SystemExit(f"unexpected keys in {args.init}: {unexp}")
            print(f"[sequence] warm-started from R2 {args.init}: "
                  f"encoder carried, memory zero-init "
                  f"({len(miss)} fresh tensors)", flush=True)
    model = model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[sequence] parameters: {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def game_batches(games, shuffle):
        """Group games of similar length, then size each batch so it holds
        ~decisions-per-batch decisions (matching the baseline's step size)."""
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

    def run_epoch(games, train):
        model.train(train)
        tot = cnt = 0.0
        batches = game_batches(games, train)
        t0 = time.time()
        for si, b in enumerate(batches, 1):
            (tokens, last_idx, pos, omask, cmask,
             deck, dmask, rew) = tr.batch(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                scores = model(tokens, last_idx, pos, omask, deck, dmask)

            g, t, o = scores.shape
            flat_scores = scores.reshape(g * t, o).float()
            flat_omask = omask.reshape(g * t, o)
            flat_cmask = cmask.reshape(g * t, o)
            flat_rew = rew.unsqueeze(1).expand(g, t).reshape(g * t)
            flat_dmask = dmask.reshape(g * t)

            per = player_loss(flat_scores, flat_omask, flat_cmask, flat_rew)
            n_real = flat_dmask.sum().clamp(min=1)
            loss = (per * flat_dmask.float()).sum() / n_real

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            nd = int(n_real)
            tot += float(loss.detach()) * nd
            cnt += nd
            if train and si % 100 == 0:
                print(f"    step {si}/{len(batches)} loss {tot/cnt:.4f} "
                      f"({cnt/(time.time()-t0):.0f} dec/s)", flush=True)
        return tot / max(cnt, 1)

    def payload(va):
        return {"model": model.state_dict(),
                "vocab_size": tr.vocab_size,
                "max_len": MAX_SEQ,
                "mem_layers": args.mem_layers,
                "max_traj": MAX_TRAJ,
                "arch": "sequence",
                "val_loss": va}

    ep_dir = pathlib.Path(args.epoch_ckpt_dir) if args.epoch_ckpt_dir else None
    if ep_dir:
        ep_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        trl = run_epoch(train_games, True)
        with torch.no_grad():
            va = run_epoch(val_games, False)
        print(f"[sequence] epoch {ep}: train {trl:.4f} | val {va:.4f}",
              flush=True)
        if va < best:
            best = va
            torch.save(payload(va), args.out)
            print(f"[sequence] saved {args.out} (val {va:.4f})", flush=True)
        if ep_dir:
            # unconditional per-epoch snapshot. Written to a temp name and
            # renamed so a reader can never see a half-written file.
            final = ep_dir / f"epoch_{ep:03d}.pt"
            tmp = ep_dir / f".epoch_{ep:03d}.pt.tmp"
            torch.save(payload(va), tmp)
            tmp.rename(final)
            print(f"[sequence] snapshot {final}", flush=True)
    if ep_dir:
        (ep_dir / "TRAINING_DONE").write_text(
            f"epochs={args.epochs} best_val={best:.6f}\n")
    print(f"[sequence] done. best val {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
