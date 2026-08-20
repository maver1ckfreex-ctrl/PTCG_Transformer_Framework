"""Load an R2-era checkpoint into either arm of the trial.

The old player (player_model.PlayerDecoder) is built from
nn.TransformerEncoderLayer; the trial's encoder is a hand-written
CausalBlock. The MATH is the same pre-LN causal block, and every tensor has
the same shape -- only the parameter names differ:

    blocks.layers.{i}.self_attn.in_proj_weight  ->  enc.blocks.{i}.qkv.weight
    blocks.layers.{i}.self_attn.out_proj.weight ->  enc.blocks.{i}.proj.weight
    blocks.layers.{i}.linear1.*                 ->  enc.blocks.{i}.ff1.*
    blocks.layers.{i}.linear2.*                 ->  enc.blocks.{i}.ff2.*
    blocks.layers.{i}.norm{1,2}.*               ->  enc.blocks.{i}.norm{1,2}.*
    embed / pos / norm / score                  ->  enc.embed / enc.pos /
                                                    enc.norm / score

The qkv layout matches too: nn.MultiheadAttention stacks in_proj as
[Wq; Wk; Wv] along dim 0, and CausalBlock reads its (B, L, 3D) output as
(3, H, dh) -- the same ordering. `verify_r2.py` checks this numerically
rather than trusting the argument.

For the sequence arm, `mem_out` is ZERO-INITIALISED so the trajectory
context contributes exactly 0 at step 0. The warm-started sequence model is
therefore bit-identical to R2 on the first batch, and only diverges as it
learns to use the memory. Without that, a randomly initialised memory
projection would inject noise straight into the option scores and throw
away the warm start.

Usage:
    python3 tools/warm_start.py --r2 ../submission_r2_t07_torch/player.pt \
        --arch sequence --out sequence/sequence_init.pt
    python3 tools/warm_start.py --r2 ../submission_r2_t07_torch/player.pt \
        --arch baseline --out baseline/baseline_init.pt
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "baseline"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "sequence"))

import torch


def remap_r2(old_sd, n_layers=4):
    """R2 PlayerDecoder state_dict -> trial encoder state_dict."""
    new = {}
    new["enc.embed.weight"] = old_sd["embed.weight"]
    new["enc.pos.weight"] = old_sd["pos.weight"]
    new["enc.norm.weight"] = old_sd["norm.weight"]
    new["enc.norm.bias"] = old_sd["norm.bias"]
    new["score.weight"] = old_sd["score.weight"]
    new["score.bias"] = old_sd["score.bias"]
    pairs = [("self_attn.in_proj_weight", "qkv.weight"),
             ("self_attn.in_proj_bias", "qkv.bias"),
             ("self_attn.out_proj.weight", "proj.weight"),
             ("self_attn.out_proj.bias", "proj.bias"),
             ("linear1.weight", "ff1.weight"), ("linear1.bias", "ff1.bias"),
             ("linear2.weight", "ff2.weight"), ("linear2.bias", "ff2.bias"),
             ("norm1.weight", "norm1.weight"), ("norm1.bias", "norm1.bias"),
             ("norm2.weight", "norm2.weight"), ("norm2.bias", "norm2.bias")]
    for i in range(n_layers):
        for old_suf, new_suf in pairs:
            new[f"enc.blocks.{i}.{new_suf}"] = \
                old_sd[f"blocks.layers.{i}.{old_suf}"]
    return new


def build(arch, vocab_size, max_len, mem_layers=2, max_traj=256):
    if arch == "sequence":
        from seq_model import SeqPlayer
        return SeqPlayer(vocab_size, max_len=max_len,
                         mem_layers=mem_layers, max_traj=max_traj)
    from baseline_model import BaselinePlayer
    return BaselinePlayer(vocab_size, max_len=max_len)


def warm_start(r2_path, arch, mem_layers=2, max_traj=256):
    ck = torch.load(r2_path, map_location="cpu", weights_only=False)
    old_sd = ck["model"]
    model = build(arch, ck["vocab_size"], ck["max_len"], mem_layers, max_traj)

    new_sd = remap_r2(old_sd)
    if arch == "sequence":
        # memory starts as a no-op: context * 0 -> scores identical to R2
        torch.nn.init.zeros_(model.mem_out.weight)
        torch.nn.init.zeros_(model.mem_out.bias)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    carried = len(new_sd)
    fresh = [k for k in missing]
    return model, ck, carried, fresh, unexpected


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="R2 -> trial warm start")
    ap.add_argument("--r2", required=True)
    ap.add_argument("--arch", choices=["baseline", "sequence"],
                    required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-layers", type=int, default=2)
    args = ap.parse_args()

    model, ck, carried, fresh, unexpected = warm_start(
        args.r2, args.arch, args.mem_layers)
    if unexpected:
        sys.exit(f"ERROR unexpected keys: {unexpected}")

    print(f"carried {carried} tensors from {args.r2}")
    if fresh:
        print(f"freshly initialised ({len(fresh)}):")
        for k in fresh:
            print(f"    {k}")
    else:
        print("freshly initialised: none -- exact R2 weights")

    out = {"model": model.state_dict(),
           "vocab_size": ck["vocab_size"],
           "max_len": ck["max_len"],
           "arch": args.arch,
           "warm_started_from": str(args.r2)}
    if args.arch == "sequence":
        out["mem_layers"] = args.mem_layers
        out["max_traj"] = 256
    torch.save(out, args.out)
    print(f"saved {args.out}")
