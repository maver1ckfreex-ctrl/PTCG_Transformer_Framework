"""Generate decks from a trained builder checkpoint.

Samples the deck in the builder's learned order (win condition first,
filler last) with a legality mask at every step:
  - max 4 copies per card NAME (basic energy exempt)
  - max 1 ACE SPEC card total
  - at least 1 Basic Pokemon (forced by slot 55 if still missing)
Always completes to exactly 60 cards.

Sampling is batched: --batch decks are decoded in lockstep and the
legality mask is computed as tensors on the same device as the model,
so a large --n keeps the GPU busy instead of stalling on a per-token
sync. Batch 1 reproduces the old serial behaviour.

Usage:
    python3 build_deck.py --ckpt builder_tf.pt --n 5 --temperature 0.9
    python3 build_deck.py --ckpt builder_tf.pt --n 200000 --batch 4096 \
        --out-npy ../pool_200k.npy
"""

import argparse
import collections
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

from builder_model import DeckDecoder
from card_vocab import Vocab, BOS, BUILD, PAD, PLAY, CARD_OFFSET

HERE = pathlib.Path(__file__).parent
FORCE_BASIC_AT = 55
DECK_SIZE = 60
MAX_COPIES = 4


class LegalityTables:
    """Per-token card properties as device tensors, built once."""

    def __init__(self, vocab, vocab_size, device):
        names = sorted(set(vocab.name.values()))
        name_idx = {nm: i for i, nm in enumerate(names)}
        self.n_groups = len(names)

        is_card = torch.zeros(vocab_size, dtype=torch.bool)
        is_ace = torch.zeros(vocab_size, dtype=torch.bool)
        is_basic_pkm = torch.zeros(vocab_size, dtype=torch.bool)
        is_energy = torch.zeros(vocab_size, dtype=torch.bool)
        # non-card tokens park in a dummy group that nothing ever reads
        group = torch.full((vocab_size,), self.n_groups, dtype=torch.long)
        for cid in vocab.card_ids:
            tok = vocab.token(cid)
            is_card[tok] = True
            is_ace[tok] = vocab.is_ace_spec[cid]
            is_basic_pkm[tok] = vocab.is_basic_pkm[cid]
            is_energy[tok] = vocab.is_basic_energy[cid]
            group[tok] = name_idx[vocab.name[cid]]
        for special in (PAD, BOS, BUILD, PLAY):
            is_card[special] = False

        self.is_card = is_card.to(device)
        self.is_ace = is_ace.to(device)
        self.is_basic_pkm = is_basic_pkm.to(device)
        self.is_energy = is_energy.to(device)
        self.group = group.to(device)


def last_logits(model, tokens):
    """logits (B, vocab) for the next token only.

    Same math as DeckDecoder.forward but the output projection runs on
    the final position alone -- the other 59 are thrown away by the
    sampler, and the head is the widest matmul in the model.
    """
    _, length = tokens.shape
    pos = torch.arange(length, device=tokens.device)
    x = model.embed(tokens) + model.pos(pos)[None]
    causal = nn.Transformer.generate_square_subsequent_mask(
        length, device=tokens.device, dtype=x.dtype)
    x = model.blocks(x, mask=causal, is_causal=True)
    return model.head(model.norm(x[:, -1]))


def sample_batch(model, tables, vocab_size, temperature, device, batch,
                 generator, autocast_dtype=None):
    """Sample `batch` complete decks at once -> (batch, 60) card ids."""
    max_ctx = min(2 + DECK_SIZE, model.max_len)
    ctx = torch.empty((batch, max_ctx), dtype=torch.int64, device=device)
    ctx[:, 0] = BOS
    ctx[:, 1] = BUILD
    ctx_len = 2

    decks = torch.empty((batch, DECK_SIZE), dtype=torch.int64, device=device)
    name_count = torch.zeros((batch, tables.n_groups + 1),
                             dtype=torch.int32, device=device)
    ace_used = torch.zeros(batch, dtype=torch.bool, device=device)
    has_basic = torch.zeros(batch, dtype=torch.bool, device=device)
    group_index = tables.group.unsqueeze(0).expand(batch, vocab_size)
    group_index = group_index.contiguous()

    with torch.no_grad():
        for step in range(DECK_SIZE):
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type,
                                    dtype=autocast_dtype):
                    logits = last_logits(model, ctx[:, :ctx_len])
            else:
                logits = last_logits(model, ctx[:, :ctx_len])
            logits = logits.float()

            counts = name_count.gather(1, group_index)
            allowed = tables.is_card.unsqueeze(0) & (
                tables.is_energy.unsqueeze(0) | (counts < MAX_COPIES))
            allowed &= ~(ace_used.unsqueeze(1) & tables.is_ace.unsqueeze(0))
            if step >= FORCE_BASIC_AT:
                allowed &= has_basic.unsqueeze(1) | \
                    tables.is_basic_pkm.unsqueeze(0)

            logits = logits.masked_fill(~allowed, float("-inf"))
            probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
            tok = torch.multinomial(probs, 1, generator=generator).squeeze(1)

            decks[:, step] = tok - CARD_OFFSET
            name_count.scatter_add_(
                1, tables.group[tok].unsqueeze(1),
                torch.ones((batch, 1), dtype=torch.int32, device=device))
            ace_used |= tables.is_ace[tok]
            has_basic |= tables.is_basic_pkm[tok]
            if ctx_len < max_ctx:
                ctx[:, ctx_len] = tok
                ctx_len += 1

    return decks.cpu().numpy().astype(np.int32)


def check_legal(vocab, deck):
    problems = []
    if len(deck) != DECK_SIZE:
        problems.append(f"{len(deck)} cards")
    counts = collections.Counter(vocab.name[c] for c in deck)
    for cid in set(deck):
        if not vocab.is_basic_energy[cid] and counts[vocab.name[cid]] > 4:
            problems.append(f">4 x {vocab.name[cid]}")
    if sum(vocab.is_ace_spec[c] for c in deck) > 1:
        problems.append(">1 ACE SPEC")
    if not any(vocab.is_basic_pkm[c] for c in deck):
        problems.append("no Basic Pokemon")
    return problems


def write_csv(deck, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(c)) for c in deck) + "\n")


def parse_mix(text, total):
    """'0.7:0.8:0.9:1.0' or '0.7,0.8' -> [(temperature, count), ...].

    Optional 'temp*weight' sets the ratio; bare values are equal parts.
    The split is exact: the counts always sum to `total`.
    """
    parts = [p for p in text.replace(":", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("--temperature is empty")
    temps, weights = [], []
    for part in parts:
        temp, _, weight = part.partition("*")
        temps.append(float(temp))
        weights.append(float(weight) if weight else 1.0)
    if any(t <= 0 for t in temps) or any(w <= 0 for w in weights):
        raise ValueError("temperatures and weights must be positive")

    # largest-remainder: floor every share, then hand the leftover to the
    # parts that lost the most rounding down
    scale = total / sum(weights)
    exact = [weight * scale for weight in weights]
    counts = [int(value) for value in exact]
    order = sorted(range(len(exact)), key=lambda i: exact[i] - counts[i],
                   reverse=True)
    for index in range(total - sum(counts)):
        counts[order[index % len(order)]] += 1
    return [(t, c) for t, c in zip(temps, counts) if c > 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample decks from builder")
    parser.add_argument("--ckpt", default=str(HERE / "builder_tf.pt"))
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", default="0.9",
                        help="one value, or a mix: '0.7:0.8:0.9:1.0' splits "
                             "--n equally between them; '0.7*2:1.1' weights "
                             "the parts 2:1")
    parser.add_argument("--cards", default=None)
    parser.add_argument("--out-prefix", default=str(HERE / "deck_tf"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch", type=int, default=512,
                        help="decks decoded in parallel; raise until the "
                             "GPU is saturated (4096+ for large --n)")
    parser.add_argument("--out-npy", default=None,
                        help="write all decks to one (n, 60) int32 .npy "
                             "instead of one csv per deck")
    parser.add_argument("--fp32", action="store_true",
                        help="disable autocast (cuda defaults to bf16/fp16)")
    parser.add_argument("--verify", action="store_true",
                        help="re-check every deck on the cpu (slow for "
                             "large --n; a sample is always checked)")
    args = parser.parse_args()

    vocab = Vocab(args.cards) if args.cards else Vocab()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab_size = ckpt["vocab_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeckDecoder(vocab_size, max_len=ckpt["max_len"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    autocast_dtype = None
    if device.type == "cuda" and not args.fp32:
        autocast_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                          else torch.float16)
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    else:
        generator.seed()

    tables = LegalityTables(vocab, vocab_size, device)
    batch = max(1, min(args.batch, args.n))
    bulk = np.empty((args.n, DECK_SIZE), dtype=np.int32) if args.out_npy \
        else None
    mix = parse_mix(args.temperature, args.n)
    deck_temp = np.empty(args.n, dtype=np.float32)

    print(f"device={device} batch={batch} n={args.n} "
          f"autocast={autocast_dtype}", flush=True)
    print("  mix: " + ", ".join(f"T={t} x{c:,}" for t, c in mix), flush=True)
    start = time.time()
    done = 0
    illegal = 0
    for temperature, quota in mix:
        made = 0
        while made < quota:
            size = min(batch, quota - made)
            decks = sample_batch(model, tables, vocab_size, temperature,
                                 device, size, generator, autocast_dtype)
            for j in range(size):
                i = done + j
                deck = decks[j]
                deck_temp[i] = temperature
                if bulk is not None:
                    bulk[i] = deck
                else:
                    write_csv(deck,
                              pathlib.Path(f"{args.out_prefix}_{i + 1}.csv"))
                if args.verify or args.n <= 20 or i < 8:
                    problems = check_legal(vocab, [int(c) for c in deck])
                    illegal += bool(problems)
                    if args.n <= 20:
                        status = "LEGAL" if not problems \
                            else f"ILLEGAL: {problems}"
                        print(f"\n=== deck {i + 1} T={temperature} "
                              f"[{status}] ===")
                        counts = collections.Counter(int(c) for c in deck)
                        for cid, cnt in sorted(counts.items(),
                                               key=lambda kv: -kv[1]):
                            print(f"  {cnt} x {vocab.name[cid]} (id {cid})")
            made += size
            done += size
            if args.n > 20:
                rate = done / max(time.time() - start, 1e-6)
                print(f"  T={temperature}  {done}/{args.n} decks  "
                      f"{rate:.0f} decks/s", flush=True)

    if bulk is not None:
        out = pathlib.Path(args.out_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, bulk)
        # provenance for later analysis: which temperature made each row
        if len(mix) > 1:
            temp_path = out.with_name(out.stem + "_temps.npy")
            np.save(temp_path, deck_temp)
            print(f"wrote {temp_path} (temperature per deck)")
        print(f"\nwrote {out} {bulk.shape}")
    elapsed = time.time() - start
    print(f"{args.n} decks in {elapsed:.1f}s "
          f"({args.n / max(elapsed, 1e-6):.0f} decks/s), "
          f"illegal in checked subset: {illegal}")
