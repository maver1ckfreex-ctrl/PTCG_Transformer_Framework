"""Generate decks from a trained builder checkpoint.

Samples the deck in the builder's learned order (win condition first,
filler last) with a legality mask at every step:
  - max 4 copies per card NAME (basic energy exempt)
  - max 1 ACE SPEC card total
  - at least 1 Basic Pokemon (forced by slot 55 if still missing)
Always completes to exactly 60 cards.

Usage:
    python3 build_deck.py --ckpt builder_tf.pt --n 5 --temperature 0.9
"""

import argparse
import collections
import pathlib

import numpy as np
import torch

from builder_model import DeckDecoder
from card_vocab import Vocab, BOS, BUILD, PAD, PLAY, CARD_OFFSET

HERE = pathlib.Path(__file__).parent
FORCE_BASIC_AT = 55


def legality_mask(vocab, deck, vocab_size):
    """Boolean mask (vocab_size,) of tokens allowed for the next card."""
    mask = np.zeros(vocab_size, dtype=bool)
    name_counts = collections.Counter(vocab.name[c] for c in deck)
    ace_used = any(vocab.is_ace_spec[c] for c in deck)
    need_basic = (len(deck) >= FORCE_BASIC_AT
                  and not any(vocab.is_basic_pkm[c] for c in deck))
    for cid in vocab.card_ids:
        if need_basic and not vocab.is_basic_pkm[cid]:
            continue
        if ace_used and vocab.is_ace_spec[cid]:
            continue
        if (not vocab.is_basic_energy[cid]
                and name_counts[vocab.name[cid]] >= 4):
            continue
        mask[vocab.token(cid)] = True
    return mask


def sample_deck(model, vocab, vocab_size, temperature, device, rng):
    tokens = [BOS, BUILD]
    deck = []
    with torch.no_grad():
        while len(deck) < 60:
            t = torch.tensor([tokens], dtype=torch.int64, device=device)
            logits = model(t)[0, -1].float().cpu().numpy()
            logits[[PAD, BOS, BUILD, PLAY]] = -np.inf
            logits[~legality_mask(vocab, deck, vocab_size)] = -np.inf
            logits = logits / max(temperature, 1e-4)
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            tok = int(rng.choice(vocab_size, p=p))
            deck.append(tok - CARD_OFFSET)
            if len(tokens) < model.max_len:
                tokens.append(tok)
    return deck


def check_legal(vocab, deck):
    problems = []
    if len(deck) != 60:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample decks from builder")
    parser.add_argument("--ckpt", default=str(HERE / "builder_tf.pt"))
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--cards", default=None)
    parser.add_argument("--out-prefix", default=str(HERE / "deck_tf"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    vocab = Vocab(args.cards) if args.cards else Vocab()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab_size = ckpt["vocab_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeckDecoder(vocab_size, max_len=ckpt["max_len"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rng = np.random.default_rng(args.seed)

    for i in range(1, args.n + 1):
        deck = sample_deck(model, vocab, vocab_size, args.temperature,
                           device, rng)
        problems = check_legal(vocab, deck)
        out = pathlib.Path(f"{args.out_prefix}_{i}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(str(c) for c in deck) + "\n")
        status = "LEGAL" if not problems else f"ILLEGAL: {problems}"
        print(f"\n=== deck {i} -> {out.name} [{status}] ===")
        counts = collections.Counter(deck)
        for cid, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {cnt} x {vocab.name[cid]} (id {cid})")
