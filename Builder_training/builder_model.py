"""Decoder-only transformer shared by the builder and the player models.

Sized for 7k-100k replay samples (~5M params at the defaults below).
Hand-tune the constants; both train scripts read them.
"""

import torch
import torch.nn as nn

D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
D_FF = 1024
DROPOUT = 0.1


class DeckDecoder(nn.Module):
    def __init__(self, vocab_size, max_len,
                 d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight        # weight tying

    def forward(self, tokens):
        """tokens (B, L) int64 -> logits (B, L, vocab)."""
        b, l = tokens.shape
        pos = torch.arange(l, device=tokens.device)
        x = self.drop(self.embed(tokens) + self.pos(pos)[None])
        causal = nn.Transformer.generate_square_subsequent_mask(
            l, device=tokens.device)
        x = self.blocks(x, mask=causal, is_causal=True)
        return self.head(self.norm(x))


def reward_weighted_loss(logits, targets, reward, pad=0):
    """Next-token loss with the sequence reward revealed at the end.

    reward +1: maximize log-likelihood of every token of the deck.
    reward -1: push the sequence's probability DOWN — implemented as
               unlikelihood, -log(1 - p), the numerically stable form
               of a -1 reward (plain negated log-likelihood diverges).
    The reward is one scalar per sequence, applied uniformly; the model
    never sees the outcome inside the tokens.
    """
    logp = torch.log_softmax(logits, dim=-1)
    ll = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (B, L)
    mask = (targets != pad).float()

    pos_loss = -ll
    neg_loss = -torch.log1p(-torch.exp(ll).clamp(max=1 - 1e-6))
    tok_loss = torch.where(reward[:, None] > 0, pos_loss, neg_loss)
    return (tok_loss * mask).sum() / mask.sum().clamp(min=1.0)
