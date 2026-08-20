"""V2 player: decoder transformer that scores the engine's options.

Reads the tokenized decision forward (state then options, causal), takes
the hidden state at each option span's end, and scores it. Training:
the option the winner chose is pushed up, the loser's choice pushed
down — reward +/-1 revealed only after the sequence is read.

Shares the card-embedding table with the builder (pass `embed`).
"""

import torch
import torch.nn as nn

from builder_model import D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT
from player_vocab import MAX_SEQ


class PlayerDecoder(nn.Module):
    def __init__(self, vocab_size, embed=None, max_len=MAX_SEQ,
                 d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.max_len = max_len
        self.embed = embed if embed is not None else nn.Embedding(
            vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)

    def forward(self, tokens, pad_mask, opt_pos, opt_mask):
        """tokens (B,L) | pad_mask (B,L) True=pad | opt_pos (B,O) |
        opt_mask (B,O) True=real option. Returns scores (B,O)."""
        b, l = tokens.shape
        pos = torch.arange(l, device=tokens.device)
        x = self.drop(self.embed(tokens) + self.pos(pos)[None])
        causal = nn.Transformer.generate_square_subsequent_mask(
            l, device=tokens.device)
        x = self.blocks(x, mask=causal, src_key_padding_mask=pad_mask,
                        is_causal=True)
        x = self.norm(x)
        gathered = x.gather(
            1, opt_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        scores = self.score(gathered).squeeze(-1)          # (B, O)
        return scores.masked_fill(~opt_mask, float("-inf"))


def player_loss(scores, opt_mask, chosen_mask, reward):
    """Reward-weighted option loss.

    scores (B,O) -inf on absent options | chosen_mask (B,O) True where
    the agent actually picked | reward (B,) +1/-1.

    r=+1: cross-entropy pushing the chosen options' probability up.
    r=-1: unlikelihood pushing the chosen options' probability down
          (what NOT to pick; the right pick stays unknown).
    """
    logp = torch.log_softmax(scores, dim=-1)
    logp = torch.where(opt_mask, logp, torch.zeros_like(logp))
    chosen = chosen_mask.float()
    n_chosen = chosen.sum(-1).clamp(min=1.0)
    chosen_ll = (logp * chosen).sum(-1) / n_chosen        # (B,)

    pos_loss = -chosen_ll
    p = chosen_ll.exp().clamp(max=1 - 1e-6)
    neg_loss = -torch.log1p(-p)
    loss = torch.where(reward > 0, pos_loss, neg_loss)
    return loss.mean()
