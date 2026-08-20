"""Shared per-decision encoder. Used UNCHANGED by both arms of the trial.

Same math as the original player_model.PlayerDecoder (pre-LN causal decoder
blocks, gelu, score read at each option span's end), with one implementation
change: attention runs through F.scaled_dot_product_attention(is_causal=True)
instead of an explicit L x L float mask.

That is a memory fix, not a design change. The explicit mask disqualifies
the fused attention kernel, so PyTorch materialises a B x H x L x L tensor
(at B=64, L=640 that is ~0.8 GB per layer, per direction) -- the reason
long-sequence batches blow up CUDA memory. is_causal=True is mathematically
identical and runs in O(L) memory.

Padding: sequences are RIGHT-padded and attention is causal, so a real
position never attends to a pad position. Pad positions compute garbage and
are never read (option positions are always < real length). No key-padding
mask is needed, which is what keeps the fused kernel eligible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
D_FF = 1024
DROPOUT = 0.1


class CausalBlock(nn.Module):
    """Pre-LN transformer block, causal self-attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.p_drop = dropout

    def forward(self, x):
        b, l, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, l, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # each (B, H, L, dh)
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.p_drop if self.training else 0.0)
        a = a.transpose(1, 2).reshape(b, l, d)
        x = x + self.drop(self.proj(a))
        h = self.norm2(x)
        x = x + self.drop(self.ff2(F.gelu(self.ff1(h))))
        return x


class DecisionEncoder(nn.Module):
    """Tokens of ONE decision -> per-token hidden states.

    tokens (B, L) right-padded. Returns (B, L, d_model).
    """

    def __init__(self, vocab_size, max_len, d_model=D_MODEL,
                 n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF,
                 dropout=DROPOUT, embed=None):
        super().__init__()
        self.max_len = max_len
        self.embed = embed if embed is not None else nn.Embedding(
            vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        b, l = tokens.shape
        pos = torch.arange(l, device=tokens.device)
        x = self.drop(self.embed(tokens) + self.pos(pos)[None])
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def gather_options(hidden, opt_pos):
    """hidden (B, L, D) | opt_pos (B, O) -> (B, O, D)."""
    idx = opt_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return hidden.gather(1, idx)


def player_loss(scores, opt_mask, chosen_mask, reward):
    """Reward-weighted option loss -- IDENTICAL in both arms of the trial.

    scores (B,O) -inf on absent options | chosen_mask (B,O) True where the
    agent actually picked | reward (B,) +1/-1.

    r=+1: cross-entropy pushing the chosen options' probability up.
    r=-1: unlikelihood pushing the chosen options' probability down.
    """
    logp = torch.log_softmax(scores, dim=-1)
    logp = torch.where(opt_mask, logp, torch.zeros_like(logp))
    chosen = chosen_mask.float()
    n_chosen = chosen.sum(-1).clamp(min=1.0)
    chosen_ll = (logp * chosen).sum(-1) / n_chosen

    pos_loss = -chosen_ll
    p = chosen_ll.exp().clamp(max=1 - 1e-6)
    neg_loss = -torch.log1p(-p)
    return torch.where(reward > 0, pos_loss, neg_loss)
