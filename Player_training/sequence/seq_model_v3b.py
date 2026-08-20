"""v3b model: outcome token at the HEAD of the trajectory.

Level 2 reads

    [ WIN|LOSE , deck , s_1 , s_2 , ... , s_N ]

so every decision is scored knowing, from the very first position, whether
this trajectory ended in a win or a loss. At training time that token is
the true outcome. At play time it is pinned to WIN -- you ask the model
"act like this is a game I win."

No outcome-prediction head here, unlike v3: the outcome is an INPUT, so a
head predicting it would just read its own input.

Indexing (level 2 sequence is length T+2, causal):
    position 0   = outcome token
    position 1   = deck            -> context for decision 0
    position 1+t = s_{t-1}         -> context for decision t
so context = out[:, 1:T+1], and decision t sees outcome + deck +
s_0..s_{t-1} -- strictly the past, plus the conditioning token.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

import torch
import torch.nn as nn

from encoder import (CausalBlock, DecisionEncoder, gather_options,
                     D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)

MEM_LAYERS = 2
MAX_TRAJ = 256
LOSE, WIN = 0, 1


class SeqPlayerV3B(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=D_MODEL,
                 n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF,
                 dropout=DROPOUT, mem_layers=MEM_LAYERS, max_traj=MAX_TRAJ):
        super().__init__()
        self.max_traj = max_traj
        self.enc = DecisionEncoder(vocab_size, max_len, d_model, n_layers,
                                   n_heads, d_ff, dropout)
        self.deck_proj = nn.Linear(d_model, d_model)
        self.outcome_emb = nn.Embedding(2, d_model)      # LOSE / WIN
        self.mem_pos = nn.Embedding(max_traj + 2, d_model)
        self.mem_blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(mem_layers)])
        self.mem_norm = nn.LayerNorm(d_model)
        self.mem_out = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)

    def encode_deck(self, deck):
        return self.deck_proj(self.enc.embed(deck).mean(dim=1))

    def encode_decisions(self, tokens, last_idx):
        hidden = self.enc(tokens)
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return hidden, hidden.gather(1, idx).squeeze(1)

    def memory(self, outcome_vec, deck_vec, summaries):
        """-> context (G, T, D); context[:, t] sees outcome + deck +
        summaries[:, :t]."""
        g, t, d = summaries.shape
        x = torch.cat([outcome_vec.unsqueeze(1),
                       deck_vec.unsqueeze(1), summaries], dim=1)
        pos = torch.arange(t + 2, device=x.device)
        x = x + self.mem_pos(pos)[None]
        for blk in self.mem_blocks:
            x = blk(x)
        x = self.mem_norm(x)
        return self.mem_out(x[:, 1:t + 1])

    def score_options(self, hidden, opt_pos, opt_mask, context):
        opts = gather_options(hidden, opt_pos) + context.unsqueeze(1)
        return self.score(opts).squeeze(-1).masked_fill(
            ~opt_mask, float("-inf"))

    def forward(self, tokens, last_idx, opt_pos, opt_mask, deck, dec_mask,
                outcome):
        """outcome (G,) long, 1=win 0=lose -> scores (G,T,O)."""
        g, t, l = tokens.shape
        hidden, summary = self.encode_decisions(
            tokens.reshape(g * t, l), last_idx.reshape(g * t))
        summary = summary.view(g, t, -1) * dec_mask.unsqueeze(-1)
        context = self.memory(self.outcome_emb(outcome),
                              self.encode_deck(deck), summary)
        o = opt_pos.shape[-1]
        return self.score_options(
            hidden, opt_pos.reshape(g * t, o), opt_mask.reshape(g * t, o),
            context.reshape(g * t, -1)).view(g, t, o)


class SeqPlayerV3BRunner:
    """Incremental wrapper for play. Conditioned on WIN by default."""

    def __init__(self, model, device=None, condition=WIN):
        self.model = model
        self.device = device or torch.device("cpu")
        self.condition = condition
        self.reset()

    def reset(self, deck_tokens=None, condition=None):
        self.summaries = []
        self.deck_vec = None
        if condition is not None:
            self.condition = condition
        with torch.no_grad():
            self.outcome_vec = self.model.outcome_emb(
                torch.tensor([self.condition], dtype=torch.int64,
                             device=self.device))
            if deck_tokens is not None:
                deck = torch.tensor([list(deck_tokens)], dtype=torch.int64,
                                    device=self.device)
                self.deck_vec = self.model.encode_deck(deck)

    @torch.no_grad()
    def scores(self, toks, opt_pos):
        m = self.model
        tokens = torch.tensor([list(toks)], dtype=torch.int64,
                              device=self.device)
        last = torch.tensor([len(toks) - 1], dtype=torch.int64,
                            device=self.device)
        hidden, summary = m.encode_decisions(tokens, last)
        if self.deck_vec is None:
            self.deck_vec = torch.zeros_like(summary)
        past = (torch.cat(self.summaries, dim=0).unsqueeze(0)
                if self.summaries
                else torch.zeros((1, 0, summary.shape[-1]),
                                 device=self.device, dtype=summary.dtype))
        padded = torch.cat([past, summary.unsqueeze(1)], dim=1)
        context = m.memory(self.outcome_vec, self.deck_vec, padded)
        n = len(opt_pos)
        pos = torch.tensor([list(opt_pos)], dtype=torch.int64,
                           device=self.device)
        mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
        out = m.score_options(hidden, pos, mask, context[:, -1])[0]
        self.summaries.append(summary)
        if len(self.summaries) > m.max_traj:
            self.summaries.pop(0)
        return out.tolist()
