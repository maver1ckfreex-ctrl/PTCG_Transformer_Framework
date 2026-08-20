"""v3 model: trajectory player with an END-OF-GAME outcome head.

Same two-level structure as seq_model.SeqPlayer:
    level 1  per decision  -> summary s_t
    level 2  causal over [deck, s_1..s_N] -> context c_t

What v3 adds is one extra read position. Level 2 already runs over N+1
positions; v1/v2 threw the last one away. v3 keeps it:

    c_1 .. c_N   -> option scores  (what to do at each step)
    c_final      -> win/lose       (only after every action has been read)

So the outcome is predicted ONCE, at the end of the trajectory, instead of
being stamped onto all N decisions.
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


class SeqPlayerV3(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=D_MODEL,
                 n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF,
                 dropout=DROPOUT, mem_layers=MEM_LAYERS, max_traj=MAX_TRAJ):
        super().__init__()
        self.max_traj = max_traj
        self.enc = DecisionEncoder(vocab_size, max_len, d_model, n_layers,
                                   n_heads, d_ff, dropout)
        self.deck_proj = nn.Linear(d_model, d_model)
        self.mem_pos = nn.Embedding(max_traj + 2, d_model)
        self.mem_blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(mem_layers)])
        self.mem_norm = nn.LayerNorm(d_model)
        self.mem_out = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)
        self.value = nn.Linear(d_model, 1)      # win/lose, END of trajectory

    def encode_deck(self, deck):
        return self.deck_proj(self.enc.embed(deck).mean(dim=1))

    def encode_decisions(self, tokens, last_idx):
        hidden = self.enc(tokens)
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return hidden, hidden.gather(1, idx).squeeze(1)

    def memory(self, deck_vec, summaries):
        """-> (context per step, full sequence).

        context[:, t] sees deck + summaries[:, :t] (strictly the past).
        The returned full sequence keeps position T, which has read every
        decision -- that is where the outcome is predicted.
        """
        g, t, d = summaries.shape
        x = torch.cat([deck_vec.unsqueeze(1), summaries], dim=1)
        pos = torch.arange(t + 1, device=x.device)
        x = x + self.mem_pos(pos)[None]
        for blk in self.mem_blocks:
            x = blk(x)
        x = self.mem_norm(x)
        return self.mem_out(x[:, :t]), x

    def score_options(self, hidden, opt_pos, opt_mask, context):
        opts = gather_options(hidden, opt_pos) + context.unsqueeze(1)
        return self.score(opts).squeeze(-1).masked_fill(
            ~opt_mask, float("-inf"))

    def forward(self, tokens, last_idx, opt_pos, opt_mask, deck, dec_mask):
        """-> scores (G,T,O), outcome_logit (G,)."""
        g, t, l = tokens.shape
        hidden, summary = self.encode_decisions(
            tokens.reshape(g * t, l), last_idx.reshape(g * t))
        summary = summary.view(g, t, -1) * dec_mask.unsqueeze(-1)
        context, full = self.memory(self.encode_deck(deck), summary)

        o = opt_pos.shape[-1]
        scores = self.score_options(
            hidden, opt_pos.reshape(g * t, o), opt_mask.reshape(g * t, o),
            context.reshape(g * t, -1)).view(g, t, o)

        # outcome is read at the position AFTER the last REAL decision, so
        # padding never shifts where the model looks
        n_real = dec_mask.sum(dim=1).clamp(min=1)              # (G,)
        idx = n_real.view(g, 1, 1).expand(-1, 1, full.shape[-1])
        outcome = self.value(full.gather(1, idx).squeeze(1)).squeeze(-1)
        return scores, outcome


class SeqPlayerV3Runner:
    """Incremental wrapper for play. reset() at the start of every game."""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self, deck_tokens=None):
        self.summaries = []
        self.deck_vec = None
        if deck_tokens is not None:
            with torch.no_grad():
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
        context, _ = m.memory(self.deck_vec, padded)
        n = len(opt_pos)
        pos = torch.tensor([list(opt_pos)], dtype=torch.int64,
                           device=self.device)
        mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
        out = m.score_options(hidden, pos, mask, context[:, -1])[0]
        self.summaries.append(summary)
        if len(self.summaries) > m.max_traj:
            self.summaries.pop(0)
        return out.tolist()
