"""SEQUENCE arm: trajectory player. Reads the whole replay in order.

Two levels:

  level 1  per decision, the SAME DecisionEncoder the baseline uses
           -> a summary vector s_t for each decision

  level 2  a causal transformer over the game's sequence
               [ deck , s_1 , s_2 , ... , s_N ]
           -> a context vector c_t for each decision, where c_t sees the
              deck and every EARLIER decision in the same game (strictly
              causal: c_1 is the deck alone)

The option score for decision t is read from level 1 as usual, with c_t
added in. So a decision is scored knowing what deck it is piloting and
everything the agent already did this game -- which is the whole point of
the trial. The +/-1 outcome weighting and the loss are unchanged from the
baseline.

Trajectories are right-padded and level 2 is causal, so a real decision
never attends to a padded one.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

import torch
import torch.nn as nn

from encoder import (CausalBlock, DecisionEncoder, gather_options,
                     D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)

MEM_LAYERS = 2          # level-2 depth; sequence is only ~60 long
MAX_TRAJ = 256          # decisions per trajectory (guard; observed max ~130)


class SeqPlayer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=D_MODEL,
                 n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF,
                 dropout=DROPOUT, mem_layers=MEM_LAYERS,
                 max_traj=MAX_TRAJ):
        super().__init__()
        self.max_traj = max_traj
        self.enc = DecisionEncoder(vocab_size, max_len, d_model, n_layers,
                                   n_heads, d_ff, dropout)

        # the deck sits at position 0 of the trajectory
        self.deck_proj = nn.Linear(d_model, d_model)

        # level 2: memory across decisions
        self.mem_pos = nn.Embedding(max_traj + 1, d_model)
        self.mem_blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(mem_layers)])
        self.mem_norm = nn.LayerNorm(d_model)
        self.mem_out = nn.Linear(d_model, d_model)

        self.score = nn.Linear(d_model, 1)

    # ---- pieces, reused by training and by incremental inference ----

    def encode_deck(self, deck):
        """deck (G, 60) card tokens -> (G, D)."""
        return self.deck_proj(self.enc.embed(deck).mean(dim=1))

    def encode_decisions(self, tokens, last_idx):
        """tokens (B, L) | last_idx (B,) index of each decision's final real
        token. Returns per-token hidden (B, L, D) and summary (B, D)."""
        hidden = self.enc(tokens)
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        summary = hidden.gather(1, idx).squeeze(1)
        return hidden, summary

    def memory(self, deck_vec, summaries):
        """deck_vec (G, D) | summaries (G, T, D) -> context (G, T, D).

        context[:, t] sees the deck and summaries[:, :t] -- strictly the
        past, so decision t is never scored using its own summary twice or
        anything from the future.
        """
        g, t, d = summaries.shape
        x = torch.cat([deck_vec.unsqueeze(1), summaries], dim=1)  # (G, T+1, D)
        pos = torch.arange(t + 1, device=x.device)
        x = x + self.mem_pos(pos)[None]
        for blk in self.mem_blocks:
            x = blk(x)
        x = self.mem_norm(x)
        return self.mem_out(x[:, :t])          # drop the last, shift by one

    def score_options(self, hidden, opt_pos, opt_mask, context):
        """hidden (B, L, D) | opt_pos (B, O) | context (B, D)."""
        opts = gather_options(hidden, opt_pos)              # (B, O, D)
        opts = opts + context.unsqueeze(1)
        scores = self.score(opts).squeeze(-1)
        return scores.masked_fill(~opt_mask, float("-inf"))

    # ---- training forward: a batch of G whole trajectories ----

    def forward(self, tokens, last_idx, opt_pos, opt_mask, deck, dec_mask):
        """tokens (G,T,L) | last_idx (G,T) | opt_pos (G,T,O) |
        opt_mask (G,T,O) | deck (G,60) | dec_mask (G,T) True=real decision.

        Returns scores (G, T, O).
        """
        g, t, l = tokens.shape
        hidden, summary = self.encode_decisions(
            tokens.reshape(g * t, l), last_idx.reshape(g * t))
        summary = summary.view(g, t, -1) * dec_mask.unsqueeze(-1)
        context = self.memory(self.encode_deck(deck), summary)   # (G,T,D)

        o = opt_pos.shape[-1]
        scores = self.score_options(
            hidden, opt_pos.reshape(g * t, o), opt_mask.reshape(g * t, o),
            context.reshape(g * t, -1))
        return scores.view(g, t, o)


class SeqPlayerRunner:
    """Incremental wrapper for play: carries trajectory memory across the
    decisions of ONE game. Call reset() at the start of every game."""

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
                self.deck_vec = self.model.encode_deck(deck)      # (1, D)

    @torch.no_grad()
    def scores(self, toks, opt_pos):
        """One decision -> list of option scores, conditioned on the deck
        and every earlier decision of this game."""
        m = self.model
        tokens = torch.tensor([list(toks)], dtype=torch.int64,
                              device=self.device)
        last = torch.tensor([len(toks) - 1], dtype=torch.int64,
                            device=self.device)
        hidden, summary = m.encode_decisions(tokens, last)

        if self.deck_vec is None:      # deck unknown -> zero prefix
            self.deck_vec = torch.zeros_like(summary)

        # context for THIS decision = deck + all previous summaries
        past = (torch.cat(self.summaries, dim=0).unsqueeze(0)
                if self.summaries
                else torch.zeros((1, 0, summary.shape[-1]),
                                 device=self.device, dtype=summary.dtype))
        padded = torch.cat([past, summary.unsqueeze(1)], dim=1)   # (1, t+1, D)
        context = m.memory(self.deck_vec, padded)[:, -1]          # (1, D)

        n = opt_pos.shape[0] if hasattr(opt_pos, "shape") else len(opt_pos)
        pos = torch.tensor([list(opt_pos)], dtype=torch.int64,
                           device=self.device)
        mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
        out = m.score_options(hidden, pos, mask, context)[0]

        self.summaries.append(summary)
        if len(self.summaries) > m.max_traj:
            self.summaries.pop(0)
        return out.tolist()
