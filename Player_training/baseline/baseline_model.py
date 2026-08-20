"""BASELINE arm: decision-level player. No trajectory, no memory.

Each decision is scored from its own tokens alone -- exactly the current
design. Shares common/encoder.py with the sequence arm so that the only
difference between the two models is the trajectory memory.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "common"))

import torch.nn as nn

from encoder import (DecisionEncoder, gather_options,
                     D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT)


class BaselinePlayer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=D_MODEL,
                 n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF,
                 dropout=DROPOUT):
        super().__init__()
        self.enc = DecisionEncoder(vocab_size, max_len, d_model, n_layers,
                                   n_heads, d_ff, dropout)
        self.score = nn.Linear(d_model, 1)

    def forward(self, tokens, opt_pos, opt_mask):
        """tokens (B,L) right-padded | opt_pos (B,O) | opt_mask (B,O).

        Returns scores (B,O) with -inf on absent options.
        """
        hidden = self.enc(tokens)
        opts = gather_options(hidden, opt_pos)
        scores = self.score(opts).squeeze(-1)
        return scores.masked_fill(~opt_mask, float("-inf"))
