"""Card vocabulary for the transformer deck builder.

Self-contained parse of EN_Card_Data.csv (the card list: 1267 cards).
No imports from any previous design.

Token space:
    0 PAD | 1 BOS | 2 BUILD | 3 PLAY | card token = card_id + CARD_OFFSET
"""

import csv
import pathlib

PAD, BOS, BUILD, PLAY = 0, 1, 2, 3
CARD_OFFSET = 3            # card_id 1..1267 -> token 4..1270
DEFAULT_CARDS = pathlib.Path(__file__).parent.parent / "EN_Card_Data.csv"


class Vocab:
    def __init__(self, cards_csv=DEFAULT_CARDS):
        self.name = {}             # card_id -> card name
        self.is_basic_energy = {}  # card_id -> bool (copy cap exempt)
        self.is_basic_pkm = {}     # card_id -> bool
        self.is_ace_spec = {}      # card_id -> bool
        with open(cards_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = int(row["Card ID"])
                if cid in self.name:      # one row per attack; first row wins
                    continue
                stage = row["Stage (Pokémon)/Type (Energy and Trainer)"].strip()
                self.name[cid] = row["Card Name"].strip()
                self.is_basic_energy[cid] = stage == "Basic Energy"
                self.is_basic_pkm[cid] = stage == "Basic Pokémon"
                self.is_ace_spec[cid] = row["Rule"].strip() == "ACE SPEC"
        self.card_ids = sorted(self.name)
        self.vocab_size = CARD_OFFSET + max(self.card_ids) + 1

    def token(self, card_id):
        return card_id + CARD_OFFSET

    def card(self, token):
        cid = token - CARD_OFFSET
        return cid if cid in self.name else None
