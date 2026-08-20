"""V2: player token space + decision tokenizer.

COVERAGE CONTRACT: the token space is derived from the ENGINE SOURCE
(ptcgProgram 22: ApiType.h enums + ApiJson.h SelectOptionJson +
ToJson.h SelectJson/Current/PlayerJson/PokemonJson), NOT from whatever
sample replays happened to contain. Every field the engine can emit is
either tokenized here or explicitly listed as intentionally dropped.

--- SELECT LEVEL (ToJson.h SelectJson) -----------------------------
  type                 -> SELTYPE_*   (12 values, ApiType.h SelectType)
  context              -> SELCTX_*    (50 values, SelectContext)
  minCount             -> DECLINE span exists iff minCount == 0
  maxCount             -> NUM_B*
  remainDamageCounter  -> RDMG_B*
  remainEnergyCost     -> RENG_B*
  option[]             -> one OPT span each (below)
  deck                 -> resolved through options with area == Deck
  contextCard          -> CTXCARD + card token
  effect               -> EFFECT + card token

--- OPTION LEVEL (ApiJson.h SelectOptionJson), by type -------------
  0  Number            number                       -> NUM_B*
  1  Yes               -                            -> ATYPE_1
  2  No                -                            -> ATYPE_2
  3  Card              area,index,playerIndex       -> AREA_*, card, OWNER_*
  4  ToolCard          +toolIndex                   -> resolved card
  5  EnergyCard        +energyIndex                 -> resolved card
  6  Energy            +energyIndex,count           -> +CNT_B*
  7  Play              index                        -> resolved hand card
  8  Attach            area,index,inPlayArea,idx    -> card + target card
  9  Evolve            same                         -> card + target card
  10 Ability           area,index                   -> resolved card
  11 Discard           area,index                   -> resolved card
  12 Retreat           -                            -> ATYPE_12
  13 Attack            attackId                     -> ATK_<id>  (1556)
  14 End               -                            -> ATYPE_14
  15 Skill             cardId,serial                -> card token
  16 SpecialCondition  specialConditionType         -> SC_*

--- STATE LEVEL (Current / PlayerJson / PokemonJson) ----------------
  turn, turnActionCount, firstPlayer, supporterPlayed, stadiumPlayed,
  energyAttached, retreated, stadium, looking, and per player:
  active, bench, benchMax, deckCount, discard, prize, handCount, hand,
  poisoned/burned/asleep/paralyzed/confused; per pokemon: id, hp/maxHp,
  appearThisTurn, energies (types), energyCards, tools, preEvolution.

  Intentionally dropped (no decision value): serial (instance ids),
  card `name` (id already identifies it), log history.

Token ids are append-only: card tokens keep their V1 ids so a V1
builder checkpoint still loads.
"""

from card_vocab import Vocab, BOS, PLAY, CARD_OFFSET

N_ATYPES = 17     # SelectOptionType: Number..SpecialCondition
N_SELTYPES = 12   # SelectType: None..SpecialCondition
N_SELCTX = 50     # SelectContext: None..RecoverSpecialCondition
N_AREAS = 25      # AreaType: All..Temporary
N_SC = 5          # SelectSpecialConditionType
N_ATTACKS = 1557  # attackId 1..1556 (all_attack()), index 0 unused
N_ETYPES = 16     # EnergyTypeIndex slots
N_BUCKETS = 7     # counts: 0, 1-10, ... 51-60
N_HP = 11         # HP fraction in tenths
N_NUM = 8         # small-integer buckets (counts, numbers, remainders)

_FLAGS = ["ENERGY_DONE", "SUPPORTER_DONE", "STADIUM_DONE", "RETREAT_DONE",
          "I_AM_FIRST"]
_STATUS = ["ASLEEP", "PARALYZED", "CONFUSED", "BURNED", "POISONED"]


class PlayerVocab:
    """V1 card vocab + player specials. Token ids stable across versions."""

    def __init__(self, cards_csv=None):
        self.cards = Vocab(cards_csv) if cards_csv else Vocab()
        base = self.cards.vocab_size
        names = (["MY_ACTIVE", "MY_BENCH", "MY_HAND", "MY_DISCARD",
                  "OPP_ACTIVE", "OPP_BENCH", "OPP_DISCARD", "STADIUM",
                  "LOOKING", "MY_PRIZE_CARDS", "EFFECT", "CTXCARD",
                  "OPT", "UNKNOWN_CARD", "DECLINE",
                  "OWNER_MINE", "OWNER_OPP", "FRESH"]
                 + [f"ATYPE_{i}" for i in range(N_ATYPES)]
                 + [f"SELTYPE_{i}" for i in range(N_SELTYPES)]
                 + [f"SELCTX_{i}" for i in range(N_SELCTX)]
                 + [f"AREA_{i}" for i in range(N_AREAS)]
                 + [f"SC_{i}" for i in range(N_SC)]
                 + [f"ATK_{i}" for i in range(N_ATTACKS)]
                 + [f"ETYPE_{i}" for i in range(N_ETYPES)]
                 + [f"NUM_B{i}" for i in range(N_NUM)]
                 + [f"CNT_B{i}" for i in range(N_NUM)]
                 + [f"RDMG_B{i}" for i in range(N_NUM)]
                 + [f"RENG_B{i}" for i in range(N_NUM)]
                 + [f"HP_B{i}" for i in range(N_HP)]
                 + [f"TURN_B{i}" for i in range(8)]
                 + [f"TACT_B{i}" for i in range(N_NUM)]
                 + [f"MY_PRIZE_{i}" for i in range(7)]
                 + [f"OPP_PRIZE_{i}" for i in range(7)]
                 + [f"BENCHMAX_{i}" for i in range(9)]
                 + [f"MY_DECK_B{i}" for i in range(N_BUCKETS)]
                 + [f"OPP_DECK_B{i}" for i in range(N_BUCKETS)]
                 + [f"OPP_HAND_B{i}" for i in range(N_BUCKETS)]
                 + [f"MY_{s}" for s in _STATUS]
                 + [f"OPP_{s}" for s in _STATUS]
                 + _FLAGS)
        self.spec = {n: base + i for i, n in enumerate(names)}
        self.vocab_size = base + len(names)

    def card_token(self, card):
        """card dict (or None) -> token; hidden cards -> UNKNOWN_CARD."""
        cid = (card or {}).get("id")
        if cid and cid in self.cards.name:
            return cid + CARD_OFFSET
        return self.spec["UNKNOWN_CARD"]


# ---- tokenizer -----------------------------------------------------------
MAX_HAND = 24
MAX_DISCARD = 8      # most recent discards kept per side
MAX_LOOKING = 16
MAX_OPTIONS = 128   # engine 'Main' selects reach ~112 options
MAX_SEQ = 640


def _bucket(n, hi=6, per=10):
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0
    return min(hi, 1 + (n - 1) // per)


def _small(n, hi=None):
    """Small non-negative integer -> bucket index. Engine fields such as
    remainEnergyCost use -1 for 'not applicable', so clamp BOTH ends."""
    hi = N_NUM if hi is None else hi
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        n = 0
    return max(0, min(hi - 1, n))


def _clamp(n, hi):
    """Enum value -> valid index; unknown/out-of-range falls back to 0."""
    try:
        n = int(n if n is not None else 0)
    except (TypeError, ValueError):
        return 0
    return n if 0 <= n < hi else 0


def _stack(pk):
    out = [pk]
    for sub in ("energyCards", "tools", "preEvolution"):
        out.extend(c for c in (pk.get(sub) or []) if c)
    return out


def _get_pokemon(cur, pi, area, index):
    try:
        ps = cur["players"][pi]
        if area == 4:
            return (ps.get("active") or [])[index]
        if area == 5:
            return (ps.get("bench") or [])[index]
    except (IndexError, TypeError, KeyError):
        pass
    return None


def _option_card(pv, opt, sel, cur, me):
    """The card an option refers to (engine area/type semantics)."""
    try:
        t = opt.get("type")
        area = opt.get("area")
        idx = opt.get("index")
        pi = opt.get("playerIndex")
        pi = me if pi is None else pi
        ps = cur["players"][pi]
        if t == 7:      # Play from hand
            return (cur["players"][me].get("hand") or [])[opt["index"]]
        if t == 13:     # Attack -> the attacker
            act = cur["players"][me].get("active") or []
            return act[0] if act else None
        if t == 15:     # Skill
            return {"id": opt.get("cardId")}
        if t == 4:      # ToolCard on a pokemon
            p = _get_pokemon(cur, pi, area, idx)
            return (p.get("tools") or [])[opt.get("toolIndex") or 0] if p else None
        if t in (5, 6):  # EnergyCard / Energy on a pokemon
            p = _get_pokemon(cur, pi, area, idx)
            ecs = (p.get("energyCards") or []) if p else []
            ei = opt.get("energyIndex") or 0
            return ecs[ei] if ei < len(ecs) else None
        if t in (3, 8, 9, 10, 11):   # Card/Attach/Evolve/Ability/Discard
            if area == 1:            # Deck -> sel['deck']
                return (sel.get("deck") or [])[idx]
            if area == 2:            # Hand
                return (ps.get("hand") or [])[idx]
            if area == 3:            # Trash (discard)
                return (ps.get("discard") or [])[idx]
            if area in (4, 5):       # Active / Bench
                return _get_pokemon(cur, pi, area, idx)
            if area == 6:            # Prize
                return (ps.get("prize") or [])[idx]
            if area == 7:            # Stadium
                return (cur.get("stadium") or [])[idx]
            if area == 12:           # Looking
                return (cur.get("looking") or [])[idx]
    except (IndexError, TypeError, KeyError, AttributeError):
        pass
    return None


def tokenize_decision(pv, cur, sel):
    """One decision -> (tokens, option_end_positions) or None.

    tokens: int list, <= MAX_SEQ. option_end_positions[i] = index of the
    last token of option i's span (where the model reads its score).

    When minCount == 0 the engine also allows taking NOTHING. That is a
    real move, so it gets its own DECLINE span appended after the
    engine's options; its index is len(options). Callers must map a win
    at that index to the empty selection [].
    """
    me = cur["yourIndex"]
    opp = 1 - me
    players = cur.get("players") or []
    if len(players) < 2:
        return None
    mine, theirs = players[me], players[opp]
    S = pv.spec

    # ---- select header: what kind of choice is this, and what for ----
    toks = [BOS, PLAY,
            S[f"SELTYPE_{_clamp(sel.get('type'), N_SELTYPES)}"],
            S[f"SELCTX_{_clamp(sel.get('context'), N_SELCTX)}"],
            S[f"NUM_B{_small(sel.get('maxCount'))}"],
            S[f"RDMG_B{_small(sel.get('remainDamageCounter'))}"],
            S[f"RENG_B{_small(sel.get('remainEnergyCost'))}"],
            S[f"TURN_B{_small(_small(cur.get('turn'), 10 ** 6) // 5, 8)}"],
            S[f"TACT_B{_small(cur.get('turnActionCount'))}"],
            S[f"MY_PRIZE_{_small(len(mine.get('prize') or []), 7)}"],
            S[f"OPP_PRIZE_{_small(len(theirs.get('prize') or []), 7)}"],
            S[f"MY_DECK_B{_bucket(mine.get('deckCount'))}"],
            S[f"OPP_DECK_B{_bucket(theirs.get('deckCount'))}"],
            S[f"OPP_HAND_B{_bucket(theirs.get('handCount'))}"],
            S[f"BENCHMAX_{_small(mine.get('benchMax'), 9)}"]]
    if cur.get("firstPlayer") == me:
        toks.append(S["I_AM_FIRST"])
    for flag, key in zip(_FLAGS[:4], ("energyAttached", "supporterPlayed",
                                      "stadiumPlayed", "retreated")):
        if cur.get(key):
            toks.append(S[flag])
    for side_tag, ps in (("MY", mine), ("OPP", theirs)):
        for st in _STATUS:
            if ps.get(st.lower()):
                toks.append(S[f"{side_tag}_{st}"])

    # which card's effect is resolving, and what triggered the choice
    if sel.get("effect"):
        toks.append(S["EFFECT"])
        toks.append(pv.card_token(sel["effect"]))
    if sel.get("contextCard"):
        toks.append(S["CTXCARD"])
        toks.append(pv.card_token(sel["contextCard"]))

    def hp_token(pk):
        def _f(x):
            try:
                return float(x or 0)
            except (TypeError, ValueError):
                return 0.0
        mh = _f(pk.get("maxHp"))
        frac = (_f(pk.get("hp")) / mh) if mh > 0 else 1.0
        return S[f"HP_B{_small(round(frac * 10), N_HP)}"]

    def emit_pokemon(pk):
        toks.append(pv.card_token(pk))
        toks.append(hp_token(pk))
        if pk.get("appearThisTurn"):
            toks.append(S["FRESH"])
        for e in (pk.get("energies") or []):       # resolved energy TYPES
            toks.append(S[f"ETYPE_{_clamp(e, N_ETYPES)}"])
        for c in _stack(pk)[1:]:                   # energy cards/tools/pre-evo
            toks.append(pv.card_token(c))

    def emit_side(tag, ps):
        stacks = []
        for grp in ("active", "bench"):
            for pk in (ps.get(grp) or []):
                if pk:
                    stacks.append((grp, pk))
        toks.append(S[f"{tag}_ACTIVE"])
        for grp, pk in stacks:
            if grp == "active":
                emit_pokemon(pk)
        toks.append(S[f"{tag}_BENCH"])
        for grp, pk in stacks:
            if grp == "bench":
                emit_pokemon(pk)

    emit_side("MY", mine)
    toks.append(S["MY_HAND"])
    toks.extend(pv.card_token(c) for c in (mine.get("hand") or [])[:MAX_HAND])
    toks.append(S["MY_DISCARD"])
    toks.extend(pv.card_token(c)
                for c in (mine.get("discard") or [])[-MAX_DISCARD:])
    revealed_prizes = [c for c in (mine.get("prize") or []) if c]
    if revealed_prizes:
        toks.append(S["MY_PRIZE_CARDS"])
        toks.extend(pv.card_token(c) for c in revealed_prizes)
    emit_side("OPP", theirs)
    toks.append(S["OPP_DISCARD"])
    toks.extend(pv.card_token(c)
                for c in (theirs.get("discard") or [])[-MAX_DISCARD:])
    toks.append(S["STADIUM"])
    toks.extend(pv.card_token(c) for c in (cur.get("stadium") or []) if c)
    looking = [c for c in (cur.get("looking") or []) if c]
    if looking:
        toks.append(S["LOOKING"])
        toks.extend(pv.card_token(c) for c in looking[:MAX_LOOKING])

    # ---- the options themselves ----
    options = sel.get("option") or []
    if len(options) < 2 or len(options) > MAX_OPTIONS:
        return None
    opt_pos = []
    for opt in options:
        t = opt.get("type")
        toks.append(S["OPT"])
        toks.append(S[f"ATYPE_{_clamp(t, N_ATYPES)}"])
        if t == 13:                                   # WHICH attack
            toks.append(S[f"ATK_{_clamp(opt.get('attackId'), N_ATTACKS)}"])
        if opt.get("area") is not None:
            toks.append(S[f"AREA_{_clamp(opt.get('area'), N_AREAS)}"])
        pi = opt.get("playerIndex")
        if pi is not None:                            # whose card is targeted
            toks.append(S["OWNER_MINE" if pi == me else "OWNER_OPP"])
        if opt.get("specialConditionType") is not None:
            toks.append(S[f"SC_{_clamp(opt.get('specialConditionType'), N_SC)}"])
        if opt.get("number") is not None:
            toks.append(S[f"NUM_B{_small(opt.get('number'))}"])
        if opt.get("count") is not None:
            toks.append(S[f"CNT_B{_small(opt.get('count'))}"])
        card = _option_card(pv, opt, sel, cur, me)
        if card is not None:
            toks.append(pv.card_token(card))
        if opt.get("inPlayArea") is not None:         # attach/evolve target
            tgt = _get_pokemon(cur, me, opt["inPlayArea"],
                               opt.get("inPlayIndex") or 0)
            if tgt is not None:
                toks.append(pv.card_token(tgt))
                toks.append(hp_token(tgt))
        opt_pos.append(len(toks) - 1)

    # "take nothing" is a legal move whenever minCount == 0 -> its own span
    if (sel.get("minCount") or 0) == 0:
        toks.append(S["OPT"])
        toks.append(S["DECLINE"])
        opt_pos.append(len(toks) - 1)

    if len(toks) > MAX_SEQ:
        return None
    return toks, opt_pos
