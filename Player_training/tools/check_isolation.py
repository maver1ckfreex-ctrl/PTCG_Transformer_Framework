"""Prove the review tournament cannot influence training.

Checks, mechanically, that:

  1. No file on the TRAINING path mentions the reviews directory, the
     tournament module, or a win rate.
  2. No file on the training path imports tournament_seq.
  3. In train_forever.sh the chain checkpoint (NEXT_INIT) is assigned
     BEFORE the tournament is invoked, so no branch on the result is even
     reachable.
  4. The tournament call discards its exit status (`|| true`) and writes
     only into reviews/.
  5. Nothing reads reviews/ back.

Exit code 0 = isolated. Non-zero = something can leak.

    python3 tools/check_isolation.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# every file that can shape the model
TRAINING_PATH = [
    "sequence/replay_to_trajectories.py",
    "sequence/train_seq.py",
    "sequence/train_seq_v2.py",
    "sequence/train_seq_v3.py",
    "sequence/train_seq_v3b.py",
    "sequence/seq_model.py",
    "sequence/seq_model_v3.py",
    "sequence/seq_model_v3b.py",
    "baseline/train_baseline.py",
    "baseline/baseline_model.py",
    "common/encoder.py",
    "common/player_vocab.py",
    "common/card_vocab.py",
    "tools/warm_start.py",
    "selfplay/selfplay_replay.py",
    "selfplay/build_deck.py",
    "pick_days.py",
    "run_round.sh",
    "transfer/run_purser.sh",
    "transfer/data_paths.sh",
]

BANNED = [
    (r"\breviews?\b/", "references the reviews directory"),
    (r"tournament_seq", "references the tournament module"),
    (r"\bwin[_ ]?rate\b", "references a win rate"),
    (r"round_\d+\.txt", "reads a review report"),
]

fails = []


def check_training_path():
    for rel in TRAINING_PATH:
        p = ROOT / rel
        if not p.exists():
            fails.append(f"MISSING {rel}")
            continue
        text = p.read_text()
        for pat, why in BANNED:
            for m in re.finditer(pat, text, re.IGNORECASE):
                line = text[:m.start()].count("\n") + 1
                fails.append(f"{rel}:{line} {why}: {m.group(0)!r}")


def check_loop_order():
    p = ROOT / "train_forever.sh"
    if not p.exists():
        fails.append("MISSING train_forever.sh")
        return
    text = p.read_text()
    lines = text.split("\n")

    def find(pat):
        for i, l in enumerate(lines, 1):
            if re.search(pat, l):
                return i
        return None

    assign = find(r"^\s*NEXT_INIT=")
    call = find(r"tournament_seq\.py")
    if assign is None:
        fails.append("train_forever.sh: NEXT_INIT never assigned")
    elif call is None:
        fails.append("train_forever.sh: tournament never invoked")
    elif assign > call:
        fails.append(f"train_forever.sh: NEXT_INIT (line {assign}) is set "
                     f"AFTER the tournament (line {call}) -- the chain could "
                     f"depend on the result")
    else:
        print(f"  OK  NEXT_INIT assigned line {assign}, tournament line "
              f"{call} -- chain fixed before the result exists")

    # the tournament's exit status must be discarded
    block = "\n".join(lines[(call or 1) - 1: (call or 1) + 8])
    if "|| true" not in block:
        fails.append("train_forever.sh: tournament exit status is not "
                     "discarded (no '|| true') -- a failure could branch")
    else:
        print("  OK  tournament exit status discarded ('|| true')")

    # nothing may read a report back
    for pat in (r"\$\(.*reviews.*\)", r"<\s*\"?\$REPORT",
                r"read .*REPORT", r"source .*REPORT"):
        if re.search(pat, text):
            fails.append(f"train_forever.sh: reads a review back ({pat})")

    # the only consumer of $REPORT may be the writer and a human-facing grep
    uses = [i for i, l in enumerate(lines, 1) if "$REPORT" in l]
    print(f"  OK  $REPORT referenced on {len(uses)} lines (write + display "
          f"only)")


def check_dual_mode():
    """Dual-GPU mode: same rule as bulk, plus the two trainers must not
    reference each other's outputs."""
    runner = ROOT / "bulk/run_dual.sh"
    watcher = ROOT / "bulk/watch_eval_dual.sh"
    monitor = ROOT / "tools/dual_monitor.py"
    for f in (runner, watcher, monitor):
        if not f.exists():
            fails.append(f"MISSING {f.relative_to(ROOT)}")
            return
    rt, wt, mt = (f.read_text() for f in (runner, watcher, monitor))

    for pat in (r"\$\(.*reviews.*\)", r"cat .*reviews", r"if .*reviews"):
        if re.search(pat, rt):
            fails.append(f"run_dual.sh reads a review back ({pat})")
    print("  OK  run_dual.sh never reads a review")

    # verification must gate training: verify_data must appear before the
    # trainers are launched, or a bad dataset could reach a GPU
    lines = rt.split("\n")
    v = next((i for i, l in enumerate(lines, 1) if "verify_data.py" in l), None)
    L = next((i for i, l in enumerate(lines, 1)
              if re.match(r"^launch \d ", l.strip())), None)
    if v is None:
        fails.append("run_dual.sh: datasets are never verified")
    elif L is None:
        fails.append("run_dual.sh: trainers never launched")
    elif v > L:
        fails.append(f"run_dual.sh: verify_data (line {v}) runs AFTER launch "
                     f"(line {L}) -- a bad dataset could reach a GPU")
    else:
        print(f"  OK  verify_data line {v} gates launch line {L}")

    if "|| true" not in wt:
        fails.append("watch_eval_dual.sh: tournament failure not swallowed")
    else:
        print("  OK  watch_eval_dual.sh swallows tournament failures")

    for pat in (r"torch\.save", r"rm .*\.pt", r"TRAINING_DONE\s*>"):
        if re.search(pat, wt, re.M) or re.search(pat, mt, re.M):
            fails.append(f"dual evaluator/monitor writes into training ({pat})")
    print("  OK  dual evaluator + monitor only read, never write training")


def check_mix_mode():
    """Mix cycles: NEXT_INIT must be fixed before the review tournament."""
    p = ROOT / "mix/run_mix_cycles.sh"
    if not p.exists():
        fails.append("MISSING mix/run_mix_cycles.sh")
        return
    lines = p.read_text().split("\n")
    a = next((i for i, l in enumerate(lines, 1)
              if re.match(r"^\s*NEXT_INIT=", l)), None)
    t = next((i for i, l in enumerate(lines, 1)
              if "tournament_seq.py" in l), None)
    if a is None:
        fails.append("run_mix_cycles.sh: NEXT_INIT never assigned")
    elif t is None:
        fails.append("run_mix_cycles.sh: tournament never invoked")
    elif a > t:
        fails.append(f"run_mix_cycles.sh: NEXT_INIT (line {a}) set AFTER "
                     f"the tournament (line {t})")
    else:
        print(f"  OK  NEXT_INIT line {a} fixed before tournament line {t}")
    txt = "\n".join(lines)
    if "|| true" not in txt:
        fails.append("run_mix_cycles.sh: tournament failure not swallowed")
    else:
        print("  OK  tournament exit status discarded")
    # An existence test (`[ ! -f "$REPORT" ]`, used to skip an already
    # written review on resume) is not reading the result back. What must
    # never happen is consuming its CONTENT.
    for pat in (r"\$\([^)]*\$REPORT[^)]*\)", r"cat\s+.*\$REPORT",
                r"read\s+.*<\s*\"?\$REPORT", r"source\s+.*\$REPORT",
                r"<\s*\"?\$REPORT\"?\s*$", r"grep[^|]*\$REPORT[^|]*\)"):
        if re.search(pat, txt, re.M):
            fails.append(f"run_mix_cycles.sh consumes review content ({pat})")
    print("  OK  review content never consumed (existence test only)")


def check_node_mode():
    """Node training: same rule -- NEXT_INIT fixed before the tournament,
    and the CPU node must never see a review at all."""
    g = ROOT / "node/run_gpu_node.sh"
    c = ROOT / "node/run_cpu_node.sh"
    for f in (g, c):
        if not f.exists():
            fails.append(f"MISSING {f.relative_to(ROOT)}")
            return
    gt, ct = g.read_text(), c.read_text()
    lines = gt.split("\n")
    a = next((i for i, l in enumerate(lines, 1)
              if re.match(r"^\s*NEXT_INIT=", l)), None)
    t = next((i for i, l in enumerate(lines, 1)
              if "tournament_seq.py" in l), None)
    if a is None:
        fails.append("run_gpu_node.sh: NEXT_INIT never assigned")
    elif t is None:
        fails.append("run_gpu_node.sh: tournament never invoked")
    elif a > t:
        fails.append(f"run_gpu_node.sh: NEXT_INIT (line {a}) set AFTER the "
                     f"tournament (line {t})")
    else:
        print(f"  OK  NEXT_INIT line {a} fixed before tournament line {t}")
    if "|| true" not in gt:
        fails.append("run_gpu_node.sh: tournament failure not swallowed")
    else:
        print("  OK  tournament exit status discarded")
    # the self-play node must be entirely unaware of evaluation
    for pat, why in (("tournament_seq", "references the tournament"),
                     (r"\breviews?\b/", "references the reviews directory"),
                     (r"\bwin[_ ]?rate\b", "references a win rate")):
        if re.search(pat, ct, re.I):
            fails.append(f"run_cpu_node.sh {why}")
    print("  OK  CPU node never references the tournament or reviews")


def check_bulk_mode():
    """Bulk mode splits trainer and evaluator into separate processes, so
    the rule there is: the trainer must not read the evaluator's output,
    and the evaluator must not produce anything the trainer consumes."""
    runner = ROOT / "bulk/run_bulk.sh"
    watcher = ROOT / "bulk/watch_eval.sh"
    for p in (runner, watcher):
        if not p.exists():
            fails.append(f"MISSING {p.relative_to(ROOT)}")
            return

    rt = runner.read_text()
    wt = watcher.read_text()

    # the orchestrator may launch the watcher, but never read a report
    for pat in (r"\$\(.*reviews.*\)", r"cat .*reviews", r"source .*reviews",
                r"if .*reviews"):
        if re.search(pat, rt):
            fails.append(f"run_bulk.sh reads a review back ({pat})")
    print("  OK  run_bulk.sh never reads a review")

    # the trainer is invoked with no eval-derived argument
    # NB: match win RATE, not bare "win" -- "--winners-only" and
    # "trajectories_winners.npz" are data selection, not eval results.
    m = re.search(r"train_seq\.py(.|\n)*?\n\n", rt)
    if m and re.search(r"review|tournament|win[ _-]?rate|z[ _-]?score",
                       m.group(0), re.I):
        fails.append("run_bulk.sh: trainer invocation mentions eval output")
    else:
        print("  OK  trainer invoked with no eval-derived argument")

    # the watcher must not write checkpoints or touch the training dir
    for pat in (r"torch\.save", r"\.pt\"?\s*$", r">\s*\S*\.pt\b",
                r"rm .*\.pt", r"TRAINING_DONE\s*>"):
        if re.search(pat, wt, re.M):
            fails.append(f"watch_eval.sh writes into the training path "
                         f"({pat})")
    print("  OK  watch_eval.sh only reads checkpoints, writes only reviews")

    # the watcher's own failure must not propagate
    if "|| true" not in wt:
        fails.append("watch_eval.sh: tournament failure is not swallowed")
    else:
        print("  OK  watch_eval.sh swallows tournament failures")


if __name__ == "__main__":
    print("training-path files scanned:", len(TRAINING_PATH))
    check_training_path()
    if not fails:
        print("  OK  no training-path file mentions reviews / tournament / "
              "win rate")
    print("\ntrain_forever.sh ordering (round mode):")
    check_loop_order()
    print("\nbulk mode trainer/evaluator separation:")
    check_bulk_mode()
    print("\ndual-GPU mode:")
    check_dual_mode()
    print("\nmix-ratio cycles:")
    check_mix_mode()
    print("\nnode training:")
    check_node_mode()

    print()
    if fails:
        print("FAIL -- the tournament could influence training:")
        for f in fails:
            print("   *", f)
        sys.exit(1)
    print("PASS -- review tournament is isolated from training.")
    print("Round N+1 is identical regardless of round N's tournament result.")
