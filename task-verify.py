#!/usr/bin/env python3
"""
Objective pass/fail gate for the retrieval task.

Reads the model's stdout, extracts the three answers, and checks each against
the key with exact matching. Prints a single VERDICT line that the shell
runner greps for.

Exit 0 = task completed correctly, 1 = wrong/undetermined.
"""
import pathlib, re, sys, json

OUT = pathlib.Path(__file__).parent
tag = sys.argv[1] if len(sys.argv) > 1 else "smoke"
# Optional 3rd arg: which side's output to grade. The paired runner writes
# task-out-<tag>.<side>.txt because stock and fork overwrite each other
# otherwise -- grading the wrong side would silently report a pass.
side = sys.argv[2] if len(sys.argv) > 2 else None
if side:
    outfile = OUT / f"task-out-{tag}.{side}.txt"
else:
    outfile = OUT / f"task-out-{tag}.txt"

key = [w.strip().lower() for w in
       (OUT / f"task-key-{tag}.txt").read_text(encoding="utf-8").split() if w.strip()]

if not outfile.exists():
    print(f"VERDICT: FAIL no-output {tag}")
    sys.exit(1)

# A file that EXISTS but is EMPTY (or holds only whitespace) means the model
# produced no answer at all. This is NOT a pass. Before this check, a 0-byte
# capture fell through to the parser, yielded `got: []`, and the paired
# benchmark still logged "PASS 3/3" -- six fabricated verdicts across three
# pairs, all of them model-load failures dressed up as retrieval successes.
# An empty run must fail as loudly as a missing one.
_raw = outfile.read_text(errors="replace")
if not _raw.strip():
    print(f"VERDICT: FAIL empty-output {tag} side={side or 'solo'}")
    print("expected: " + repr(key))
    print("got:      []  (output file is empty -- the model never answered)")
    print(f"correct:  0/{len(key)}")
    sys.exit(1)

text = outfile.read_text(errors="replace")
# Drop the echoed prompt: the context contains every word, so a naive match
# against the whole file would always "succeed".
#
# The marker must match what task-gen.py actually writes: the archive is
# wrapped as "ARCHIVE OF FIELD OBSERVATIONS -- BEGIN" / "... -- END". The
# earlier literal omitted the middle words, so the split NEVER matched and
# the grader was reading the whole echoed archive.
MARK = "ARCHIVE OF FIELD OBSERVATIONS -- END"
# The prompt echo is TRUNCATED by llama-cli ("R0023 pereg ... (truncated)"), so
# the END marker is usually ABSENT from stdout. Anchoring on the truncation
# notice instead, and falling back to the last non-empty content line. A gate
# that cannot parse a correct answer is worse than no gate: it would report
# FAIL for a run that plainly succeeded.
TRUNC = re.compile(r"\.\.\. \(truncated\)")
gen = text
if MARK in text:
    gen = text.split(MARK, 1)[1]
else:
    m = TRUNC.search(text)
    gen = text[m.end():] if m else text
# Drop llama's own trailer chrome: "[ Prompt: x t/s | Generation: y t/s ]",
# "Exiting...", the prompt echo marker.
gen = re.sub(r"\[ *Prompt:.*?\]", " ", gen, flags=re.S)
gen = re.sub(r"(?im)^\s*(Exiting\.\.\.|>)\s*$", " ", gen)

# The model was told to answer with one word per question. Look for the
# question markers first, then fall back to bare-word scanning.
answers, found = [], []
blocks = re.split(r"Question\s*\d+\s*:", gen)[1:]
if len(blocks) == len(key):
    for b in blocks:
        b = b.split("Question", 1)[0]
        words = re.findall(r"[A-Za-z]{4,}", b.lower())
        answers.append(words[0] if words else "")
else:
    # Model answered inline, one bare word per line. Take the FIRST len(key)
    # word-like tokens of the generation, not the last: the trailer chrome is
    # stripped above, and trailing tokens would otherwise be llama's.
    words = re.findall(r"[A-Za-z]{4,}", gen.lower())
    answers = words[:len(key)] if len(words) >= len(key) else []
    found.append("inline")

ok = sum(1 for a, k in zip(answers, key) if a == k)
passed = ok == len(key) and len(key) > 0

print(f"expected: {key}")
print(f"got:      {answers}  mode={found or 'marker'}")
print(f"correct:  {ok}/{len(key)}")
print(f"VERDICT: {'PASS' if passed else 'FAIL'} correct={ok}/{len(key)} tag={tag} side={side or 'solo'}")
sys.exit(0 if passed else 1)
