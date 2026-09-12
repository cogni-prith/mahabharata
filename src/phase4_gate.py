"""Phase 4 gate. Precision on the 300-triple sample, by predicate.

The verdicts below were made by reading the evidence sentence of every sampled
triple. The judge is the author of the extraction patterns, so this is not an
independent evaluation and is labelled as such in the report. It is a precision
measurement against Debroy's prose, not an inter-annotator agreement figure.

A triple counts as correct when the cited sentence asserts it. Sentences that
report a boast, a conditional or a prophecy that the epic does not fulfil count
as wrong, because the extraction presents them as assertions.
"""
import pandas as pd

WRONG = {
    # bearer taken from a comma list of names the parser could not segment
    810, 522, 464, 333, 630, 389, 290,
    295, 387, 479, 360, 519, 323, 410, 325, 142, 477,
    # apposition attached across a clause boundary
    175,
    # a claim, a conditional or a boast, not an assertion
    691, 624, 619,
    # context is the front-matter shloka count table, not narrative
    657,
    # "O Bhagavan" is a title, not a name
    839,
}

s = pd.read_csv("build/phase4_precision_sample.csv")
s["verdict"] = ["wrong" if i in WRONG else "correct" for i in s.row_id]
s.to_csv("build/phase4_precision_sample.csv", index=False)

rel = pd.read_parquet("build/relations.parquet")
counts = rel.predicate.value_counts()

rows, num, den = [], 0.0, 0
for p, g in s.groupby("predicate"):
    ok = int((g.verdict == "correct").sum())
    rows.append((p, len(g), ok, ok / len(g), int(counts[p])))
    num += ok / len(g) * counts[p]
    den += counts[p]

with open("build/phase4_gate.md", "w", encoding="utf-8") as f:
    f.write("# Phase 4 gate: precision on a 300-triple sample\n\n")
    f.write("Sample drawn by `src/phase4_sample.py`, seed 20260912, stratified by\n"
            "predicate. Verdicts in `build/phase4_precision_sample.csv`.\n\n")
    f.write("Not an independent evaluation. The person who judged the sample wrote\n"
            "the extraction patterns. It measures whether the cited sentence asserts\n"
            "the triple, which is what the gate asks, and nothing about agreement.\n\n")
    f.write("| predicate | sampled | correct | precision | instances in corpus |\n")
    f.write("|---|---|---|---|---|\n")
    for p, n, ok, prec, tot in rows:
        f.write("| %s | %d | %d | %.3f | %d |\n" % (p, n, ok, prec, tot))
    f.write("| **sample total** | %d | %d | **%.3f** | %d |\n"
            % (len(s), int((s.verdict == 'correct').sum()),
               (s.verdict == 'correct').mean(), den))
    f.write("\nPrecision weighted by corpus instance count: **%.3f**. The sample is\n"
            "stratified, so the unweighted sample total over-represents the small\n"
            "predicates and the weighted figure is the one to quote.\n" % (num / den))
    f.write("\n## What the remaining errors are\n\n"
            "Every error in the sample falls into one of four classes.\n\n"
            "1. **Comma lists of names (17 of 23).** Debroy renders a muster roll as\n"
            "   `Duryodhana, Drona, Shakuni, Vrishasena, the king of Madra, you,\n"
            "   Somadatta`. The role phrase is a further list member, but the parser\n"
            "   attaches it as an apposition to whichever name precedes it. A guard\n"
            "   now drops appositions whose bearer is a conjunct, which removed most\n"
            "   of these and cost recall; what survives is lists the parser did not\n"
            "   mark as coordination at all. This is the dominant residual error and\n"
            "   it concentrates in `rules` and `hasBiologicalParent`.\n"
            "2. **Apposition across a clause boundary (1).** `arrows sliced down by\n"
            "   Pradyumna, the king of Soubha resorted to maya` reads the king of\n"
            "   Soubha as Pradyumna. He is Shalva.\n"
            "3. **Unasserted propositions (3).** A boast (`Karna will kill Arjuna`),\n"
            "   a disjunction (`either Karna has killed the two Krishnas, or`) and a\n"
            "   conditional. Negation and comparatives are guarded; these forms are\n"
            "   not. They are a small share here but they are exactly the class that\n"
            "   a knowledge graph must not silently assert.\n"
            "4. **Non-narrative context (1) and a title read as a name (1).**\n")
print("weighted=%.4f sample=%.4f" % (num / den, (s.verdict == "correct").mean()))
