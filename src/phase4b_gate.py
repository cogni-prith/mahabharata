"""Report precision on the coreference-derived triples.

The sample was drawn by `src/phase4b_sample.py` against the first coreference run
and judged sentence by sentence against Debroy's prose. The judgements are in
`build/phase4b_judged.csv`, keyed on the triple rather than on a row number so
they survive a re-run of the extractor.

Two guards were added after judging, both root causes rather than symptoms, and
both subtractive: they only remove triples. Precision after each is therefore
measurable on the judgements already made, by scoring the sample rows that the
guard did not remove. That is not a fresh sample and it is not an independent
evaluation. It is the same 199 judgements, re-read against a smaller population.

The judgement was made by the author of the patterns. Same caveat as D14.
"""
import pandas as pd

KEY = ["predicate", "subject", "object", "pattern", "parva", "adhyaya_ordinal"]

j = pd.read_csv("build/phase4b_judged.csv")
r = pd.read_parquet("build/relations.parquet")
cor = r[r.coref_subject.notna() | r.coref_object.notna()]
direct = r[r.coref_subject.isna() & r.coref_object.isna()]
m = j.merge(cor[KEY].drop_duplicates(), on=KEY)

L = ["# Phase 4b gate: precision of the coreference-derived triples", "",
     "199 instances judged, stratified by predicate, seed 20260914. Judged by the",
     "author of the patterns, so this is a self-assessment and not an independent",
     "evaluation.", "",
     "## Where the measurement went", "",
     "| stage | coreference triples | sample rows surviving | precision |",
     "|---|---|---|---|",
     "| as first run, first and second person included | 686 | 244 | 0.443 |",
     "| first and second person dropped, non-assertive clauses guarded, preceptor direction fixed | 401 | 199 | 0.513 |",
     "| possessor-headed mentions dropped | 334 | 183 | 0.557 |",
     "| non-person argument heads and plurals dropped | %d | %d | %.3f |" %
     (len(cor), len(m), (m.verdict == "correct").mean()), "",
     "## By predicate, final", "",
     "| predicate | judged | correct | precision |", "|---|---|---|---|"]
for p, g in m.groupby("predicate"):
    L.append("| %s | %d | %d | %.3f |" % (p, len(g), (g.verdict == "correct").sum(),
                                          (g.verdict == "correct").mean()))

w = 0.0
for p, g in m.groupby("predicate"):
    w += len(cor[cor.predicate == p]) * (g.verdict == "correct").mean()
L += ["", "## Verdict", "",
      "- Coreference-derived: %d instances, weighted precision %.3f" % (len(cor), w / len(cor)),
      "- Directly read: %d instances, 0.931 from `build/phase4_gate.md`" % len(direct),
      "- Combined: %d instances, %.3f" % (len(r), (w + 0.931 * len(direct)) / len(r)), "",
      "The coreference tier does not reach the quality of the directly read tier and",
      "is not close. It is kept in the graph because every one of its triples carries",
      "`mbh:argumentsByCoreference` and a discounted confidence, so it can be filtered",
      "out, and it is filtered out of any precision claim made about the graph. The",
      "errors that remain are wrong antecedents, which no guard on the extractor side",
      "can catch: 'the lord of the gods himself killed Vritra' resolved to Maheshvara",
      "rather than Indra, and first person 'me' inside a quoted speech resolved to the",
      "frame narrator. LingMess has no notion of a diegetic level, and this corpus is",
      "four levels of nested speech deep.", "",
      "Against the stated purpose, coreference failed. D15 projected 3,000 to 5,000",
      "triples from this phase. It produced %d at 0.6 precision." % len(cor)]
open("build/phase4b_gate.md", "w").write("\n".join(L) + "\n")
print("\n".join(L[-8:]))
