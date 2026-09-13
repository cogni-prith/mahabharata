"""Score the predicate-expansion precision sample and report Phase 4 again.

Judgements are in build/phase4e_wrong.txt, which lists the row_ids judged
false; every other sampled row is judged true. They were made by reading the
evidence span, not the labels. A row whose arguments the sentence does not
settle is counted false, so a noun-phrase link the sentence cannot confirm
counts against the link even where the graph may be right.
"""
import pandas as pd

s = pd.read_csv("build/phase4e_precision_sample.csv")
wrong = {int(x) for x in open("build/phase4e_wrong.txt").read().split()}
s["ok"] = ~s.row_id.isin(wrong)
s["source"] = ["nplink" if (pd.notna(a) or pd.notna(b)) else
               "coref" if (pd.notna(c) or pd.notna(d)) else "direct"
               for a, b, c, d in zip(s.nplink_subject, s.nplink_object,
                                     s.coref_subject, s.coref_object)]

rel = pd.read_parquet("build/relations.parquet")
L = ["# Phase 4e gate: the predicate expansion", "",
     "Sample drawn by `src/phase4e_sample.py`, seed 20260915, %d rows." % len(s),
     "Only the patterns the expansion added, plus every noun-phrase-linked row.",
     "The patterns judged earlier are not redrawn.", "",
     "| stratum | n | precision |", "|---|---|---|"]
for k, g in s.groupby("pattern"):
    L.append("| %s | %d | %.3f |" % (k, len(g), g.ok.mean()))
L.append("| **all** | %d | **%.3f** |" % (len(s), s.ok.mean()))
L += ["", "## By argument provenance", "", "| source | n | precision |", "|---|---|---|"]
for k, g in s.groupby("source"):
    L.append("| %s | %d | %.3f |" % (k, len(g), g.ok.mean()))
L += ["", "## Corpus counts after the expansion", "",
      "| predicate | instances | distinct triples |", "|---|---|---|"]
dis = rel.drop_duplicates(["subject", "predicate", "object"]).predicate.value_counts()
for k, v in rel.predicate.value_counts().items():
    L.append("| %s | %d | %d |" % (k, v, dis[k]))
L.append("| **total** | %d | %d |" % (len(rel), dis.sum()))
open("build/phase4e_gate.md", "w").write("\n".join(L) + "\n")
print(s.ok.mean(), len(s))
