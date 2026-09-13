"""Phase 4 gate: precision on the current graph, from the judgements on record.

Two hand-judged sets are carried: build/phase4e_precision_sample.csv with
build/phase4e_wrong.txt, 250 rows over the patterns the predicate expansion
added, and build/phase4g_judged.csv, 64 rows over the noun-phrase-linked
arguments. Both are keyed on the triple and its pattern rather than on a row
number, so a judgement survives a re-run of the extractor and a guard can be
scored against work already done. Rows the guards have since removed drop out
and are reported separately, which is the only measurement that says whether a
guard removed errors or removed evidence.
"""
import pandas as pd

K = ["subject", "predicate", "object", "pattern"]
rel = pd.read_parquet("build/relations.parquet")

e = pd.read_csv("build/phase4e_precision_sample.csv")
e["ok"] = ~e.row_id.isin({int(x) for x in open("build/phase4e_wrong.txt").read().split()})
g = pd.read_csv("build/phase4g_judged.csv")
j = pd.concat([e[K + ["ok"]], g[K + ["ok"]]]).drop_duplicates(K)

cur = rel[K].drop_duplicates().assign(present=True)
m = j.merge(cur, on=K, how="left")
m["present"] = m.present.fillna(False)
live = m[m.present]

src = rel.assign(source=[
    "nplink" if (pd.notna(a) or pd.notna(b)) else
    "coref" if (pd.notna(c) or pd.notna(d)) else "direct"
    for a, b, c, d in zip(rel.nplink_subject, rel.nplink_object,
                          rel.coref_subject, rel.coref_object)])
live = live.merge(src[K + ["source"]].drop_duplicates(K), on=K, how="left")

L = ["# Phase 4 precision on the current graph", "",
     "Judgements on record: %d. Still present after the guards: %d."
     % (len(m), len(live)),
     "Removed by the guards: %d judged false, %d judged true."
     % (int(((~m.ok) & (~m.present)).sum()), int((m.ok & (~m.present)).sum())),
     "", "| pattern | judged | precision |", "|---|---|---|"]
for k, x in live.groupby("pattern"):
    L.append("| %s | %d | %.3f |" % (k, len(x), x.ok.mean()))
L.append("| **all** | %d | **%.3f** |" % (len(live), live.ok.mean()))
L += ["", "## By argument provenance", "", "| source | judged | precision |", "|---|---|---|"]
for k, x in live.groupby("source"):
    L.append("| %s | %d | %.3f |" % (k, len(x), x.ok.mean()))
L += ["", "## Corpus counts", "", "| predicate | instances | distinct |", "|---|---|---|"]
dis = rel.drop_duplicates(["subject", "predicate", "object"]).predicate.value_counts()
for k, v in rel.predicate.value_counts().items():
    L.append("| %s | %d | %d |" % (k, v, dis[k]))
L.append("| **total** | %d | %d |" % (len(rel), int(dis.sum())))
open("build/phase4_precision.md", "w").write("\n".join(L) + "\n")
print("judged=%d live=%d precision=%.3f" % (len(m), len(live), live.ok.mean()))
