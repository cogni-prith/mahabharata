"""Phase 4 gate: precision on the current graph, from the judgements on record.

Two hand-judged sets are carried: annotations/phase4e_judged.csv, 250 rows over
the patterns the predicate expansion added, and annotations/phase4g_judged.csv,
64 rows over the noun-phrase-linked arguments.

The 4e set used to be 59 row ids in annotations/phase4e_wrong.txt resolved
against build/phase4e_precision_sample.csv. A row id is a position in
build/relations.parquet, so the resolution was only correct while that file did
not change, and D21 changed it. Regenerating the sample moved every id and this
script reported 0.965 instead of 0.838 with nothing saying anything had moved.
tools/recover_phase4e_judgements.py rebuilt the set from the worksheet the human
judged and keyed it on the triple. The row-id file stays as the raw record and
is no longer read.

The judgements live in annotations/ and not in build/ because build/ is
gitignored as regenerable, and a human's judgement is the one thing here that is
not. Deleting build/ would have destroyed 314 judged rows with no way to get
them back. The sample they judge is still generated into build/, which is
correct: the sample is derived, the judgement is not.

Both are keyed on the triple and its pattern rather than on a row
number, so a judgement survives a re-run of the extractor and a guard can be
scored against work already done. Rows the guards have since removed drop out
and are reported separately, which is the only measurement that says whether a
guard removed errors or removed evidence.
"""
import pandas as pd

K = ["subject", "predicate", "object", "pattern"]
rel = pd.read_parquet("build/relations.parquet")

e = pd.read_csv("annotations/phase4e_judged.csv")
g = pd.read_csv("annotations/phase4g_judged.csv")
j = pd.concat([e[K + ["ok"]], g[K + ["ok"]]]).drop_duplicates(K)

cur = rel[K].drop_duplicates().assign(present=True)
m = j.merge(cur, on=K, how="left")
m["present"] = m.present.fillna(False).astype(bool)
live = m[m.present]

src = rel.assign(source=[
    "nplink" if (pd.notna(a) or pd.notna(b)) else
    "coref" if (pd.notna(c) or pd.notna(d)) else "direct"
    for a, b, c, d in zip(rel.nplink_subject, rel.nplink_object,
                          rel.coref_subject, rel.coref_object)])
live = live.merge(src[K + ["source"]].drop_duplicates(K), on=K, how="left")

gone = ~m.present
cut_false, cut_true = int(((~m.ok) & gone).sum()), int((m.ok & gone).sum())
# The two removed counts partition the rows the guards took out. They did not,
# while present was an object column: ~True is -2 and ~False is -1, both true,
# so each mask collapsed to the judgement alone and the line printed the whole
# judged set. D22 Fault 8.
assert cut_false + cut_true == len(m) - len(live), "removed counts are not a partition"

L = ["# Phase 4 precision on the current graph", "",
     "Judgements on record: %d. Still present after the guards: %d."
     % (len(m), len(live)),
     "Removed by the guards: %d judged false, %d judged true." % (cut_false, cut_true),
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
