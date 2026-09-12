"""Draw the Phase 4 precision sample.

Stratified by predicate so the small predicates are judgeable at all. A uniform
sample of 300 from 1,579 instances would return about one cursedBy and no
instructedBy, and a per-predicate precision breakdown is what the gate asks for.
Every stratum is capped at 80 and floored at the whole stratum where it is
smaller, so cursedBy and instructedBy are taken entire.

The judgement is made against the evidence sentence, which is Debroy's prose.
It is not an independent evaluation: the person adjudicating is the person who
wrote the patterns. That is stated in the report.
"""
import pandas as pd

SEED = 20260912
TARGET = 300
CAP = 80

rel = pd.read_parquet("build/relations.parquet")
rel = rel.reset_index(drop=True)
rel["row_id"] = rel.index

groups = {p: g for p, g in rel.groupby("predicate")}
# Water-filling: give every stratum an equal share, spill the remainder of any
# stratum smaller than its share onto the rest, repeat until it settles.
alloc = {p: 0 for p in groups}
left = TARGET
while left > 0:
    open_ = [p for p in groups if alloc[p] < min(len(groups[p]), CAP)]
    if not open_:
        break
    share = max(1, left // len(open_))
    for p in open_:
        room = min(len(groups[p]), CAP) - alloc[p]
        take = min(share, room, left)
        alloc[p] += take
        left -= take

parts = [groups[p].sample(n=alloc[p], random_state=SEED) for p in sorted(groups) if alloc[p]]
s = pd.concat(parts).sample(frac=1.0, random_state=SEED)
s["verdict"] = ""
s[["row_id", "predicate", "subject", "object", "pattern", "confidence",
   "parva", "adhyaya_ordinal", "evidence", "verdict"]].to_csv(
    "build/phase4_precision_sample.csv", index=False)
print("seed=%d sampled=%d alloc=%s" % (SEED, len(s), alloc))
