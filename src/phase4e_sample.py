"""Draw the precision sample for what the predicate expansion added.

Two things are new and unjudged: the combat, defeat and verb-based speech
patterns, and the arguments recovered by noun-phrase linking. Everything else
was judged in build/phase4_gate.md and build/phase4b_gate.md and is not redrawn.

Stratified by pattern rather than by predicate, because the patterns are what is
new and they fail in different ways: a direct object of "pierce" is a different
reading problem from an addressee behind "to". Every nplink-derived row is taken
entire, there being only 46.
"""
import pandas as pd

SEED = 20260915
CAP = 40
NEW = {"combat_dobj", "combat_passive", "defeat_dobj", "defeat_passive",
       "speech_dobj", "speech_passive", "speech_to"}

rel = pd.read_parquet("build/relations.parquet").reset_index(drop=True)
rel["row_id"] = rel.index
new = rel[rel.pattern.isin(NEW)]
npl = rel[rel.nplink_subject.notna() | rel.nplink_object.notna()]

parts = [g.sample(n=min(len(g), CAP), random_state=SEED) for _, g in new.groupby("pattern")]
s = pd.concat(parts + [npl]).drop_duplicates("row_id").sample(frac=1.0, random_state=SEED)
s["verdict"] = ""
s[["row_id", "predicate", "subject", "object", "pattern", "confidence",
   "coref_subject", "coref_object", "nplink_subject", "nplink_object",
   "depth", "parva", "adhyaya_ordinal", "evidence", "verdict"]].to_csv(
    "build/phase4e_precision_sample.csv", index=False)
print("seed=%d new_patterns=%d nplink=%d sampled=%d" % (SEED, len(new), len(npl), len(s)))
