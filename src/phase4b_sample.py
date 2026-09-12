"""Draw the precision sample for the coreference-derived triples.

Phase 4b added a new source of arguments, so the triples that depend on it are
unjudged. The triples that do not depend on it come from the extractor judged in
build/phase4_gate.md, whose patterns and guards are byte-identical, so that
verdict is carried over rather than redrawn; the combined figure in the gate
report is the weighted combination of the two.

Stratified by predicate, capped at 60, so instructedBy and cursedBy are taken
entire. Grammatical person is carried into the worksheet because first and
second person resolutions are the ones most likely to be wrong, and the gate
reports precision separately for them.
"""
import pandas as pd

SEED = 20260914
CAP = 60

rel = pd.read_parquet("build/relations.parquet").reset_index(drop=True)
rel["row_id"] = rel.index
cor = rel[rel.coref_subject.notna() | rel.coref_object.notna()]

parts = [g.sample(n=min(len(g), CAP), random_state=SEED) for _, g in cor.groupby("predicate")]
s = pd.concat(parts).sample(frac=1.0, random_state=SEED)
s["verdict"] = ""
s[["row_id", "predicate", "subject", "object", "pattern", "confidence",
   "coref_subject", "coref_object", "depth", "parva", "adhyaya_ordinal",
   "evidence", "verdict"]].to_csv("build/phase4b_precision_sample.csv", index=False)
print("seed=%d population=%d sampled=%d" % (SEED, len(cor), len(s)))
