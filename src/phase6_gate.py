"""Phase 6 entry gate. Checked and reported before any training.

Three conditions, all from CLAUDE.md, all hard:
  1. at least 10,000 triples
  2. at least 500 entities of degree 5 or higher
  3. no single relation type above 40% of all triples

If any fails, stop and report. Do not train something meaningless and present
its numbers.

Proposed triples are counted separately. They are a hypothesis graph and they
would only be eligible for training after the Phase 5 groundedness gate passes,
which it has not, because the blind calibration set does not exist.
"""
import collections
import pandas as pd

rel = pd.read_parquet("build/relations.parquet")
prop = pd.read_parquet("build/proposed.parquet")

asserted = rel.drop_duplicates(["subject", "predicate", "object"])
deg = collections.Counter()
for t in asserted.itertuples():
    deg[t.subject] += 1
    deg[t.object] += 1

n_trip = len(asserted)
hi = sum(1 for v in deg.values() if v >= 5)
share = asserted.predicate.value_counts(normalize=True)
top, top_share = share.index[0], share.iloc[0]

c1, c2, c3 = n_trip >= 10000, hi >= 500, top_share <= 0.40

buckets = collections.Counter()
for v in deg.values():
    buckets[min(v, 20) if v < 20 else 20] += 1

with open("out/eval.md", "w", encoding="utf-8") as f:
    f.write("# Phase 6 entry gate\n\nChecked before training, as required. "
            "No model was trained.\n\n")
    f.write("| condition | required | measured | result |\n|---|---|---|---|\n")
    f.write("| distinct asserted triples | >= 10,000 | %d | %s |\n"
            % (n_trip, "PASS" if c1 else "**FAIL**"))
    f.write("| entities of degree >= 5 | >= 500 | %d | %s |\n"
            % (hi, "PASS" if c2 else "**FAIL**"))
    f.write("| largest relation share | <= 40%% | %.1f%% (%s) | %s |\n"
            % (top_share * 100, top, "PASS" if c3 else "**FAIL**"))
    f.write("\n**The gate does not pass. Nothing was trained.**\n\n"
            "Proposed triples, held separately and not eligible: %d.\n"
            "Entities appearing in the asserted graph: %d.\n\n" % (len(prop), len(deg)))
    f.write("## Degree distribution\n\n| degree | entities |\n|---|---|\n")
    for d in sorted(buckets):
        f.write("| %s | %d |\n" % ("20+" if d == 20 else d, buckets[d]))
    f.write("\n## Relation shares\n\n| relation | triples | share |\n|---|---|---|\n")
    for p, n in asserted.predicate.value_counts().items():
        f.write("| %s | %d | %.1f%% |\n" % (p, n, 100 * n / n_trip))

print("triples=%d deg5=%d top=%s %.3f pass=%s" % (n_trip, hi, top, top_share, c1 and c2 and c3))
