"""One-off: give the 250 Phase 4e judgements a content key instead of a row id.

`annotations/phase4e_wrong.txt` is 59 row ids. A row id is the positional index
into `build/relations.parquet`, so it means nothing once the extractor changes.
D21 changed the extractor. The judgements survived only because the stale sample
file happened to still be sitting in gitignored `build/`; regenerating it moved
every id and the precision number jumped from 0.838 to 0.965 with nothing saying
so.

The mapping is recoverable because the worksheet the human actually judged,
`build/_judge4e.txt`, carries `row_id | pattern | subject predicate object |
evidence` for all 250 rows. This reads it once and writes
`annotations/phase4e_judged.csv` keyed on the triple and the pattern, which is
the form `annotations/phase4g_judged.csv` already uses and which survives a
re-extraction.

Refuses to overwrite. Run once: .venv/bin/python tools/recover_phase4e_judgements.py
"""
import os
import sys

import pandas as pd

WORKSHEET = "build/_judge4e.txt"
WRONG = "annotations/phase4e_wrong.txt"
OUT = "annotations/phase4e_judged.csv"
# The annotator's fifteen plus the two automated terms, used only to find the
# predicate token inside "subject predicate object". Names carry spaces; the
# predicate is the one token that is a predicate.
PREDS = {"hasBiologicalParent", "hasLegalParent", "hasFosterParent",
         "hasDivineProgenitor", "bornTo", "hasSpouse", "slainBy", "defeatedBy",
         "foughtWith", "speaksTo", "cursedBy", "instructedBy", "rules",
         "isAmshaOf", "rebornAs", "isAlliedWith", "isEnemyOf"}


def split_triple(text):
    toks = text.split()
    at = [i for i, t in enumerate(toks) if t in PREDS]
    if len(at) != 1:
        raise SystemExit("cannot split %r on a single predicate" % text)
    i = at[0]
    return " ".join(toks[:i]), toks[i], " ".join(toks[i + 1:])


def main():
    if os.path.exists(OUT):
        raise SystemExit("%s exists. This runs once; a second run would rewrite a "
                         "human's judgements from a worksheet that may no longer "
                         "match them." % OUT)
    wrong = {int(x) for x in open(WRONG).read().split()}
    rows = []
    for line in open(WORKSHEET, encoding="utf-8"):
        if not line.strip():
            continue
        rid, pattern, triple = [p.strip() for p in line.split("|")[:3]]
        s, p, o = split_triple(triple)
        rows.append({"subject": s, "predicate": p, "object": o, "pattern": pattern,
                     "ok": int(rid) not in wrong, "row_id": int(rid)})
    d = pd.DataFrame(rows)
    assert len(d) == 250, len(d)
    assert d.row_id.nunique() == len(d), "duplicate row id in the worksheet"
    missing = wrong - set(d.row_id)
    assert not missing, "judged wrong but absent from the worksheet: %s" % sorted(missing)
    assert (~d.ok).sum() == len(wrong), ((~d.ok).sum(), len(wrong))
    d.drop(columns=["row_id"]).to_csv(OUT, index=False)
    print("wrote %s: %d judgements, %d wrong, %d distinct triples"
          % (OUT, len(d), int((~d.ok).sum()),
             len(d.drop_duplicates(["subject", "predicate", "object", "pattern"]))))


if __name__ == "__main__":
    sys.exit(main())
