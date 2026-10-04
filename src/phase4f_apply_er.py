"""Phase 4f. Fold relation endpoints through the Phase 2b entity links.

Optional, and off in tools/run_all.sh. Phase 4 links the strings the text used,
so Dhananjaya, Phalguna and Arjuna are three nodes. This step rewrites each
endpoint of build/relations.parquet to the entity Phase 2b accepted for that
name, so Phases 5 to 8 and the explorer are built on the folded graph.

What it may fold, most specific first (phase2b_confidence.endpoint_map):
the accepted link of that name in that sentence; the chapter, where every
accepted mention of the name there agrees; corpus-wide, only for a name with a
single candidate everywhere. A name whose mentions went to human review keeps
the string it was written with. Nothing is folded on a guess.

The extraction as Phase 4 wrote it is kept in build/relations_raw.parquet and is
always the input, so running this twice gives the same result. The written name
of every endpoint stays on the row as subject_written / object_written. A triple
whose two ends fold to one person ("Partha spoke to Dhananjaya", both accepted as
Arjuna) is dropped, as Phase 4 drops subj == obj.

The judged precision samples in annotations/ are keyed on the unfolded triple,
so the order matters: phase4_precision.py runs before this step.

Writes build/relations.parquet, out/relations.ttl and build/er/apply_report.md.
Run: python src/phase4f_apply_er.py
"""
import collections
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2b_confidence import endpoint_map, gate  # noqa: E402

RAW = "build/relations_raw.parquet"
LINKS = "build/er/mention_links.csv"


def main():
    if not os.path.exists(LINKS):
        raise SystemExit("%s does not exist. Run src/phase2b_confidence.py first." % LINKS)
    if not os.path.exists(RAW):
        shutil.copy("build/relations.parquet", RAW)
    rel = pd.read_parquet(RAW)
    links = pd.read_csv(LINKS, dtype={"parva": str})
    m = endpoint_map(links)

    def fold(name, r):
        ch = None if pd.isna(r.adhyaya_ordinal) else "%s/%d" % (r.parva, int(r.adhyaya_ordinal))
        sid = None if pd.isna(r.sent_id) else int(r.sent_id)
        return m(name, r.parva, sid, ch)

    subj = [fold(r.subject, r) for r in rel.itertuples()]
    obj = [fold(r.object, r) for r in rel.itertuples()]
    out = rel.assign(subject_written=rel.subject, object_written=rel.object,
                     subject=subj, object=obj)
    self_loop = out.subject == out.object
    out = out[~self_loop].reset_index(drop=True)
    out.to_parquet("build/relations.parquet")

    import phase4_relations
    phase4_relations.write_ttl(out, "out/relations.ttl")

    changed = collections.Counter()
    for a, b in zip(list(rel.subject) + list(rel.object), subj + obj):
        if a != b:
            changed[(a, b)] += 1
    trip = lambda d: [tuple(x) for x in d[["subject", "predicate", "object"]].values]
    before, after = gate(trip(rel)), gate(trip(out))
    L = ["# Phase 4f. Relation endpoints folded through Phase 2b links", "",
         "Input `%s` (Phase 4 as extracted), output `build/relations.parquet`." % RAW, "",
         "- relation rows: %d in, %d out" % (len(rel), len(out)),
         "- endpoints rewritten: %d of %d" % (sum(changed.values()), 2 * len(rel)),
         "- rows dropped because both ends folded to one person: %d" % int(self_loop.sum()), "",
         "| measure | as extracted | folded | required |", "|---|---|---|---|",
         "| distinct triples | %d | %d | >= 10,000 |" % (before["triples"], after["triples"]),
         "| entities | %d | %d | |" % (before["entities"], after["entities"]),
         "| entities of degree >= 5 | %d | %d | >= 500 |" % (before["degree_ge_5"], after["degree_ge_5"]),
         "| largest relation share | %.1f%% %s | %.1f%% %s | <= 40%% |"
         % (100 * before["top_share"], before["top_relation"],
            100 * after["top_share"], after["top_relation"]),
         "| mean degree | %.2f | %.2f | |" % (before["mean_degree"], after["mean_degree"]),
         "| highest degree | %s | %s | |" % (before["max_degree"], after["max_degree"]), "",
         "## Most frequent rewrites", "", "| written | folded to | endpoints |", "|---|---|---|"]
    L += ["| %s | %s | %d |" % (a, b, n) for (a, b), n in changed.most_common(40)]
    open("build/er/apply_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:18]))


if __name__ == "__main__":
    main()
