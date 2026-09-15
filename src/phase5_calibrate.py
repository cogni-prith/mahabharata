"""Phase 5 gate. Groundedness per predicate against the blind calibration set.

Reads annotations/phase5_blind.json and, only after that file exists and passes
validation, build/blind_calibration_key.csv. The guard is on the blind judgement
file specifically and not on the annotations/ directory, which Phase 2 and Phase
4 also write into: guarding the directory would unseal the key as soon as any
unrelated annotation landed.

The key holds the stratum and the pipeline's own output. Opening it before the
annotation is done would contaminate the only uncontaminated set this project
has. Running this early is therefore a no-op with a message, not an error.

Writes build/phase5_calibration.md. This is not build/groundedness_report.md,
which src/phase5_propose.py writes about the proposal step itself.

Run: .venv/bin/python src/phase5_calibrate.py
"""
import collections
import csv
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phase2_entities
from annotation_data import PREDICATES, load_dataset, validate_review

BLIND = "annotations/phase5_blind.json"
KEY = "build/blind_calibration_key.csv"
OUT = "build/phase5_calibration.md"
MIN_N = 10                       # below this a cell gets a count, never a rate
SYMMETRIC = {"hasSpouse", "foughtWith"}


def norm(s):
    return phase2_entities.norm_surface(s)


def parse(cell):
    """The key stores triples as 'subject|predicate|object ; ...'."""
    out = []
    for part in str(cell or "").split(" ; "):
        bits = part.split("|")
        if len(bits) == 3 and any(b.strip() for b in bits):
            out.append(tuple(b.strip() for b in bits))
    return out


def same(a, b):
    """Two triples name the same relation between the same two surfaces."""
    if a[1] != b[1]:
        return False
    x, y = (norm(a[0]), norm(a[2])), (norm(b[0]), norm(b[2]))
    return x == y or (a[1] in SYMMETRIC and x == y[::-1])


def near(a, b):
    """Same predicate and exactly one end agreeing. A surface mismatch, not a
    disagreement about what the sentence says."""
    if a[1] != b[1]:
        return False
    hits = sum(norm(a[i]) == norm(b[i]) for i in (0, 2))
    return hits == 1


def rate(num, den):
    return "n/a" if not den else "%.3f" % (num / den)


def cell(num, den):
    """Below MIN_N the honest output is the count. A rate over 3 rows reads as a
    measurement and is not one."""
    if den == 0:
        return "no rows"
    if den < MIN_N:
        return "%d/%d (count only, n<%d)" % (num, den, MIN_N)
    return "%.3f (%d/%d)" % (num / den, num, den)


def score(rows, column, emitted):
    """Per-predicate grounded / total over one stratum, plus the named buckets."""
    grounded = collections.Counter()
    total = collections.Counter()
    buckets = collections.Counter()
    missed = collections.Counter()
    for r in rows:
        pipeline = parse(r["row"][column])
        human = [t for t in r["triples"]]
        for p in pipeline:
            total[p[1]] += 1
            hit = [h for h in human if same(p, (h["subject"], h["predicate"], h["object"]))]
            if hit:
                grounded[p[1]] += 1
                if any(h["unclear"] for h in hit):
                    buckets["matched, annotator marked the triple unclear"] += 1
            elif any(near(p, (h["subject"], h["predicate"], h["object"])) for h in human):
                buckets["same predicate, one surface differs"] += 1
            elif r["status"] == "none":
                buckets["annotator read the sentence as stating no relation"] += 1
            else:
                buckets["annotator wrote nothing matching"] += 1
        for h in human:
            t = (h["subject"], h["predicate"], h["object"])
            if not h["predicate"]:
                continue
            if not any(same(p, t) for p in pipeline):
                missed[h["predicate"]] += 1
                if h["predicate"] not in emitted:
                    buckets["annotator used a predicate the extractor cannot emit"] += 1
    return grounded, total, buckets, missed


def main():
    if not os.path.exists(BLIND):
        print("%s does not exist, so there is nothing to calibrate and the key stays"
              " closed. Annotate build/annot_blind.html first, then install the export"
              " with: .venv/bin/python src/annot_pages.py --import PATH" % BLIND)
        return
    review = json.load(open(BLIND, encoding="utf-8"))
    data = load_dataset("blind")
    try:
        validate_review(review, data, require_submitted=True)
    except ValueError as e:
        raise SystemExit("%s is not a complete submitted review for the current draw "
                         "(%s). The key stays closed." % (BLIND, e))

    # Only now. Everything above this line runs on public inputs.
    with open(KEY, encoding="utf-8", newline="") as f:
        key = {r["key"]: r for r in csv.DictReader(f)}
    if key and next(iter(key.values())).get("sheet_id") != data["sheet_id"]:
        raise SystemExit("the key belongs to a different draw than the sheet just judged")

    emitted = set(pd.read_parquet("build/relations.parquet").predicate.unique())
    rows = collections.defaultdict(list)
    for r in review["rows"]:
        k = key.get(r["key"])
        if k is None:
            raise SystemExit("judged row %s is absent from the key" % r["key"])
        rows[k["stratum"]].append({"row": k, "status": r["status"],
                                   "triples": r["triples"], "notes": r["notes"]})

    a_g, a_t, a_b, a_m = score(rows["A"], "pipeline_asserted", emitted)
    c_g, c_t, c_b, c_m = score(rows["C"], "pipeline_proposed", emitted)
    b_written = collections.Counter()
    b_unclear = 0
    for r in rows["B"]:
        for h in r["triples"]:
            if h["predicate"]:
                b_written[h["predicate"]] += 1
                b_unclear += bool(h["unclear"])

    prop = pd.read_parquet("build/proposed.parquet")
    n_aggregate = int((prop.sent_id < 0).sum())
    below = [p for p in sorted(a_t) if a_t[p] < MIN_N]

    L = ["# Phase 5 calibration against the blind set", "",
         "Automated, from `src/phase5_calibrate.py`. Procedure:",
         "`docs/PHASE5_SAMPLING.md`. Sheet id `%s`, annotator %s."
         % (data["sheet_id"], json.dumps(review["annotator"])), "",
         "Strata are reported apart at every line and are never pooled. A cell under",
         "n=%d gets a count and not a rate." % MIN_N, "",
         "This measures agreement between the extractor and one annotator reading the",
         "same sentence. It is not proof that a triple is true, and one annotator gives",
         "no inter-annotator agreement figure.", "",
         "| stratum | rows judged |", "|---|---|",
         "| A, Phase 4 asserted | %d |" % len(rows["A"]),
         "| B, pipeline produced nothing | %d |" % len(rows["B"]),
         "| C, Phase 5 proposed, census | %d |" % len(rows["C"]), "",
         "## Stratum A groundedness, per predicate", "",
         "Of the triples Phase 4 asserted on these sentences, the share the annotator",
         "also wrote. Matched on the predicate and on both names after",
         "`phase2_entities.norm_surface`; `hasSpouse` and `foughtWith` match unordered.",
         "", "| predicate | grounded | asserted | result |", "|---|---|---|---|"]
    for p in sorted(a_t, key=lambda p: -a_t[p]):
        L.append("| %s | %d | %d | %s |" % (p, a_g[p], a_t[p], cell(a_g[p], a_t[p])))
    L.append("| **all** | %d | %d | %s |"
             % (sum(a_g.values()), sum(a_t.values()), cell(sum(a_g.values()), sum(a_t.values()))))
    if below:
        L += ["", "Reported as counts, under n=%d: %s." % (MIN_N, ", ".join(below))]
    L += ["", "Matched and counted as grounded above, but the annotator marked the"
          " triple unclear: %d."
          % a_b.pop("matched, annotator marked the triple unclear", 0),
          "", "### Unmatched assertions, in named buckets", "",
          "| bucket | triples |", "|---|---|"]
    L += ["| %s | %d |" % (k, v) for k, v in a_b.most_common()] or ["| none | 0 |"]

    L += ["", "## Stratum C groundedness, the proposal layer", "",
          "A census of the %d proposal-bearing sentences, reported separately from A and"
          % len(rows["C"]),
          "never pooled with it. This is the number that decides whether a proposal may",
          "be asserted.", "", "| predicate | grounded | proposed | result |", "|---|---|---|---|"]
    for p in sorted(c_t, key=lambda p: -c_t[p]):
        L.append("| %s | %d | %d | %s |" % (p, c_g[p], c_t[p], cell(c_g[p], c_t[p])))
    L += ["",
          "%d `isEnemyOf` proposals are outside every stratum: they aggregate combat over"
          % n_aggregate,
          "four or more chapters and carry `sent_id = -1`, so no single sentence grounds",
          "one. They are counted here and no rate is computed over them.", ""]
    if c_b:
        L += ["| bucket | triples |", "|---|---|"]
        L += ["| %s | %d |" % (k, v) for k, v in c_b.most_common()] + [""]

    L += ["## Stratum B, what the pipeline never saw", "",
          "Stratum B is defined as the sentences the pipeline produced nothing from, so",
          "every triple the annotator wrote here is a miss by construction and a rate",
          "would be 1.000 by definition. The honest form is how many per %d sentences,"
          % len(rows["B"]),
          "and of what kind.", "",
          "| predicate | triples the annotator wrote | extractor emits it |",
          "|---|---|---|"]
    for p, n in b_written.most_common():
        L.append("| %s | %d | %s |" % (p, n, "yes" if p in emitted else "no"))
    L += ["| **all** | %d | |" % sum(b_written.values()), "",
          "%d of them are marked unclear by the annotator." % b_unclear, "",
          "## Misses on stratum A", "",
          "Triples the annotator wrote on a stratum A sentence that the extractor did",
          "not produce. Not a precision figure; the recall side of the same rows.", "",
          "| predicate | missed | extractor emits it |", "|---|---|---|"]
    for p, n in a_m.most_common():
        L.append("| %s | %d | %s |" % (p, n, "yes" if p in emitted else "no"))
    L += ["", "Five of the fifteen annotator predicates are not emitted by the extractor",
          "at all, so they can only ever appear here. Extractor inventory in use: %s."
          % ", ".join(sorted(emitted)), "",
          "Annotator inventory: %s." % ", ".join(sorted(PREDICATES)), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")

    overall = cell(sum(a_g.values()), sum(a_t.values()))
    print("stratum A %s | stratum C %s | stratum B %d triples over %d sentences"
          % (overall, cell(sum(c_g.values()), sum(c_t.values())),
             sum(b_written.values()), len(rows["B"])))
    print("wrote %s" % OUT)


def selftest():
    """The matcher, which is where a wrong number would come from."""
    assert same(("Karna", "slainBy", "Arjuna"), ("karna", "slainBy", "Arjuna's"))
    assert not same(("Karna", "slainBy", "Arjuna"), ("Arjuna", "slainBy", "Karna"))
    assert same(("A", "foughtWith", "B"), ("B", "foughtWith", "A"))     # symmetric
    assert not same(("A", "speaksTo", "B"), ("B", "speaksTo", "A"))     # not symmetric
    assert near(("Karna", "slainBy", "Arjuna"), ("Karna", "slainBy", "Partha"))
    assert not near(("Karna", "slainBy", "Arjuna"), ("Karna", "slainBy", "Arjuna"))
    assert not near(("Karna", "slainBy", "Arjuna"), ("X", "cursedBy", "Y"))
    assert parse("a|b|c ; d|e|f") == [("a", "b", "c"), ("d", "e", "f")]
    assert parse("") == [] and parse(None) == []
    assert cell(3, 4).startswith("3/4 (count only") and cell(0, 0) == "no rows"
    assert cell(5, 10) == "0.500 (5/10)"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
