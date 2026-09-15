"""Phase 2 gate. Precision of the candidate entity links, from the human census.

Reads annotations/phase2_entities.json, which src/annot_pages.py installs from
an export of build/annot_entities.html. Refuses to run, and says so, when that
file does not exist.

This is a census of every candidate link, not a sample, so the numbers are
counts of the population and carry no sampling interval. A link judged `yes`
means the appellation can refer to that candidate in the cited passage. It does
not merge the two entities: nothing here writes to the graph, and a merge would
need the decision recorded in docs/DECISIONS.md first.

Writes build/phase2_link_precision.md.

Run: .venv/bin/python src/phase2_score.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotation_data import load_dataset, validate_review

RULED = "annotations/phase2_entities.json"
OUT = "build/phase2_link_precision.md"
MIN_N = 10


def cell(num, den):
    if den == 0:
        return "no rows"
    if den < MIN_N:
        return "%d/%d (count only, n<%d)" % (num, den, MIN_N)
    return "%.3f (%d/%d)" % (num / den, num, den)


def bucket(n):
    n = int(n or 0)
    return "0" if n == 0 else "1" if n == 1 else "2 to 4" if n < 5 else "5 or more"


def load():
    if not os.path.exists(RULED):
        print("%s does not exist, so there is nothing to score. Rule on"
              " build/annot_entities.html first, then install the export with:"
              " .venv/bin/python src/annot_pages.py --import PATH" % RULED)
        return None, None
    review = json.load(open(RULED, encoding="utf-8"))
    data = load_dataset("entities")
    try:
        validate_review(review, data, require_submitted=True)
    except ValueError as e:
        raise SystemExit("%s is not a complete submitted review for the current"
                         " candidate set (%s)" % (RULED, e))
    return review, data


def main():
    review, data = load()
    if review is None:
        return
    res = json.load(open("build/phase2_resolution.json", encoding="utf-8"))
    ev = json.load(open("build/appellation_evidence.json", encoding="utf-8"))["evidence"]

    rows = []
    for r in review["rows"]:
        s = r["source"]
        pair = "%s||%s" % (s["appellation"], s["candidate_entity"])
        rows.append({"appellation": s["appellation"], "entity": s["candidate_entity"],
                     "label": s["appellation_label"], "entity_label": s["entity_label"],
                     "status": r["status"], "corrected": r["corrected_referent"].strip(),
                     "notes": r["notes"].strip(),
                     "sightings": ev.get(pair, {}).get("total", 0),
                     "resolution": res.get(s["appellation"], {}).get("status", "unknown")})

    tally = collections.Counter(r["status"] for r in rows)
    # A link ruled not_a_person is a labelling error upstream, not a wrong link,
    # so it leaves the precision denominator instead of counting as a failure.
    judged = [r for r in rows if r["status"] in ("yes", "no")]
    yes = sum(r["status"] == "yes" for r in judged)

    def table(title, keyfn, order=None):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[keyfn(r)].append(r)
        keys = order or sorted(groups, key=lambda k: -len(groups[k]))
        out = ["", "### %s" % title, "",
               "| %s | links | yes | no | unclear | not a person | precision |" % title.lower(),
               "|---|---|---|---|---|---|---|"]
        for k in keys:
            g = groups.get(k, [])
            if not g:
                continue
            c = collections.Counter(r["status"] for r in g)
            d = c["yes"] + c["no"]
            out.append("| %s | %d | %d | %d | %d | %d | %s |"
                       % (k, len(g), c["yes"], c["no"], c["unclear"], c["not_a_person"],
                          cell(c["yes"], d)))
        return out

    L = ["# Phase 2 link precision, from the human census", "",
         "Automated, from `src/phase2_score.py`. Reviewer %s." % json.dumps(review["annotator"]),
         "",
         "Every candidate link was ruled on, so this is a census of %d links over %d"
         % (len(rows), len({r["appellation"] for r in rows})),
         "appellations and carries no sampling interval. A cell under n=%d gets a count."
         % MIN_N, "",
         "A `yes` means this appellation can refer to this candidate in the cited",
         "passage. It is not a merge: an appellation with two valid bearers in different",
         "passages is the normal case in this text, and nothing here writes to the graph.",
         "",
         "Links ruled `not a person` are a labelling error in the corpus annotation, not",
         "a wrong link, so they leave the precision denominator rather than counting as",
         "a failure. They are reported on their own line.", "",
         "| result | links |", "|---|---|"]
    for k in ("yes", "no", "unclear", "not_a_person", "undecided"):
        L.append("| %s | %d |" % (k.replace("_", " "), tally[k]))
    L += ["| **total** | %d |" % len(rows), "",
          "**Precision over the %d links ruled yes or no: %s.**" % (len(judged), cell(yes, len(judged))),
          ""]
    L += ["## Broken down"]
    L += table("Resolution status", lambda r: r["resolution"])
    L += table("Apposition sightings", lambda r: bucket(r["sightings"]),
               ["0", "1", "2 to 4", "5 or more"])
    L += table("Corpus label of the appellation", lambda r: r["label"] or "unlabelled")

    corrected = [r for r in rows if r["corrected"]]
    L += ["", "## Corrected referents offered", "",
          "%d rulings name a different referent. These are suggestions and are not"
          % len(corrected),
          "written to the graph; each would need its own evidence before it became a",
          "candidate link.", ""]
    if corrected:
        L += ["| appellation | candidate ruled on | ruling | referent offered |",
              "|---|---|---|---|"]
        L += ["| %s | %s | %s | %s |" % (r["appellation"], r["entity"], r["status"], r["corrected"])
              for r in sorted(corrected, key=lambda r: r["appellation"])]
    noted = [r for r in rows if r["notes"]]
    L += ["", "## Rulings carrying a note", "",
          "%d of %d. The notes are the reviewer's own words and are reproduced whole."
          % (len(noted), len(rows)), ""]
    L += ["- **%s / %s** (%s): %s" % (r["appellation"], r["entity"], r["status"], r["notes"])
          for r in sorted(noted, key=lambda r: r["appellation"])]
    L += [""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("links=%d ruled=%d precision=%s not_a_person=%d unclear=%d"
          % (len(rows), len(judged), cell(yes, len(judged)),
             tally["not_a_person"], tally["unclear"]))
    print("wrote %s" % OUT)


def selftest():
    assert bucket(0) == "0" and bucket(1) == "1" and bucket(4) == "2 to 4" and bucket(9) == "5 or more"
    assert cell(0, 0) == "no rows" and cell(1, 2).startswith("1/2 (count only")
    assert cell(9, 10) == "0.900 (9/10)"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
