"""Phase 4l check. Hand-judged packets for four non-kinship relation types.

Before the judge (src/phase4k_judge.py) is trusted on a new relation type, a
sample of that type is judged by hand. Four people share the work, so the
sample is cut into four packets of 40: 5 shared items every packet carries, to
measure how well the people agree, and 35 of the packet's own. Each packet
mixes all four types, so no one person's reading decides a type.

Types and sources:
  killed      graph slainBy           + GLiREL "killed"
  teacher     graph instructedBy      + GLiREL "is the teacher of"
  charioteer                            GLiREL "is the charioteer of"
  fought      graph foughtWith        + GLiREL "is an enemy of"

GLiREL gives most pairs in both orders; per sentence and pair the
higher-scoring order is kept, and the page asks about direction.

Like the kinship pages, the packets quote a copyrighted translation: share them
privately, never publish them or commit them.

Run:  python src/phase4l_relations_check.py                    draw, write 4 packet pages
      python src/phase4l_relations_check.py --import F [F ...]  store packet CSVs, report
      python src/phase4l_relations_check.py --report             report on what is stored
      python src/phase4l_relations_check.py --teacher-rest       page for the graph's undrawn teacher relations
      python src/phase4l_relations_check.py --teacher-rest --import F
"""
import argparse
import glob
import json
import os
import random
import re

import pandas as pd

from phase4k_sample import TEMPLATE, VERDICTS, context
from phase4k_judge_split import wilson

OUT = "build/relations_check"
SAMPLE = OUT + "/sample.csv"
JUDGED = "annotations/relations_judged_packet_%d.csv"
REST_SAMPLE = OUT + "/teacher_rest.csv"
REST_PAGE = OUT + "/teacher_rest.html"
REST_JUDGED = "annotations/relations_judged_teacher_rest.csv"
SEED = 20261029
PACKETS = 4
SHARED = 5
# Items per (type, source); 145 in all = 5 shared + 4 x 35.
DRAW = {("killed", "graph"): 16, ("killed", "glirel"): 21,
        ("teacher", "graph"): 12, ("teacher", "glirel"): 24,
        ("charioteer", "glirel"): 36,
        ("fought", "graph"): 16, ("fought", "glirel"): 20}
GRAPH = {"slainBy": "killed", "instructedBy": "teacher", "foughtWith": "fought"}
GLIREL = {"killed": "killed", "is the teacher of": "teacher",
          "is the charioteer of": "charioteer", "is an enemy of": "fought"}
RULE = {
    "killed": "Correct only if the passage says or clearly implies the first person killed the second. "
              "Wounding, defeating or attacking is not killing. Wrong direction: the victim is given as the killer.",
    "teacher": "Correct if the passage says the first person taught, trained or instructed the second as a teacher "
               "(weapons, scripture, a vow of study). One piece of advice in a conversation is not enough. "
               "Wrong direction: the pupil is given as the teacher.",
    "charioteer": "Correct if the first person drives, or drove, the second person's chariot. "
                  "Wrong direction: the chariot's owner is given as the driver.",
    "fought": "Correct if the passage says the two fought each other or were enemies. Being on opposite sides "
              "of the war is not enough unless the passage has them fight or calls them enemies.",
}


def claim(t, a, b):
    return {"killed": "%s killed %s", "teacher": "%s was the teacher of %s",
            "charioteer": "%s was the charioteer of %s"}.get(t, "%s and %s fought each other or were enemies") % (a, b)


def name_like(s):
    s = str(s)
    return len(s) > 2 and s[0].isupper() and not re.search(r"\d", s)


def pools():
    r = pd.read_parquet("build/reviewed_relations.parquet")
    r = r[r.predicate.isin(GRAPH) & (r.status == "asserted")]
    # slainBy and instructedBy read "subject is slain by / taught by object".
    flip = r.predicate.isin(["slainBy", "instructedBy"])
    g = pd.DataFrame({"type": r.predicate.map(GRAPH), "source": "graph",
                      "a": r.object.where(flip, r.subject), "b": r.subject.where(flip, r.object),
                      "via_hint": r.coref_subject.notna() | r.coref_object.notna(),
                      "parva": r.parva, "sent_id": r.sent_id})
    x = pd.read_parquet("build/glirel_relations.parquet")
    x = x[x.label.isin(GLIREL) & (x["head"] != x["tail"]) & x["head"].map(name_like) & x["tail"].map(name_like)]
    x = x.assign(pair=[tuple(sorted((h, t))) for h, t in zip(x["head"], x["tail"])])
    x = x.sort_values("score", ascending=False).drop_duplicates(["parva", "sent_id", "label", "pair"])
    l = pd.DataFrame({"type": x.label.map(GLIREL), "source": "glirel", "a": x["head"], "b": x["tail"],
                      "via_hint": (x.head_via == "coref") | (x.tail_via == "coref"),
                      "parva": x.parva, "sent_id": x.sent_id})
    both = pd.concat([g, l]).drop_duplicates(["type", "parva", "sent_id", "a", "b"]).reset_index(drop=True)
    return both


def draw():
    rel = pools()
    rng = random.Random(SEED)
    picked = []
    for (t, src), n in DRAW.items():
        pool = sorted(rel.index[(rel.type == t) & (rel.source == src)])
        if len(pool) < n:
            raise SystemExit("only %d %s/%s relations, need %d" % (len(pool), t, src, n))
        picked += rng.sample(pool, n)
    s = rel.loc[picked].reset_index(drop=True)
    s["claim"] = [claim(t, a, b) for t, a, b in zip(s.type, s.a, s.b)]
    # Shared: one of each type, plus one more killed, the most-asked type.
    shared = []
    for t in ("killed", "teacher", "charioteer", "fought", "killed"):
        shared.append(rng.choice([i for i in s.index[s.type == t] if i not in shared]))
    s["packet"] = 0
    own = [i for i in s.index if i not in shared]
    # Deal each type round-robin so every packet gets a similar mix.
    own.sort(key=lambda i: (s.type[i], rng.random()))
    for n, i in enumerate(own):
        s.loc[i, "packet"] = n % PACKETS + 1
    order = list(s.index)
    rng.shuffle(order)
    s = s.loc[order].reset_index(drop=True)
    s.insert(0, "id", ["X%03d" % (i + 1) for i in range(len(s))])
    return s


def packet_page(rows, ctx, label, slug):
    items = []
    for r, c in zip(rows.itertuples(), ctx):
        items.append({"id": r.id, "claim": r.claim, "rule": RULE[r.type], "a": r.a, "b": r.b, "type": r.type,
                      "evidence": "", "via_hint": bool(r.via_hint), "parva": r.parva, "sent_id": int(r.sent_id), **c})
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    html = (TEMPLATE.replace("__DATA__", data)
            .replace("<title>Kinship check</title>", "<title>Relations check, %s</title>" % label)
            .replace("<h1>Kinship check</h1>", "<h1>Relations check &middot; %s</h1>" % label)
            .replace('"kinship-check-v1"', json.dumps("relations-check-%s-v1" % slug))
            .replace('a.download="kinship_judged.csv"', 'a.download=%s' % json.dumps("relations_judged_%s.csv" % slug))
            .replace('<div class="mute">Quote the model gave: “${esc(x.evidence)}”</div>',
                     '<div class="rule">${esc(x.rule)}</div>')
            .replace(".hints{", ".rule{font-size:14px;color:var(--mute);margin-top:4px}.hints{")
            .replace("e.g. child given as parent", "e.g. victim given as killer, pupil as teacher"))
    for old in ("Kinship check", "kinship-check-v1", "kinship_judged.csv", "Quote the model gave"):
        assert old not in html, old
    return html


def build():
    s = draw()
    os.makedirs(OUT, exist_ok=True)
    s.to_csv(SAMPLE, index=False)
    ctx = dict(zip(s.id, context(s)))
    for n in range(1, PACKETS + 1):
        rows = s[s.packet.isin([0, n])].sample(frac=1, random_state=SEED + n).reset_index(drop=True)
        path = "%s/packet_%d.html" % (OUT, n)
        open(path, "w", encoding="utf-8").write(packet_page(rows, [ctx[i] for i in rows.id], "packet %d" % n, "packet_%d" % n))
        print("%s: %d items (%d shared)" % (path, len(rows), int((rows.packet == 0).sum())))
    print("\n%d relations, seed %d -> %s" % (len(s), SEED, SAMPLE))
    print(pd.crosstab(s.type, s.source, margins=True).to_string())


def draw_teacher_rest():
    """The graph's teacher relations the packets did not draw."""
    rel = pools()
    rel = rel[(rel.type == "teacher") & (rel.source == "graph")]
    keys = ["parva", "sent_id", "a", "b"]
    seen = pd.read_csv(SAMPLE)[keys].assign(seen=True)
    s = rel.merge(seen, how="left", on=keys)
    s = s[s.seen != True].drop(columns="seen").reset_index(drop=True)
    s["claim"] = [claim(t, a, b) for t, a, b in zip(s.type, s.a, s.b)]
    s.insert(0, "id", ["T%02d" % (i + 1) for i in range(len(s))])
    return s


def build_teacher_rest():
    s = draw_teacher_rest()
    s.to_csv(REST_SAMPLE, index=False)
    open(REST_PAGE, "w", encoding="utf-8").write(packet_page(s, context(s), "teacher, the rest", "teacher_rest"))
    print("%d relations -> %s\npage -> %s" % (len(s), REST_SAMPLE, REST_PAGE))
    print(s[["id", "claim", "parva", "sent_id"]].to_string(index=False))


def store_teacher_rest(path):
    s = pd.read_csv(REST_SAMPLE)
    j = pd.read_csv(path)
    if {"id", "verdict"} - set(j.columns) or set(j.id) != set(s.id):
        raise SystemExit("%s does not match the teacher page's items" % path)
    bad = j.verdict.notna() & ~j.verdict.isin(VERDICTS)
    if bad.any():
        raise SystemExit("%s: unknown verdicts %s" % (path, sorted(j.verdict[bad].unique())))
    s = s.merge(j[["id", "verdict", "note"] if "note" in j.columns else ["id", "verdict"]], on="id")
    s.to_csv(REST_JUDGED, index=False)
    # Every graph teacher relation: this page plus the packets.
    files = glob.glob(JUDGED.replace("%d", "*"))
    p = pd.concat([pd.read_csv(f) for f in files])
    p = p[(p.type == "teacher") & (p.source == "graph")].drop_duplicates("id")
    every = pd.concat([p, s])
    print(every[["id", "claim", "verdict"]].to_string(index=False))
    ok = int((every.verdict == "correct").sum())
    print("\ngraph teacher relations: %d of %d correct" % (ok, len(every)))


def store(paths):
    s = pd.read_csv(SAMPLE)
    for path in paths:
        j = pd.read_csv(path)
        if {"id", "verdict"} - set(j.columns):
            raise SystemExit("%s lacks id/verdict columns. Use the page's Download button." % path)
        own = s[s.id.isin(j.id) & (s.packet > 0)].packet.unique()
        if len(own) != 1 or set(j.id) != set(s[s.packet.isin([0, own[0]])].id):
            raise SystemExit("%s does not match any one packet's items" % path)
        bad = j.verdict.notna() & ~j.verdict.isin(VERDICTS)
        if bad.any():
            raise SystemExit("%s: unknown verdicts %s" % (path, sorted(j.verdict[bad].unique())))
        n = int(own[0])
        os.makedirs("annotations", exist_ok=True)
        s.merge(j[["id", "verdict", "note"] if "note" in j.columns else ["id", "verdict"]], on="id") \
            .to_csv(JUDGED % n, index=False)
        print("packet %d <- %s: %d judged, %d unsure, %d blank"
              % (n, path, int(j.verdict.isin(["correct", "wrong", "wrong_direction"]).sum()),
                 int((j.verdict == "unsure").sum()), int(j.verdict.isna().sum())))


def report():
    files = sorted(glob.glob(JUDGED.replace("%d", "*")))
    if not files:
        raise SystemExit("no packets stored yet")
    j = pd.concat([pd.read_csv(f).assign(by=int(re.search(r"(\d+)\.csv$", f).group(1))) for f in files])
    have = sorted(int(b) for b in j.by.unique())
    print("packets stored: %s%s" % (have, "" if len(have) == PACKETS else "  (missing %s)"
                                    % sorted(set(range(1, PACKETS + 1)) - set(have))))
    shared = j[j.packet == 0].pivot(index="id", columns="by", values="verdict")
    if shared.shape[1] > 1:
        print("\nshared items (agreement between people):")
        print(shared.fillna("-").to_string())
        cols = list(shared.columns)
        same = tot = 0
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                both = shared[[cols[a], cols[b]]].dropna()
                same += int((both[cols[a]] == both[cols[b]]).sum())
                tot += len(both)
        if tot:
            print("pairwise agreement: %d of %d = %.0f%%" % (same, tot, 100 * same / tot))
    # One verdict per item: shared items by majority (ties dropped).
    def majority(v):
        v = v.dropna().value_counts()
        return v.index[0] if len(v) and (len(v) == 1 or v.iloc[0] > v.iloc[1]) else None
    one = j.groupby("id").agg(type=("type", "first"), source=("source", "first"), verdict=("verdict", majority))
    done = one[one.verdict.isin(["correct", "wrong", "wrong_direction"])]
    print("\nhand-checked precision of the extracted relations (unsure and ties left out):")
    for keys, g in [(("all",), done)] + list(done.groupby(["type"])) + list(done.groupby(["type", "source"])):
        k = int((g.verdict == "correct").sum())
        ci = wilson(k, len(g))
        print("  %-22s %3d / %3d correct = %3.0f%%  (95%% CI %.0f-%.0f%%), wrong direction %d"
              % (" / ".join(keys) if isinstance(keys, tuple) else keys, k, len(g), 100 * k / len(g),
                 100 * ci[0], 100 * ci[1], int((g.verdict == "wrong_direction").sum())))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--import", dest="imp", nargs="+", metavar="FILE", help="downloaded packet CSVs")
    ap.add_argument("--report", action="store_true", help="report on the packets stored so far")
    ap.add_argument("--teacher-rest", action="store_true",
                    help="the graph's teacher relations the packets did not draw")
    args = ap.parse_args()
    if args.teacher_rest:
        if args.imp:
            store_teacher_rest(args.imp[0])
        else:
            build_teacher_rest()
    elif args.imp:
        store(args.imp)
        report()
    elif args.report:
        report()
    else:
        build()


if __name__ == "__main__":
    main()
