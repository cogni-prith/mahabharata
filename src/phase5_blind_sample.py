"""Phase 5. Draw the blind calibration sample.

The procedure is `docs/PHASE5_SAMPLING.md` and was written before this ran.
This script implements it and nothing else: population conditions 1 to 3, a
predicate-allocated stratum A of 200, a parva-allocated stratum B of 200, a
census stratum C, one seed.

The sheet carries no label, no span, no extracted triple and no stratum. The
key file carries all of that and is not to be opened before annotating.

The sheet is frozen once drawn. run_all.sh runs this on every pass and the
eligible frame moves when an earlier phase changes, so redrawing silently would
invalidate annotation already made. This refuses to overwrite and says so; a
redraw is deleting the sheet and the key on purpose.

Run: .venv/bin/python src/phase5_blind_sample.py
"""
import csv
import hashlib
import os
import random
import sys

import pandas as pd

import phase1_mentions

SEED = 20260914
STRATUM_A = 200
STRATUM_B = 200
FLOOR = 5
MIN_CHARS = 60
MASTER = "mahabharata_ner_by_sentence_master (2).csv"
SHEET = "build/blind_calibration_sheet.csv"
KEY = "build/blind_calibration_key.csv"
REPORT = "build/blind_calibration_report.md"


def row_key(sentence):
    """The content key. Keyed on the sentence, so a judgement survives a redraw
    of everything around it. Normalized first so punctuation drift in an
    upstream re-export does not orphan an annotation."""
    n = phase1_mentions.normalize(sentence)
    return hashlib.blake2b(n.encode("utf-8"), digest_size=5).hexdigest()


def sheet_id(keys):
    """Identifies a draw by its contents. Every judgement file carries this, so
    a judgement made against a retired sheet is detectable."""
    return hashlib.sha1(" ".join(sorted(keys)).encode("utf-8")).hexdigest()[:12]


def allocate_parva(counts, n):
    """Proportional to eligible count, floor of FLOOR, remainder to the largest."""
    total = sum(counts.values())
    take = {p: max(FLOOR, int(n * c / total)) for p, c in counts.items()}
    take = {p: min(v, counts[p]) for p, v in take.items()}
    order = sorted(counts, key=lambda p: -counts[p])
    # The floor can push the total past n. Take it back from the largest parvas:
    # the earlier version took it from the smallest, down to 1, which undid the
    # floor for exactly the parvas the floor exists to protect.
    while sum(take.values()) > n:
        before = sum(take.values())
        for p in order:
            if take[p] > min(FLOOR, counts[p]) and sum(take.values()) > n:
                take[p] -= 1
        if sum(take.values()) == before:               # every parva is at its floor
            raise SystemExit("floor of %d over %d parvas cannot fit in %d"
                             % (FLOOR, len(counts), n))
    i = 0
    while sum(take.values()) < n:
        p = order[i % len(order)]
        if take[p] < counts[p]:
            take[p] += 1
        i += 1
    return take


def allocate_predicate(counts, n):
    """Equal share across predicates, rarest first, capped by supply.

    The gate is a rate per predicate, so the draw has to guarantee the cell
    sizes rather than let the corpus mix decide them. Rarest first means a
    scarce predicate takes its share before a common one can absorb it, and the
    share a capped predicate cannot use flows to the ones that still have
    supply instead of being lost.
    """
    order = sorted(counts, key=lambda p: counts[p])
    take, left = {}, n
    for i, p in enumerate(order):
        take[p] = min(left // (len(order) - i), counts[p])
        left -= take[p]
    while left > 0:
        room = [p for p in order if take[p] < counts[p]]
        if not room:
            break
        for p in room[::-1]:                           # the largest pools absorb it
            if left == 0:
                break
            take[p] += 1
            left -= 1
    return take


def draw_by_parva(df, n, rng):
    out = []
    counts = df.parva.value_counts().to_dict()
    for parva, k in allocate_parva(counts, n).items():
        idx = list(df[df.parva == parva].index)
        rng.shuffle(idx)
        out.extend(idx[:k])
    return out


def draw_by_predicate(df, carriers, n, rng):
    """carriers: predicate -> set of df index labels carrying it.

    A sentence carrying several predicates is drawn once and counts for all of
    them, so the realised cell for a common predicate runs above its allocation.
    That is reported, not corrected: correcting it would mean rejecting a
    sentence for what else it says.
    """
    counts = {p: len(v) for p, v in carriers.items()}
    take = allocate_predicate(counts, n)
    picked = []
    seen = set()
    for p in sorted(carriers, key=lambda p: counts[p]):
        avail = [i for i in carriers[p] if i not in seen]
        rng.shuffle(avail)
        for i in avail[:take[p]]:
            seen.add(i)
            picked.append(i)
    short = n - len(picked)
    if short > 0:                                      # caps left room; top up at random
        rest = [i for i in df.index if i not in seen]
        rng.shuffle(rest)
        picked.extend(rest[:short])
    return picked, take, short


def main():
    if os.path.exists(SHEET):
        print("%s exists; not redrawing. Delete it and %s on purpose to redraw,"
              " after changing docs/PHASE5_SAMPLING.md and recording a new seed." % (SHEET, KEY))
        return

    rng = random.Random(SEED)
    sents = pd.read_csv(MASTER)
    rel = pd.read_parquet("build/relations.parquet")
    prop = pd.read_parquet("build/proposed.parquet")

    # Condition 1: a chapter anchor. The spine is on the mention layer, so the
    # eligible sentences are the ones Phase 1 anchored to a chapter.
    men = pd.read_parquet("build/mentions.parquet")
    anchored = men[men.adhyaya_ordinal.notna()][["parva", "sent_id", "adhyaya_ordinal"]]
    anchored = anchored.drop_duplicates(subset=["parva", "sent_id"])

    # Condition 3: two distinct PERSON surfaces, from the annotation layer,
    # which is permitted as a retrieval index and never as evidence.
    per = men[men.label == "PERSON"].groupby(["parva", "sent_id"]).surface.nunique()
    per = per[per >= 2].reset_index()[["parva", "sent_id"]]

    if "sentence" not in sents.columns:
        raise SystemExit("master CSV has no sentence column; read it before assuming one")
    text = sents[["parva", "sent_id", "sentence"]].drop_duplicates(subset=["parva", "sent_id"])

    df = text.merge(anchored, on=["parva", "sent_id"]).merge(per, on=["parva", "sent_id"])
    df = df.assign(key=[row_key(s) for s in df.sentence])

    # Stratum membership. C is the proposal layer, which is 19 sentences against
    # Phase 4's thousands, so it is a census and takes precedence over A. The 7
    # isEnemyOf proposals carry sent_id -1 because they aggregate combat over
    # several chapters; no sentence grounds one and they are outside every
    # stratum by construction.
    asserted = set(map(tuple, rel[["parva", "sent_id"]].drop_duplicates().values))
    proposed = set(map(tuple, prop[prop.sent_id >= 0][["parva", "sent_id"]]
                       .drop_duplicates().values))
    k = list(zip(df.parva, df.sent_id))
    df = df.assign(in_a=[x in asserted for x in k], in_c=[x in proposed for x in k])

    # Deduplicate on the content key before any draw, because the judgement is
    # keyed on it and two rows sharing one would share a judgement. Stratum C
    # rows sort first so a census row is never the copy that gets dropped.
    n_raw = len(df)
    df = df.sort_values(["in_c", "in_a"], ascending=False).drop_duplicates(subset="key")
    df = df.sort_values(["parva", "sent_id"]).reset_index(drop=True)
    assert df.key.is_unique

    # Condition 2, the 60-character floor, excludes vocatives and interjections
    # from a random draw. Stratum C is a census of a layer of 19, so the floor
    # is waived there and the waiver is reported: 2 of the 19 are shorter and
    # both state a birth.
    long_enough = df.sentence.str.len() >= MIN_CHARS
    pool_c = df[df.in_c]
    pool_a = df[df.in_a & ~df.in_c & long_enough]
    pool_b = df[~df.in_a & ~df.in_c & long_enough]

    # Which predicates each stratum A sentence carries, for the allocation.
    sid = pool_a.set_index(["parva", "sent_id"]).key.to_dict()
    idx_of = {r.key: i for i, r in zip(pool_a.index, pool_a.itertuples())}
    carriers = {}
    for r in rel.itertuples():
        kk = sid.get((r.parva, r.sent_id))
        if kk is not None:
            carriers.setdefault(r.predicate, set()).add(idx_of[kk])

    pick_a, alloc, short = draw_by_predicate(pool_a, carriers, STRATUM_A, rng)
    pick_b = draw_by_parva(pool_b, STRATUM_B, rng)
    sheet = pd.concat([df.loc[pick_a].assign(stratum="A"),
                       df.loc[pick_b].assign(stratum="B"),
                       pool_c.assign(stratum="C")])
    sheet = sheet.sample(frac=1, random_state=SEED)    # hide the stratum in the order
    sid_hash = sheet_id(sheet.key)

    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "parva", "adhyaya", "sentence"])
        for r in sheet.itertuples():
            w.writerow([r.key, r.parva, int(r.adhyaya_ordinal), r.sentence])

    # The key. Everything the annotator must not see, including what the
    # pipeline claimed about each sentence, so scoring needs no re-derivation.
    # The sentence is here too: the calibrator joins on the key, and a key file
    # that cannot be read back to its own text is not auditable.
    rel_by = rel.groupby(["parva", "sent_id"])
    prop_by = prop.groupby(["parva", "sent_id"])

    def triples(by, kk):
        if kk not in by.groups:
            return ""
        return " ; ".join("%s|%s|%s" % tuple(t)
                          for t in by.get_group(kk)[["subject", "predicate", "object"]].values)

    with open(KEY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "parva", "sent_id", "adhyaya", "stratum", "seed", "sheet_id",
                    "sentence", "pipeline_asserted", "pipeline_proposed"])
        for r in sheet.itertuples():
            kk = (r.parva, r.sent_id)
            w.writerow([r.key, r.parva, r.sent_id, int(r.adhyaya_ordinal), r.stratum,
                        SEED, sid_hash, r.sentence, triples(rel_by, kk), triples(prop_by, kk)])

    # The run report. Realised cells, not just allocated ones, because a
    # sentence carrying several predicates lands in several cells.
    a_keys = set(sheet[sheet.stratum == "A"].key)
    realised_s, realised_t = {}, {}
    for r in rel.itertuples():
        kk = sid.get((r.parva, r.sent_id))
        if kk in a_keys:
            realised_s.setdefault(r.predicate, set()).add(kk)
            realised_t[r.predicate] = realised_t.get(r.predicate, 0) + 1

    counts = sheet.stratum.value_counts().to_dict()
    n_short = sum(1 for s in pool_c.sentence if len(s) < MIN_CHARS)
    L = ["# Blind calibration draw", "",
         "Automated, from `src/phase5_blind_sample.py`. Procedure:",
         "`docs/PHASE5_SAMPLING.md`. Seed %d. Sheet id `%s`." % (SEED, sid_hash), "",
         "The sheet is frozen. This script refuses to overwrite it.", "",
         "## Population", "",
         "| step | sentences |", "|---|---|",
         "| conditions 1 and 3 | %d |" % n_raw,
         "| after deduplicating on the content key | %d |" % len(df),
         "| condition 2, at least %d characters | %d |" % (MIN_CHARS, int(long_enough.sum())),
         "| stratum A pool, Phase 4 asserted | %d |" % len(pool_a),
         "| stratum B pool, pipeline produced nothing | %d |" % len(pool_b),
         "| stratum C, Phase 5 proposed, census | %d |" % len(pool_c),
         "",
         "%d of the %d stratum C sentences are under the %d-character floor, which is"
         % (n_short, len(pool_c), MIN_CHARS),
         "waived for a census. Both state a birth. %d isEnemyOf proposals carry"
         % int((prop.sent_id < 0).sum()),
         "`sent_id = -1` and are outside every stratum; they are counted, never rated.",
         "",
         "## Stratum A, allocated by predicate", "",
         "Equal share of %d across the %d predicates Phase 4 asserts, rarest first,"
         % (STRATUM_A, len(carriers)),
         "capped by supply. A sentence is drawn once and counts for every predicate it",
         "carries, so a realised cell runs above its allocation.", "",
         "| predicate | eligible sentences | allocated | drawn | triples drawn |",
         "|---|---|---|---|---|"]
    for p in sorted(carriers, key=lambda p: -len(carriers[p])):
        L.append("| %s | %d | %d | %d | %d |"
                 % (p, len(carriers[p]), alloc[p], len(realised_s.get(p, ())),
                    realised_t.get(p, 0)))
    L += ["", "Topped up at random after the caps: %d." % short,
          "Cells under 10 get a count in the calibration report and not a rate.", "",
          "## Strata", "",
          "| stratum | rows |", "|---|---|"]
    L += ["| %s | %d |" % (s, counts.get(s, 0)) for s in ("A", "B", "C")]
    L += ["| total | %d |" % len(sheet), "",
          "Parvas represented: %d of 18." % sheet.parva.nunique(), ""]
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")

    print("eligible=%d drawn=%d strata=%s parvas=%d seed=%d sheet_id=%s"
          % (len(df), len(sheet), counts, sheet.parva.nunique(), SEED, sid_hash))
    print("sheet=%s key=%s report=%s" % (SHEET, KEY, REPORT))


def selftest():
    """The two allocators, which are the only real logic here."""
    a = allocate_predicate({"rare": 6, "mid": 20, "common": 900}, 200)
    assert sum(a.values()) == 200 and a["rare"] == 6 and a["mid"] == 20, a
    b = allocate_predicate({"x": 3, "y": 4}, 200)
    assert b == {"x": 3, "y": 4}, b                    # supply below the target
    c = allocate_parva({"p": 100, "q": 2, "r": 98}, 50)
    assert sum(c.values()) == 50 and c["q"] == 2, c    # the floor cannot exceed supply
    d = allocate_parva({"a": 1000, "b": 1}, 10)
    assert sum(d.values()) == 10 and d["b"] == 1, d
    assert row_key("Arjuna spoke to Krishna.") == row_key("arjuna spoke to krishna")
    assert sheet_id(["a", "b"]) == sheet_id(["b", "a"])
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
