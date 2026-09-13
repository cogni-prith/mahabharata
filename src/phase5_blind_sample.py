"""Phase 5. Draw the blind calibration sample.

The procedure is `docs/PHASE5_SAMPLING.md` and was written before this ran.
This script implements it and nothing else: population conditions 1 to 3, two
strata of 200, proportional allocation with a floor of 5 per parva, one seed.

The sheet carries no label, no span, no extracted triple and no stratum. The
key file carries all of that and is not to be opened before annotating.

Run: .venv/bin/python src/phase5_blind_sample.py
"""
import csv
import random
import pandas as pd

SEED = 20260913
PER_STRATUM = 200
FLOOR = 5
MIN_CHARS = 60
MASTER = "mahabharata_ner_by_sentence_master (2).csv"


def allocate(counts, n):
    """Proportional to eligible count, floor of FLOOR, remainder to the largest."""
    total = sum(counts.values())
    take = {p: max(FLOOR, int(n * c / total)) for p, c in counts.items()}
    take = {p: min(v, counts[p]) for p, v in take.items()}
    order = sorted(counts, key=lambda p: -counts[p])
    while sum(take.values()) > n:                      # the floor overshot
        for p in reversed(order):
            if take[p] > 1 and sum(take.values()) > n:
                take[p] -= 1
    i = 0
    while sum(take.values()) < n:
        p = order[i % len(order)]
        if take[p] < counts[p]:
            take[p] += 1
        i += 1
    return take


def draw(df, counts, n, rng):
    out = []
    for parva, k in allocate(counts, n).items():
        rows = df[df.parva == parva]
        idx = list(rows.index)
        rng.shuffle(idx)
        out.extend(idx[:k])
    return out


def main():
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
    per = per[per >= 2].reset_index().rename(columns={"surface": "n_person"})

    text = sents[["parva", "sent_id", "sentence"]].drop_duplicates(subset=["parva", "sent_id"]) \
        if "sentence" in sents.columns else None
    if text is None:
        raise SystemExit("master CSV has no sentence column; read it before assuming one")

    df = text.merge(anchored, on=["parva", "sent_id"]).merge(per, on=["parva", "sent_id"])
    df = df[df.sentence.str.len() >= MIN_CHARS]        # condition 2

    # Stratum membership: did the pipeline produce anything from this sentence.
    produced = set(map(tuple, rel[["parva", "sent_id"]].drop_duplicates().values)) | \
        set(map(tuple, prop[["parva", "sent_id"]].drop_duplicates().values))
    key = list(zip(df.parva, df.sent_id))
    df = df.assign(stratum=["A" if k in produced else "B" for k in key])

    picks = []
    for s in ("A", "B"):
        sub = df[df.stratum == s]
        counts = sub.parva.value_counts().to_dict()
        picks.extend(draw(sub, counts, PER_STRATUM, rng))
    sheet = df.loc[picks].copy()
    sheet["id"] = ["c%03d" % i for i in range(len(sheet))]
    sheet = sheet.sample(frac=1, random_state=SEED)    # hide the stratum in the order

    with open("build/blind_calibration_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "parva", "adhyaya", "sentence", "triples", "none", "unclear", "notes"])
        for r in sheet.itertuples():
            w.writerow([r.id, r.parva, int(r.adhyaya_ordinal), r.sentence, "", "", "", ""])

    # The key. Everything the annotator must not see, including what the
    # pipeline claimed about each sentence, so scoring needs no re-derivation.
    rel_by = rel.groupby(["parva", "sent_id"])
    prop_by = prop.groupby(["parva", "sent_id"])
    with open("build/blind_calibration_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "parva", "sent_id", "adhyaya", "stratum", "seed",
                    "pipeline_asserted", "pipeline_proposed"])
        for r in sheet.itertuples():
            k = (r.parva, r.sent_id)
            a = ["%s|%s|%s" % tuple(t) for t in rel_by.get_group(k)[["subject", "predicate", "object"]].values] \
                if k in rel_by.groups else []
            p = ["%s|%s|%s" % tuple(t) for t in prop_by.get_group(k)[["subject", "predicate", "object"]].values] \
                if k in prop_by.groups else []
            w.writerow([r.id, r.parva, r.sent_id, int(r.adhyaya_ordinal), r.stratum,
                        SEED, " ; ".join(a), " ; ".join(p)])

    n = sheet.stratum.value_counts().to_dict()
    print("eligible=%d drawn=%d strata=%s parvas=%d seed=%d"
          % (len(df), len(sheet), n, sheet.parva.nunique(), SEED))
    print("sheet=build/blind_calibration_sheet.csv key=build/blind_calibration_key.csv")


if __name__ == "__main__":
    main()
