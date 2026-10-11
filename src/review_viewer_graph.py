"""Recheck current extraction records without replacing a concurrently trained graph.

The viewer consumes this corrected snapshot. Rejected records remain auditable;
vectors and their original training inputs are not rewritten by this review.
"""
import collections
from pathlib import Path
import pandas as pd
import spacy
import nplink
from phase4_relations import extract_sentence, load_coref, load_entities, write_ttl

ROOT = Path(__file__).resolve().parents[1]


def folder():
    """The Phase 4f endpoint fold, or None when Phase 4f has not run.

    Re-extraction reads the names as written, "Dhananjaya", while a folded
    relations.parquet holds "Arjuna". Without the same fold the keys never
    meet, and 373 of 478 folded rows were withheld as unsupported.
    """
    links = ROOT / "build/er/mention_links.csv"
    if not ((ROOT / "build/relations_raw.parquet").exists() and links.exists()):
        return None
    from phase2b_confidence import endpoint_map
    m = endpoint_map(pd.read_csv(links, dtype={"parva": str}))

    def fold(row):
        adh = row.get("adhyaya_ordinal")
        ch = None if adh is None or pd.isna(adh) else "%s/%d" % (row["parva"], int(adh))
        sid = int(row["sent_id"])
        s, o = (m(row[k], row["parva"], sid, ch) for k in ("subject", "object"))
        return None if s == o else {**row, "subject": s, "object": o}
    return fold


def review(frame):
    archived = [pd.read_parquet(path) for name in ("relation_candidates.parquet", "withheld_relations.parquet")
                if (path := ROOT / "build" / name).exists()]
    population = pd.concat([frame, *archived], ignore_index=True)
    persons, places, mentions = load_entities()
    sources = mentions.drop_duplicates(["parva", "sent_id"]).set_index(["parva", "sent_id"])
    corefs = load_coref()
    keys = list(dict.fromkeys((r.parva, int(r.sent_id)) for r in population.itertuples() if r.pattern != "vocative"))
    contexts = {}
    depths = population.drop_duplicates(["parva", "sent_id"]).set_index(["parva", "sent_id"]).depth.fillna(0)
    for key in keys:
        if key not in sources.index:
            continue
        s = sources.loc[key]
        contexts[key] = {"parva": key[0], "sent_id": key[1], "sentence": s.sentence,
                         "adhyaya_ordinal": s.adhyaya_ordinal, "chapter_cited": s.chapter_cited,
                         "depth": int(depths.loc[key])}
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    docs = dict(zip(contexts, nlp.pipe((r["sentence"] for r in contexts.values()), batch_size=128)))

    def sweep(table):
        rows = []
        for key, doc in docs.items():
            extract_sentence(doc, contexts[key], persons, places, rows, corefs.get(key), table)
        for row in rows:
            row.setdefault("status", "asserted")
            row.setdefault("reason", None)
        return rows

    first = pd.DataFrame(sweep(None))
    table = nplink.build_table(first[first.status == "asserted"])
    fold = folder()
    candidates = collections.defaultdict(list)
    for row in sweep(table):
        if fold:
            row = fold(row)
            if row is None:
                continue          # both ends are one person once folded, as in 4f
        candidates[(row["parva"], row["sent_id"], row["subject"], row["object"])].append(row)
    kept, rejected = [], []
    retyped = 0
    for r in frame.to_dict("records"):
        if r["pattern"] == "vocative":
            kept.append(r)
            continue
        matches = [c for c in candidates[(r["parva"], r["sent_id"], r["subject"], r["object"])]
                   if c["predicate"] == r["predicate"] or (r["predicate"] == "hasBiologicalParent"
                      and c["predicate"] in ("hasDivineProgenitor", "hasFosterParent", "bornTo"))]
        accepted = [c for c in matches if c["status"] == "asserted"]
        if accepted:
            c = max(accepted, key=lambda c: c["confidence"])
            retyped += c["predicate"] != r["predicate"]
            kept.append(c)
        else:
            reason = next((c["reason"] for c in matches if c["reason"]),
                          "Not supported by the stricter named-argument, coreference-span or apposition checks.")
            rejected.append({**r, "status": "withheld", "reason": reason,
                             "evidence": contexts.get((r["parva"], int(r["sent_id"])), {}).get("sentence", r["evidence"])})
    corrected = pd.DataFrame(kept)
    # Keep unresolved source statements after the main pipeline removes them
    # from relations.parquet. A second export must not erase their evidence.
    held = [pd.DataFrame(rejected), *archived]
    held = [d for d in held if len(d)]
    withheld = (pd.concat(held, ignore_index=True).drop_duplicates(
        ["parva", "sent_id", "subject", "predicate", "object", "pattern"], keep="first")
        if held else pd.DataFrame(columns=list(dict.fromkeys([*frame.columns, "reason"]))))
    accepted_keys = set(zip(corrected.parva, corrected.sent_id, corrected.subject,
                            corrected.predicate, corrected.object))
    if len(withheld):
        # An old "kind unspecified" label is not verification. Recheck archived
        # candidates too, or corrected attachment errors return as source links.
        for r in withheld[withheld.reason.fillna("").str.startswith("Parentage is stated")].itertuples():
            matches = candidates[(r.parva, int(r.sent_id), r.subject, r.object)]
            if not any(c["predicate"] == "bornTo" and c["status"] == "asserted" for c in matches):
                withheld.loc[r.Index, "reason"] = "Not reproduced by current named parentage checks."
        withheld = withheld[[k not in accepted_keys for k in zip(withheld.parva, withheld.sent_id,
                     withheld.subject, withheld.predicate, withheld.object)]].copy()
        withheld["status"] = "withheld"
    corrected.to_parquet(ROOT / "build/reviewed_relations.parquet")
    withheld.to_parquet(ROOT / "build/withheld_relations.parquet")
    write_ttl(corrected, str(ROOT / "out/reviewed_relations.ttl"))
    before = len(frame.drop_duplicates(["subject", "predicate", "object"]))
    after = len(corrected.drop_duplicates(["subject", "predicate", "object"]))
    stats = {"original_instances": len(frame), "retained_instances": len(corrected),
             "withheld_instances": len(withheld), "retyped_instances": int(retyped),
             "original_triples": before, "reviewed_triples": after,
             "newly_withheld_instances": len(rejected),
             "scope": "Rule-based recheck, not independent verification"}
    reasons = withheld.reason.value_counts().to_dict() if len(withheld) else {}
    (ROOT / "build/relation_review.md").write_text(
        "# Relation integrity review\n\n"
        f"{len(frame)} input instances; {len(corrected)} retained; {len(rejected)} newly withheld; "
        f"{retyped} parentage predicates corrected. Distinct triples: {before} -> {after}.\n\n"
        f"{len(withheld)} withheld records retained, including unresolved parentage from the main pipeline.\n\n"
        "This is a rule-based recheck, not independent precision or factual validation. "
        "Vocative telling records are unchanged. This export does not train vectors; "
        "the viewer compares its displayed links with the bundled model graph at load time.\n\n"
        + "\n".join(f"- {n}: {reason}" for reason, n in reasons.items()) + "\n")
    return corrected, withheld, stats, sources
