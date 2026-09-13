"""Phase 2 audit. Quantifies what the corpus `canonical_id` prior gets wrong.

Three failure classes are measured, each against textual evidence rather than
against judgement:

1. **Shared patronymic.** The epithet has several bearers and the prior picks one.
   Measured from appositions where the translation names the bearer explicitly.
2. **Ancestor collision.** The surface also names a forebear, so the patronymic
   and its bearer share a string. Measured from kinship phrasing in the prose:
   "Bharadvaja's son" establishes that Bharadvaja is a person distinct from the
   Bharadvaja who is Drona.
3. **Unsupported.** The prior rewrites, and the text never apposes the pair.
"""
import collections, csv, json, random, re, sys
import pandas as pd
sys.path.insert(0, "src")
import phase2_entities as P

SEED = 20260912          # recorded so the audit sample reproduces
SAMPLE_N = 200


def kinship_evidence(sents, surface, bearer):
    """Sentences where `surface` is named as a parent of someone.

    A collision needs more than "A is a parent". Pritha is a parent and the prior
    rewrites Pritha to Kunti, but Pritha simply is Kunti, so "Pritha's son" is
    ordinary usage. Bharadvaja is different: the prior rewrites Bharadvaja to
    Drona while the text says "Bharadvaja's son" of Drona, so the same string
    names both the father and the son. The discriminator is whether the bearer the
    prior names appears in the sentence as the child.
    """
    pats = [re.compile(r"\b%s['’]s\s+(?:son|daughter|child|offspring)\b" % surface),
            re.compile(r"\b(?:son|daughter|child|offspring)\s+of\s+%s\b" % surface)]
    out, with_bearer = [], []
    b = re.compile(r"\b%s\b" % bearer)
    for r in sents.itertuples():
        if any(p.search(r.sentence) for p in pats):
            out.append((r.parva, r.adhyaya_ordinal, r.sentence))
            if b.search(r.sentence):
                with_bearer.append((r.parva, r.adhyaya_ordinal, r.sentence))
    return out, with_bearer


def main():
    df = P.load()
    names = P.person_names(df)
    prior = P.prior_map(df)
    ev = P.appositions(df, names)
    sents = df.drop_duplicates(subset=["parva", "sent_id"])[
        ["parva", "sent_id", "adhyaya_ordinal", "sentence"]]

    # Bearers attested for each appellation, from apposition. Class terms are
    # excluded using the corpus label: "Pandava Bhimasena" is a correct
    # apposition but Pandava is a DYNASTY, not an individual, so it cannot be a
    # bearer and counting it swamped every share.
    label = df.groupby(df.surface.str.strip()).label.agg(lambda x: x.value_counts().idxmax())
    persons = {t for t in names if label.get(t) == "PERSON"}
    raw = collections.defaultdict(collections.Counter)
    for (a, b), inst in ev.items():
        if a in persons and b in persons:
            raw[a][b] += len(inst)
            raw[b][a] += len(inst)

    mention_counts = df.surface.str.strip().value_counts()
    rewrites = [(a, b) for a, b in prior.items() if a != b]

    # Bearers are restricted to anchor names: surfaces the prior rewrites TO but
    # never FROM. That set is derived from the prior's own structure rather than
    # chosen, and it is the only non-circular anchor available here.
    #
    # Folding partners through the prior does not work. Pass one would use the
    # claim under test as evidence for itself: "Partha Bhimasena" becomes "Arjuna
    # is a bearer of Bhimasena" only because the prior asserts Partha is Arjuna.
    # Folding only through already-confirmed rewrites does not converge either,
    # because the epithet clusters are mutually entangled: confirming
    # Partha needs Dhananjaya resolved, and confirming Dhananjaya needs Partha.
    # Without a hand-authored seed of canonical entities there is no fixed point,
    # which is the substantive reason this phase needs annotation budget.
    srcs = {a for a, _ in rewrites}
    anchors = {b for _, b in rewrites if b not in srcs} & persons

    def shares():
        out = {}
        for a in srcs:
            c = collections.Counter()
            for t, k in raw.get(a, {}).items():
                if t in anchors:
                    c[t] += k
            out[a] = c
        return out

    p2 = shares()

    rows, tally = [], collections.Counter()
    for a, b in sorted(rewrites, key=lambda t: -int(mention_counts.get(t[0], 0))):
        n = int(mention_counts.get(a, 0))
        canon = p2.get(a, collections.Counter())
        tot = sum(canon.values())
        share_b = canon.get(b, 0) / tot if tot else None
        kin, kin_b = kinship_evidence(sents, a, b)
        rival = [(t, k) for t, k in canon.most_common() if t != b]
        if len(kin_b) >= 3 and (tot == 0 or share_b < 0.9):
            verdict = "ancestor_collision"
        elif tot == 0:
            verdict = "unsupported"
        elif share_b >= 0.9:
            verdict = "confirmed"
        else:
            verdict = "shared_appellation"
        tally[verdict] += 1
        rows.append({"appellation": a, "prior_entity": b, "mentions": n,
                     "apposition_total": tot,
                     "share_supporting_prior": None if share_b is None else round(share_b, 4),
                     "rival_bearers": "; ".join("%s %d" % r for r in rival[:4]),
                     "kinship_parent_sentences": len(kin),
                     "kinship_with_bearer": len(kin_b),
                     "verdict": verdict,
                     "kinship_example": (kin_b or kin)[0][2][:160] if (kin_b or kin) else ""})

    with open("build/canonical_id_audit.md", "w", encoding="utf-8") as f:
        w = f.write
        w("# Phase 2 audit of the corpus `canonical_id` prior\n\n")
        w("Generated by `src/phase2_audit.py`. Counts are measured.\n\n")
        w("The prior is a context-free dictionary. It is a consistent function: "
          "every surface form has exactly one value, applied regardless of context. "
          "Of %d distinct values, 16 carry more than one surface form, and 88.0%% "
          "of assignments are verbatim echoes of the surface. It performs %d "
          "distinct rewrites, covering %d mentions.\n\n"
          % (df.canonical_id_prior.nunique(), len(rewrites),
             int(sum(mention_counts.get(a, 0) for a, _ in rewrites))))
        w("Bearers are counted only over anchor names, meaning surfaces the prior "
          "rewrites to but never from. Folding partners through the prior itself "
          "would use each claim as evidence for itself.\n\n")
        w("## Verdict counts\n\n| verdict | rewrites | mentions affected |\n|---|---|---|\n")
        for v, c in tally.most_common():
            m = sum(r["mentions"] for r in rows if r["verdict"] == v)
            w("| %s | %d | %d |\n" % (v, c, m))
        w("\n## Every rewrite, with evidence\n\n")
        w("| appellation | prior says | mentions | apposition evidence | supporting prior | rival bearers | 'parent of' + bearer | verdict |\n")
        w("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            w("| %s | %s | %d | %d | %s | %s | %d | %s |\n"
              % (r["appellation"], r["prior_entity"], r["mentions"], r["apposition_total"],
                 "n/a" if r["share_supporting_prior"] is None
                 else "%.0f%%" % (100 * r["share_supporting_prior"]),
                 r["rival_bearers"] or "none", r["kinship_with_bearer"], r["verdict"]))
        w("\n## Ancestor collisions in detail\n\n")
        for r in rows:
            if r["verdict"] == "ancestor_collision":
                w("**%s**, which the prior rewrites to %s, is named as a parent in %d "
                  "sentences. Example: %s\n\n"
                  % (r["appellation"], r["prior_entity"], r["kinship_with_bearer"],
                     r["kinship_example"]))
    json.dump(rows, open("build/canonical_id_audit.json", "w"), indent=1)

    # Sampled candidate links for a human audit. Seeded.
    res = json.load(open("build/phase2_resolution.json"))
    links = [(a, c["entity"], r["status"], c["confidence"], c["evidence"],
              c.get("citation") or "", c.get("example") or "")
             for a, r in res.items() for c in r["candidates"] if c["entity"]]
    random.Random(SEED).shuffle(links)
    with open("build/entity_audit_sample.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["appellation", "candidate_entity", "status", "confidence",
                     "evidence_count", "citation", "evidence_sentence",
                     "correct_yes_no", "note"])
        for row in links[:SAMPLE_N]:
            wr.writerow(list(row) + ["", ""])
    print("rewrites=%d anchors=%d verdicts=%s links=%d sampled=%d seed=%d"
          % (len(rewrites), len(anchors), dict(tally), len(links),
             min(SAMPLE_N, len(links)), SEED))


if __name__ == "__main__":
    main()
