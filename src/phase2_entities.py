"""Phase 2. Entity resolution.

This is construction, not validation. The corpus `canonical_id` field is a
context-free find-and-replace dictionary: 88.0% of its values are verbatim
echoes of the surface form and only 16 of 3,933 distinct values carry more than
one surface form. It is used here as a weak prior and never as truth.

Resolution evidence comes from the text itself. Where the translation writes an
epithet next to the name it belongs to ("Vaikartana Karna", "Partha
Yudhishthira"), that apposition is direct evidence of co-designation, and every
instance carries a (parvan, adhyaya) citation like any other assertion.

An epithet whose apposition evidence is split across several bearers is NOT
merged. It keeps its own node and carries a candidate set with confidences, per
the hard rule against guessing an entity link. "Kounteya" is apposed to
Yudhishthira 14 times, Bhimasena 12, and Arjuna 12; the prior assigns all 611
of its mentions to Arjuna.
"""
import bisect, collections, csv, json, re, sys
import pandas as pd

import vocab

BASE = "http://example.org/mbh/"   # provisional, does not resolve, nothing published

# Titles and honorifics are capitalized in the translation and are not names.
TITLES = {"King", "Queen", "Prince", "Princess", "Lord", "Sage", "Rishi", "Maharshi",
          "Brahmana", "Kshatriya", "Emperor", "Bhagavan", "Shri", "Sri", "Great",
          "Mother", "Father", "Son", "Daughter", "Chief", "Commander", "Sir",
          "Grandfather", "Uncle", "Brother", "Sister", "Lady", "Master", "Preceptor"}

# Merge thresholds. A merge needs dominant evidence and no live competitor.
MIN_EVIDENCE = 3      # apposition instances supporting the winning bearer
MIN_SHARE = 0.90      # winning bearer's share of this epithet's apposition evidence
MAX_RIVAL = 0.10      # any rival above this share blocks the merge

_POSS = re.compile(r"['’]s\b", re.IGNORECASE)
_NAME = re.compile(r"\b([A-Z][a-z]{2,})\b")
_APPOS = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")


def norm_surface(s):
    """Orthographic normalization only. Merges case, possessives, spacing and
    hyphens. This is the one merge that needs no evidence beyond the string."""
    s = _POSS.sub("", str(s)).replace("-", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


def load():
    df = pd.read_parquet("build/mentions.parquet")
    df["surface_norm"] = df.surface.map(norm_surface)
    return df[df.surface_norm.astype(bool)]


def person_names(df):
    """Single-token PERSON surfaces frequent enough to carry evidence."""
    per = df[df.label == "PERSON"]
    vc = per.surface.str.strip().value_counts()
    return {s for s, n in vc.items()
            if n >= 10 and re.fullmatch(r"[A-Z][a-z]{2,}", s) and s not in TITLES}


def appositions(df, names):
    """Adjacent name-name pairs, with a citation and the sentence for each.

    The sentence is kept because every human ruling this phase asks for is a
    reading of the passage, and a citation with no text means hunting for it.

    Both orders are recorded because the translation uses both ("Partha
    Dhananjaya", "Dharmaraja Yudhishthira"). Direction is resolved later by
    which member is the more frequent bare name.
    """
    sents = df.drop_duplicates(subset=["parva", "sent_id"])
    ev = collections.defaultdict(list)
    for r in sents.itertuples():
        for m in _APPOS.finditer(r.sentence):
            a, b = m.group(1), m.group(2)
            if a in names and b in names and a != b:
                ev[(a, b)].append((r.parva, r.adhyaya_ordinal, r.sentence))
    return ev


def bare_counts(df, names):
    """How often each name appears as a standalone mention surface."""
    s = df[df.label == "PERSON"].surface.str.strip()
    vc = s.value_counts()
    return {n: int(vc.get(n, 0)) for n in names}


def prior_map(df):
    """The corpus `canonical_id` as a surface-to-value map. It is a consistent
    function: every surface has exactly one value. A value equal to its own
    surface is an echo and carries no cross-surface claim."""
    s = df[df.canonical_id_prior.notna()]
    return dict(zip(s.surface.str.strip(), s.canonical_id_prior.str.strip()))


def build_candidates(ev, bare, prior):
    """For each epithet, the bearers it is apposed to, weighted.

    Direction defaults to the member with the lower bare-mention count, because
    "Vaikartana Karna" has Karna standing alone constantly and Vaikartana almost
    never. Where the prior performs a rewrite on one member, that member is taken
    as the dependent instead; without this, "Partha Dhananjaya" resolves Dhananjaya
    onto Partha, merging one epithet of Arjuna into another.
    """
    cand = collections.defaultdict(collections.Counter)
    cites = collections.defaultdict(list)
    for (a, b), inst in ev.items():
        a_rw = prior.get(a) not in (None, a)
        b_rw = prior.get(b) not in (None, b)
        if a_rw and not b_rw:
            dep, head = a, b
        elif b_rw and not a_rw:
            dep, head = b, a
        else:
            dep, head = (a, b) if bare.get(a, 0) <= bare.get(b, 0) else (b, a)
        cand[dep][head] += len(inst)
        cites[(dep, head)].extend(inst)
    return cand, cites


def homonym_risk(ev, prior):
    """Heads that cannot safely absorb an epithet, because one of their own
    apposition partners is assigned elsewhere by the prior.

    Partner count does not detect this. "Krishna" has sixteen apposition partners
    because Vasudeva Krishna genuinely carries fifteen epithets (Janardana,
    Madhava, Govinda, Achyuta and the rest). The homonyms hide inside that
    legitimate cluster: Krishna Dvaipayana is Vyasa and Draupadi is also called
    Krishna. A rich epithet cluster and a homonym look identical to apposition
    evidence, which is the core limitation of automatic resolution on this corpus.

    The one signal available is disagreement: if a head is apposed to a term the
    prior assigns to a different entity, the head is a homonymy risk and nothing
    is merged into it.
    """
    partners = collections.defaultdict(set)
    for (a, b) in ev:
        partners[a].add(b)
        partners[b].add(a)
    risk = {}
    for head, ps in partners.items():
        # Compare partners against the head's canonical form, not its surface.
        # Achyuta and Janardana are both epithets of Krishna, so comparing
        # surfaces made every member of a legitimate epithet cluster block every
        # other one.
        canon = prior.get(head, head)
        off = sorted(t for t in ps
                     if prior.get(t) not in (None, t) and prior.get(t) != canon)
        if off:
            risk[head] = off
    return risk


def resolve(cand, prior, risk):
    """Classify every appellation. No transitive closure and no silent choice.

    Chaining merges would fuse Arjuna, Partha and Yudhishthira into one node
    through a shared ambiguous epithet, which is the failure this phase exists to
    avoid.

    A merge is asserted only where two independent sources agree: apposition
    evidence in the translated text, and a rewrite in the corpus prior. Either
    source alone produces a candidate link, never a merge.

    Apposition alone is not enough. It proves that a phrase designates one
    person, not that the bare epithet designates the bare head elsewhere.
    "Krishna Dvaipayana" is Vyasa, and merging on apposition alone resolved
    Dvaipayana onto Vasudeva Krishna. So did Droupadi, who is also called
    Krishna. Both are exactly the conflation the hard rules forbid.

    The prior alone is not enough either: it is a context-free dictionary that
    assigns all 611 mentions of Kounteya to Arjuna where the text disambiguates
    to Yudhishthira more often.
    """
    out = {}
    for dep, c in cand.items():
        tot = sum(c.values())
        ranked = c.most_common()
        head, n = ranked[0]
        share = n / tot
        rival = (ranked[1][1] / tot) if len(ranked) > 1 else 0.0
        pv = prior.get(dep)
        prior_claims = pv is not None and pv != dep     # an echo is not a claim
        dominant = (n >= MIN_EVIDENCE and share >= MIN_SHARE and rival <= MAX_RIVAL
                    and head not in risk)
        if prior_claims and dominant and pv == head:
            status, entity, conf = "asserted", head, round(share, 4)
        elif prior_claims and dominant and pv != head:
            status, entity, conf = "contested", None, None
        elif dominant:
            status, entity, conf = "apposition_only", None, round(share, 4)
        else:
            status, entity, conf = "ambiguous", None, None
        out[dep] = {"status": status, "entity": entity, "confidence": conf,
                    "head_homonym_risk": risk.get(head) or None,
                    "total": tot, "share": round(share, 4), "rival": round(rival, 4),
                    "prior": pv if prior_claims else None,
                    "candidates": [{"entity": h, "evidence": k,
                                    "confidence": round(k / tot, 4)} for h, k in ranked]}
    # Appellations the prior rewrites but the text never apposes. A dictionary
    # claim with no textual support is a candidate, not an assertion.
    for dep, pv in prior.items():
        if pv != dep and dep not in out:
            out[dep] = {"status": "prior_only", "entity": None, "confidence": None,
                        "total": 0, "share": 0.0, "rival": 0.0, "prior": pv,
                        "candidates": [{"entity": pv, "evidence": 0, "confidence": None}]}
    return out


def iri(s):
    return "<%se/%s>" % (BASE, re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_"))


def write_appellations(res, cites, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["appellation", "candidate_entity", "confidence", "evidence_count",
                    "status", "prior_says", "apposition_share", "runner_up_share",
                    "first_citation", "evidence_sentence"])
        for dep in sorted(res):
            r = res[dep]
            for c in r["candidates"]:
                cc = cites.get((dep, c["entity"]), [])
                w.writerow([dep, c["entity"],
                            "" if c["confidence"] is None else c["confidence"],
                            c["evidence"], r["status"], r["prior"] or "",
                            r["share"], r["rival"],
                            "%s/%s" % cc[0][:2] if cc else "",
                            cc[0][2] if cc else ""])


def write_entities_ttl(res, names, path, counts=None, places=()):
    """Entities, merged appellations, and candidate links that stay unresolved.

    An asserted appellation becomes `skos:altLabel` on its entity. Anything else
    keeps a node of its own with `mbh:designatesCandidate` links, each carrying a
    confidence and an evidence count, so no silent choice is made and the
    unresolved cases stay visible rather than disappearing into a merge.

    Every surface the annotation labelled as an entity is declared here, not only
    the ones frequent enough to carry apposition evidence. The ten-mention
    threshold is what a merge needs; it is not what existing needs. While it
    governed this file the graph referred to 251 entity nodes that had no
    declaration anywhere. Each node carries its mention count, so the evidence
    tier stays visible as data instead of as an absence.
    """
    lines = ["@prefix mbh: <%s> ." % BASE,
             "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
             "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
             "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", ""]
    merged = {d: r["entity"] for d, r in res.items() if r["status"] == "asserted"}
    alt = collections.defaultdict(list)
    for d, e in merged.items():
        alt[e].append(d)
    ents = sorted(set(names) | set(merged.values()) | set(counts or ()))
    for n in ents:
        if n in merged:
            continue
        mc = ' ; mbh:mentionCount %d' % counts[n] if counts is not None and n in counts else ""
        lines.append("%s a mbh:Person ; rdfs:label %s%s ." % (iri(n), json.dumps(n), mc))
        for a in sorted(alt.get(n, [])):
            lines.append("%s skos:altLabel %s ." % (iri(n), json.dumps(a)))
        lines.append("")
    for n in sorted(places):
        mc = ' ; mbh:mentionCount %d' % counts[n] if counts is not None and n in counts else ""
        lines.append("%s a mbh:Place ; rdfs:label %s%s ." % (iri(n), json.dumps(n), mc))
    if places:
        lines.append("")
    for dep in sorted(res):
        r = res[dep]
        if r["status"] == "asserted":
            continue
        lines.append("%s a mbh:Appellation ; rdfs:label %s ; mbh:referentUnresolved true ; "
                     'mbh:resolutionStatus "%s" .' % (iri(dep), json.dumps(dep), r["status"]))
        for c in r["candidates"]:
            if c["entity"] is None:
                continue
            conf = "" if c["confidence"] is None else (
                ' mbh:confidence "%s"^^xsd:decimal ;' % c["confidence"])
            lines.append("%s mbh:designatesCandidate [ mbh:entity %s ;%s "
                         "mbh:evidenceCount %d ] ."
                         % (iri(dep), iri(c["entity"]), conf, c["evidence"]))
        lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return len([n for n in ents if n not in merged]), len(merged)


def main():
    df = load()
    names = person_names(df)
    bare = bare_counts(df, names)
    prior = prior_map(df)
    ev = appositions(df, names)
    cand, cites = build_candidates(ev, bare, prior)
    risk = homonym_risk(ev, prior)
    res = resolve(cand, prior, risk)
    json.dump({k: v for k, v in sorted(risk.items())},
              open("build/phase2_homonym_risk.json", "w"), indent=1)
    persons, places, counts, _ = vocab.vocabulary(df)
    n_ent, n_merged = write_entities_ttl(
        res, names, "out/entities.ttl",
        counts={n: int(counts[n]) for n in persons}, places=places)
    write_appellations(res, cites, "build/appellations.csv")
    # One example sentence per candidate link, so the human audit reads the
    # passage instead of going looking for it.
    for dep, r in res.items():
        for c in r["candidates"]:
            cc = cites.get((dep, c["entity"]), [])
            c["citation"] = "%s/%s" % cc[0][:2] if cc else None
            c["example"] = cc[0][2] if cc else None
    json.dump(res, open("build/phase2_resolution.json", "w"), indent=1)
    by = collections.Counter(r["status"] for r in res.values())
    print("evidence_names=%d apposition_pairs=%d entities=%d places=%d merged=%d"
          % (len(names), len(ev), n_ent, len(places), n_merged))
    print("appellations by status: %s" % dict(by))
    print("heads blocked as homonymy risks: %d" % len(risk))


if __name__ == "__main__":
    main()
