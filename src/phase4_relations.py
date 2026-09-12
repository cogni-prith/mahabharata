"""Phase 4. Relation extraction by English dependency parsing.

Sanskrit dependency parsing is not available here and the corpus carries no
Sanskrit, so extraction runs on Debroy's English prose with spaCy. Every relation
in this graph is therefore mediated by one translator's word choices. That is
recorded in docs/DECISIONS.md D7 and belongs in the paper.

Twelve predicates, fixed. Adding to this list is a deliberate decision, not a
code change.

Entities are surface forms, not merged identities, because Phase 2 asserted no
merges. A triple therefore links the strings the text used. Phase 2's candidate
links carry the information needed to fold them later.
"""
import collections, json, re, sys
import pandas as pd
import spacy

BASE = "http://example.org/mbh/"

PREDICATES = {
    "hasBiologicalParent": "X is the biological offspring of Y.",
    "hasFosterParent": "X was raised by Y, who is not the biological parent.",
    "hasDivineProgenitor": "X was begotten by the god Y.",
    "hasSpouse": "X and Y are married.",
    "slainBy": "X was killed by Y in the narrative.",
    "cursedBy": "Y pronounced a curse on X.",
    "isAlliedWith": "X and Y fight on the same side or make a pact.",
    "isEnemyOf": "X and Y are declared opponents.",
    "instructedBy": "Y taught X, as preceptor or in a discourse.",
    "rules": "X is the reigning king of the place Y.",
    "bornTo": "X was born to Y, where the text marks birth without specifying kind.",
    "speaksTo": "X addresses Y in direct speech.",
}

# Verb lemma to predicate. Direction is recorded per entry: "agent" means the
# grammatical agent fills the object slot of the triple.
KILL = {"kill", "slay", "slaughter", "behead", "murder", "destroy", "fell"}
CURSE = {"curse"}
MARRY = {"marry", "wed"}
TEACH = {"teach", "instruct", "train"}
PARENT_NOUNS = {"son": "hasBiologicalParent", "daughter": "hasBiologicalParent",
                "child": "hasBiologicalParent", "offspring": "hasBiologicalParent",
                "father": "INV_hasBiologicalParent", "mother": "INV_hasBiologicalParent"}
SPOUSE_NOUNS = {"wife": "INV_hasSpouse", "husband": "INV_hasSpouse",
                "consort": "INV_hasSpouse"}
FOSTER = {"foster", "adopted", "adoptive"}
TEACHER_NOUNS = {"preceptor": "instructedBy", "teacher": "instructedBy",
                 "guru": "instructedBy"}

MIN_CONF = {"passive_agent": 0.80, "active_dobj": 0.75, "poss_kinship": 0.85,
            "of_kinship": 0.80, "appos_kinship": 0.70, "noun_role": 0.70,
            "vocative": 0.90}


# Capitalized forms the corpus labels PERSON that are not names. They arrive
# sentence-initially, so the shape test cannot reject them.
NOT_A_NAME = {"You", "He", "She", "They", "We", "It", "Thus", "Then", "Having",
              "There", "Here", "This", "That", "These", "Those", "But", "And",
              "His", "Her", "Their", "Your", "Who", "What", "When", "Where",
              "Thou", "Thee", "Thy", "Yet", "Now", "One", "Some", "All"}


def load_entities():
    """Surfaces that Phase 1 labelled as entities, so extraction only links to
    strings the annotation already treated as entities."""
    df = pd.read_parquet("build/mentions.parquet")
    lab = df.groupby(df.surface.str.strip()).label.agg(lambda x: x.value_counts().idxmax())
    persons = {s for s, l in lab.items()
               if l == "PERSON" and re.fullmatch(r"[A-Z][a-z]{2,}", s)
               and s not in NOT_A_NAME}
    places = {s for s, l in lab.items() if l in ("GPE", "LOC") and re.fullmatch(r"[A-Z][a-z]{2,}", s)}
    return persons, places, df


def in_name_list(tok):
    """True if this token sits in a comma list of proper nouns.

    Debroy's battle prose lists warriors and then a role phrase as a further
    list member, "Duryodhana, Drona, Shakuni, Vrishasena, the king of Madra,
    you, Somadatta". The parser attaches "the king of Madra" as an apposition to
    whichever list member precedes it, which is the wrong one about a quarter of
    the time. Apposition inside a list is not decidable here, so it is dropped.
    """
    return tok.dep_ == "conj" or any(c.dep_ == "conj" for c in tok.children)


def epithet_head(tok):
    """True for "Indra among men", a form of address, not the god Indra."""
    nxt = tok.doc[tok.i + 1] if tok.i + 1 < len(tok.doc) else None
    return nxt is not None and nxt.text.lower() == "among"


def simile(doc, a, b):
    """True if a comparative marker falls between the two tokens.

    "Slay Dhartarashtra, like Vritra was destroyed by Indra" gave
    Dhartarashtra slain by Indra. The clauses are joined only by the simile.
    """
    lo, hi = sorted((a.i, b.i))
    return any(doc[i].text.lower() in ("like", "as", "just") for i in range(lo, hi))


def usable(tok):
    return not in_name_list(tok) and not epithet_head(tok)


def ent_of(tok, persons, places):
    """The entity string a token denotes, if any. Proper nouns only."""
    t = tok.text.strip()
    if t in persons:
        return t, "person"
    if t in places:
        return t, "place"
    return None, None


def emit(out, subj, pred, obj, conf, pattern, r, evidence):
    if pred.startswith("INV_"):
        pred, subj, obj = pred[4:], obj, subj
    if subj == obj:
        return
    out.append({"subject": subj, "predicate": pred, "object": obj,
                "confidence": round(conf, 3), "pattern": pattern,
                "parva": r["parva"], "adhyaya_ordinal": r["adhyaya_ordinal"],
                "sent_id": r["sent_id"], "cited": r["chapter_cited"],
                "evidence": evidence[:200]})


def extract_sentence(doc, r, persons, places, out):
    for tok in doc:
        lemma = tok.lemma_.lower()

        negated = any(c.dep_ == "neg" for c in tok.children) or any(
            c.text.lower() in ("cannot", "never", "not") for c in tok.children)

        # Passive with an explicit agent: "Bhishma was slain by Arjuna".
        if lemma in KILL and not negated and tok.pos_ in ("VERB", "AUX", "NOUN"):
            victims = [c for c in tok.children if c.dep_ in ("nsubjpass", "nsubj:pass")]
            agents = [g for c in tok.children if c.dep_ == "agent"
                      for g in c.children if g.dep_ == "pobj"]
            for v in victims:
                vs, vk = ent_of(v, persons, places)
                for a in agents:
                    as_, ak = ent_of(a, persons, places)
                    if (vs and as_ and vk == "person" and ak == "person"
                            and not simile(doc, v, a)):
                        emit(out, vs, "slainBy", as_, MIN_CONF["passive_agent"],
                             "passive_agent", r, doc.text)
            # Active with a direct object: "Arjuna killed Bhishma".
            subs = [c for c in tok.children if c.dep_ == "nsubj"]
            objs = [c for c in tok.children if c.dep_ in ("dobj", "obj")]
            for s in subs:
                ss, sk = ent_of(s, persons, places)
                for o in objs:
                    os_, ok = ent_of(o, persons, places)
                    if ss and os_ and sk == "person" and ok == "person":
                        emit(out, os_, "slainBy", ss, MIN_CONF["active_dobj"],
                             "active_dobj", r, doc.text)

        if lemma in CURSE and tok.pos_ == "VERB" and not negated:
            for s in [c for c in tok.children if c.dep_ == "nsubj"]:
                ss, sk = ent_of(s, persons, places)
                for o in [c for c in tok.children if c.dep_ in ("dobj", "obj")]:
                    os_, ok = ent_of(o, persons, places)
                    if ss and os_ and sk == ok == "person":
                        emit(out, os_, "cursedBy", ss, MIN_CONF["active_dobj"],
                             "active_dobj", r, doc.text)
            for v in [c for c in tok.children if c.dep_ in ("nsubjpass", "nsubj:pass")]:
                vs, vk = ent_of(v, persons, places)
                for a in [g for c in tok.children if c.dep_ == "agent"
                          for g in c.children if g.dep_ == "pobj"]:
                    as_, ak = ent_of(a, persons, places)
                    if vs and as_ and vk == ak == "person":
                        emit(out, vs, "cursedBy", as_, MIN_CONF["passive_agent"],
                             "passive_agent", r, doc.text)

        if lemma in MARRY and tok.pos_ == "VERB" and not negated:
            for s in [c for c in tok.children if c.dep_ == "nsubj"]:
                ss, sk = ent_of(s, persons, places)
                for o in [c for c in tok.children if c.dep_ in ("dobj", "obj")]:
                    os_, ok = ent_of(o, persons, places)
                    if ss and os_ and sk == ok == "person":
                        emit(out, ss, "hasSpouse", os_, MIN_CONF["active_dobj"],
                             "active_dobj", r, doc.text)

        # "instructed Ghatotkacha to engage in the duel" is an order, not
        # teaching, and the predicate definition is teaching. An infinitive
        # complement on the verb marks the command reading, so skip it.
        if lemma in TEACH and tok.pos_ == "VERB" and not negated and not any(
                c.dep_ in ("xcomp", "ccomp") for c in tok.children):
            for s in [c for c in tok.children if c.dep_ == "nsubj"]:
                ss, sk = ent_of(s, persons, places)
                for o in [c for c in tok.children if c.dep_ in ("dobj", "obj")]:
                    os_, ok = ent_of(o, persons, places)
                    if ss and os_ and sk == ok == "person":
                        emit(out, os_, "instructedBy", ss, MIN_CONF["active_dobj"],
                             "active_dobj", r, doc.text)

        # Kinship and role nouns.
        roles = {}
        roles.update(PARENT_NOUNS)
        roles.update(SPOUSE_NOUNS)
        roles.update(TEACHER_NOUNS)
        if lemma in roles and tok.pos_ == "NOUN":
            pred = roles[lemma]
            foster = any(c.lemma_.lower() in FOSTER for c in tok.children)
            if foster and pred.endswith("hasBiologicalParent"):
                pred = pred.replace("hasBiologicalParent", "hasFosterParent")
            # possessive: "Kunti's son"
            holders = [c for c in tok.children if c.dep_ == "poss"]
            # "son of Kunti"
            holders += [g for c in tok.children if c.dep_ == "prep" and c.text.lower() == "of"
                        for g in c.children if g.dep_ == "pobj"]
            # the bearer: apposition or the clause subject
            bearers = [c for c in tok.children if c.dep_ == "appos"]
            if tok.dep_ == "appos" and tok.head is not None:
                bearers.append(tok.head)
            # Only a copula subject is the bearer: "Nisha was Manu's wife".
            # Taking any clause subject read "Brihaspati approached Utathya's
            # wife" as Brihaspati being the spouse.
            if tok.dep_ in ("attr", "acomp"):
                bearers += [c for c in tok.head.children if c.dep_ == "nsubj"] if tok.head else []
            for h in holders:
                hs, hk = ent_of(h, persons, places)
                if not hs:
                    continue
                for b in bearers:
                    bs, bk = ent_of(b, persons, places)
                    if bs and bk == hk == "person" and usable(b) and usable(h):
                        pat = "poss_kinship" if any(c.dep_ == "poss" for c in tok.children) else "of_kinship"
                        emit(out, bs, pred, hs, MIN_CONF[pat], pat, r, doc.text)

        # "the king of Hastinapura", "ruler of Panchala"
        if lemma in ("king", "ruler", "lord", "sovereign") and tok.pos_ == "NOUN":
            places_ = [g for c in tok.children if c.dep_ == "prep" and c.text.lower() == "of"
                       for g in c.children if g.dep_ == "pobj"]
            bearers = [c for c in tok.children if c.dep_ == "appos"]
            if tok.dep_ == "appos" and tok.head is not None:
                bearers.append(tok.head)
            for p in places_:
                ps, pk = ent_of(p, persons, places)
                for b in bearers:
                    bs, bk = ent_of(b, persons, places)
                    if ps and bs and pk == "place" and bk == "person" and usable(b):
                        emit(out, bs, "rules", ps, MIN_CONF["noun_role"],
                             "noun_role", r, doc.text)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    persons, places, df = load_entities()
    sents = df.drop_duplicates(subset=["parva", "sent_id"])[
        ["parva", "sent_id", "adhyaya_ordinal", "chapter_cited", "sentence"]]
    if limit:
        sents = sents.head(limit)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    rows = sents.to_dict("records")
    out = []
    texts = [r["sentence"] for r in rows]
    for r, doc in zip(rows, nlp.pipe(texts, batch_size=256, n_process=1)):
        extract_sentence(doc, r, persons, places, out)

    # speaksTo from Phase 3 tellings: narrator addresses the vocative narratee
    tel = pd.read_parquet("build/tellings.parquet")
    for t in tel.itertuples():
        if t.narratee and t.narrator in persons and t.narratee in persons:
            out.append({"subject": t.narrator, "predicate": "speaksTo",
                        "object": t.narratee, "confidence": MIN_CONF["vocative"],
                        "pattern": "vocative", "parva": t.parva,
                        "adhyaya_ordinal": t.adhyaya_ordinal, "sent_id": t.sent_id,
                        "cited": t.cited, "evidence": "vocative narratee in telling"})

    rel = pd.DataFrame(out)
    rel.to_parquet("build/relations.parquet")
    write_ttl(rel, "out/relations.ttl")
    report(rel, len(sents))
    print("sentences=%d triples=%d distinct=%d predicates=%d"
          % (len(sents), len(rel), len(rel.drop_duplicates(["subject", "predicate", "object"])),
             rel.predicate.nunique()))


def iri(kind, *parts):
    tail = "/".join(re.sub(r"[^A-Za-z0-9]+", "_", str(p)).strip("_") for p in parts)
    return "<%s%s/%s>" % (BASE, kind, tail)


def write_ttl(rel, path):
    lines = ["@prefix mbh: <%s> ." % BASE,
             "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", ""]
    for t in rel.itertuples():
        s = iri("e", t.subject)
        o = iri("e", t.object)
        lines.append("[] a mbh:Assertion ; mbh:subject %s ; mbh:predicate mbh:%s ; "
                     "mbh:object %s ; mbh:confidence \"%s\"^^xsd:decimal ; "
                     "mbh:pattern \"%s\" ; mbh:citedAtChapterLevel \"%s\"^^xsd:boolean%s ."
                     % (s, t.predicate, o, t.confidence, t.pattern,
                        "true" if t.cited else "false",
                        "" if pd.isna(t.adhyaya_ordinal)
                        else " ; mbh:inPassage %s" % iri("p", t.parva, int(t.adhyaya_ordinal))))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def report(rel, n_sents):
    d = rel.drop_duplicates(["subject", "predicate", "object"])
    with open("build/phase4_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 4 run report\n\nGenerated by `src/phase4_relations.py`.\n\n")
        f.write("- Sentences parsed: %d\n- Triple instances: %d\n- Distinct triples: %d\n\n"
                % (n_sents, len(rel), len(d)))
        f.write("## By predicate\n\n| predicate | instances | distinct | mean confidence |\n|---|---|---|---|\n")
        for p, g in rel.groupby("predicate"):
            f.write("| %s | %d | %d | %.2f |\n"
                    % (p, len(g), len(g.drop_duplicates(["subject", "object"])),
                       g.confidence.mean()))
        f.write("\n## By pattern\n\n| pattern | instances |\n|---|---|\n")
        for p, g in rel.groupby("pattern"):
            f.write("| %s | %d |\n" % (p, len(g)))
        f.write("\n## Citation coverage\n\n- Chapter-cited: %d\n- Parva level only: %d\n"
                % (int(rel.cited.sum()), int((~rel.cited).sum())))


if __name__ == "__main__":
    main()
