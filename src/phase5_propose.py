"""Phase 5. Propose and verify.

Proposals live in their own graph and never enter the asserted graph. Every one
carries a score and the label `proposed` in every output.

Verification is against Debroy's prose, not against the annotation layer. The
annotation changed 1.25% of tokens against the LLM output it was edited from, so
checking a proposal against it would compare a model to a lightly edited copy of
its own output. The annotation is used only to retrieve candidate sentences and
to supply the entity inventory.

Calibration is blocked. The gate asks for groundedness against a blind set of
300 to 500 sentences re-annotated from scratch by someone who has not seen the
LLM output. That set does not exist and cannot be produced from this side. What
is reported instead is a judged sample, labelled as such, which is a precision
figure and not a calibration.
"""
import collections, re, sys
import pandas as pd
import spacy

BASE = "http://example.org/mbh/"
MIN_ENEMY_CHAPTERS = 4      # distinct chapters of exchanged blows before the
                            # pair counts as declared opponents

# Progenitors the text treats as gods. A parent edge to one of these is
# hasDivineProgenitor, not hasBiologicalParent, and the distinction is a
# modelling invariant, not a nicety.
GODS = {"Vayu", "Indra", "Dharma", "Surya", "Ashvins", "Agni", "Varuna", "Soma",
        "Shiva", "Vishnu", "Brahma", "Yama", "Kubera", "Shakra", "Vasava",
        "Purandara", "Maheshvara", "Mahadeva", "Rudra", "Ishana", "Kartikeya"}

COMBAT = {"pierce", "strike", "attack", "counter", "afflict", "cover", "wound",
          "assail", "repulse", "resist", "oppose", "challenge", "fight"}

BIRTH = re.compile(r"\bgave birth to\b")
FOSTER = re.compile(r"\b(?:brought up|raised|reared|nurtured)\s+by\b")
ADOPT = re.compile(r"\badopt(?:ed|s|ion)\b")
ALLY = re.compile(r"\b(?:ally|allies|allied|alliance)\b")
SIDE = re.compile(r"\bon (?:the |his |your |their )?side of\b")

NOT_A_NAME = {"You", "He", "She", "They", "We", "It", "Thus", "Then", "Having",
              "There", "Here", "This", "That", "These", "Those", "But", "And",
              "His", "Her", "Their", "Your", "Who", "What", "When", "Where",
              "Thou", "Thee", "Thy", "Yet", "Now", "One", "Some", "All", "O"}


def load():
    df = pd.read_parquet("build/mentions.parquet")
    lab = df.groupby(df.surface.str.strip()).label.agg(lambda x: x.value_counts().idxmax())
    persons = {s for s, l in lab.items()
               if l == "PERSON" and re.fullmatch(r"[A-Z][a-z]{2,}", s)
               and s not in NOT_A_NAME}
    sents = df.drop_duplicates(subset=["parva", "sent_id"])[
        ["parva", "sent_id", "adhyaya_ordinal", "chapter_cited", "sentence"]]
    return persons, sents


def ent(tok, persons):
    t = tok.text.strip()
    return t if t in persons else None


def conj_group(tok):
    """A token and everything coordinated with it, so "she gave birth to Druhyu,
    Anu and Puru" yields three children rather than one."""
    out = [tok]
    for c in tok.children:
        if c.dep_ == "conj":
            out.extend(conj_group(c))
    return out


def propose(doc, r, persons, out, combat):
    text = doc.text

    for tok in doc:
        lemma = tok.lemma_.lower()

        # "Sharmishtha gave birth to three sons, Druhyu, Anu and Puru"
        if lemma == "birth" and BIRTH.search(text):
            verb = tok.head
            mothers = [m for c in verb.children if c.dep_ == "nsubj"
                       for m in conj_group(c)]
            # The genealogy formula is "A married B and she gave birth to C".
            # The mother is a pronoun in the second clause, so she is recovered
            # from the object of "married" in the first. This is the one place a
            # pronoun is resolved, and only within a single sentence.
            if all(m.pos_ == "PRON" for m in mothers):
                mothers = [o for t2 in doc if t2.lemma_.lower() in ("marry", "wed")
                           for c in t2.children if c.dep_ in ("dobj", "obj")
                           for o in conj_group(c)]
            kids = [k for c in tok.children if c.dep_ == "prep" and c.text.lower() == "to"
                    for g in c.children if g.dep_ == "pobj" for k in conj_group(g)]
            kids += [k for c in verb.children if c.dep_ == "prep" and c.text.lower() == "to"
                     for g in c.children if g.dep_ == "pobj" for k in conj_group(g)]
            for mo in mothers:
                for kd in kids:
                    ms, ks = ent(mo, persons), ent(kd, persons)
                    if ms and ks:
                        add(out, ks, "bornTo", ms, 0.85, "gave_birth_to", r, text)

        # "raised by X", "adopted by X"
        if tok.dep_ == "pobj" and tok.head.dep_ == "agent" or (
                tok.dep_ == "pobj" and tok.head.text.lower() == "by"):
            if FOSTER.search(text) or ADOPT.search(text):
                verb = tok.head.head
                kids = [k for c in verb.children
                        if c.dep_ in ("nsubjpass", "nsubj:pass", "nsubj")
                        for k in conj_group(c)]
                for kd in kids:
                    ps, ks = ent(tok, persons), ent(kd, persons)
                    if ps and ks:
                        add(out, ks, "hasFosterParent", ps, 0.70, "raised_by", r, text)

        # combat encounters. Not a triple. Evidence toward isEnemyOf, counted
        # over distinct chapters so one arrow exchange cannot assert enmity.
        if lemma in COMBAT and tok.pos_ == "VERB":
            subs = [s for c in tok.children if c.dep_ == "nsubj" for s in conj_group(c)]
            objs = [o for c in tok.children if c.dep_ in ("dobj", "obj") for o in conj_group(c)]
            for s_ in subs:
                for o_ in objs:
                    a, b = ent(s_, persons), ent(o_, persons)
                    if a and b and a != b:
                        combat[tuple(sorted((a, b)))].add(
                            (r["parva"], r["adhyaya_ordinal"]))

        # "an ally of X", "on the side of the Pandavas"
        if (ALLY.search(text) or SIDE.search(text)) and tok.dep_ == "pobj" \
                and tok.head.text.lower() == "of":
            head = tok.head.head
            if head.lemma_.lower() in ("ally", "side", "alliance"):
                subs = [s for c in head.sent.root.children if c.dep_ == "nsubj"
                        for s in conj_group(c)]
                for s_ in subs:
                    a, b = ent(s_, persons), ent(tok, persons)
                    if a and b and a != b:
                        add(out, a, "isAlliedWith", b, 0.60, "ally_of", r, text)


def add(out, s, p, o, conf, pat, r, ev):
    if s == o:
        return
    out.append({"subject": s, "predicate": p, "object": o,
                "score": round(conf, 3), "pattern": pat,
                "parva": r["parva"], "adhyaya_ordinal": r["adhyaya_ordinal"],
                "sent_id": r["sent_id"], "cited": r["chapter_cited"],
                "evidence": str(ev)[:220]})


def reclassify_divine(prop):
    """A parent edge whose parent is a god is a divine progenitor edge.

    Kunti bore Yudhishthira, Bhima and Arjuna to Dharma, Vayu and Indra. Storing
    those as biological parenthood would make three gods ordinary fathers and
    would lose the only thing the epic is careful about here.
    """
    n = 0
    for t in prop:
        if t["predicate"] in ("bornTo", "hasBiologicalParent") and t["object"] in GODS:
            t["predicate"] = "hasDivineProgenitor"
            n += 1
    return n


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    persons, sents = load()
    if limit:
        sents = sents.head(limit)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    rows = sents.to_dict("records")
    out, combat = [], collections.defaultdict(set)
    for r, doc in zip(rows, nlp.pipe([r["sentence"] for r in rows], batch_size=256)):
        propose(doc, r, persons, out, combat)

    enemies = 0
    for (a, b), chapters in combat.items():
        n = len({c for c in chapters if c[1] is not None})
        if n >= MIN_ENEMY_CHAPTERS:
            ch = sorted(c for c in chapters if c[1] is not None)[0]
            out.append({"subject": a, "predicate": "isEnemyOf", "object": b,
                        "score": round(min(0.95, 0.5 + 0.05 * n), 3),
                        "pattern": "combat_over_%d_chapters" % n,
                        "parva": ch[0], "adhyaya_ordinal": ch[1],
                        "sent_id": -1, "cited": True,
                        "evidence": "exchanged blows in %d distinct chapters" % n})
            enemies += 1

    divine = reclassify_divine(out)
    prop = pd.DataFrame(out)
    prop.to_parquet("build/proposed.parquet")
    write_ttl(prop, "out/proposed.ttl")
    report(prop, len(sents), combat, enemies, divine)
    print("proposed=%d distinct=%d predicates=%s enemy_pairs=%d divine=%d combat_pairs=%d"
          % (len(prop), len(prop.drop_duplicates(["subject", "predicate", "object"])),
             sorted(prop.predicate.unique()), enemies, divine, len(combat)))


def iri(kind, *parts):
    tail = "/".join(re.sub(r"[^A-Za-z0-9]+", "_", str(p)).strip("_") for p in parts)
    return "<%s%s/%s>" % (BASE, kind, tail)


def write_ttl(prop, path):
    """One named graph for every proposal. Nothing here is asserted."""
    lines = ["@prefix mbh: <%s> ." % BASE,
             "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
             "",
             "# Proposed triples. Not asserted. Every node carries mbh:status",
             "# \"proposed\" and a score, and must be labelled `proposed` in any view.",
             ""]
    for t in prop.itertuples():
        lines.append("[] a mbh:Proposal ; mbh:status \"proposed\" ; mbh:subject %s ; "
                     "mbh:predicate mbh:%s ; mbh:object %s ; "
                     "mbh:score \"%s\"^^xsd:decimal ; mbh:pattern \"%s\" ; "
                     "mbh:citedAtChapterLevel \"%s\"^^xsd:boolean%s ."
                     % (iri("e", t.subject), t.predicate, iri("e", t.object),
                        t.score, t.pattern, "true" if t.cited else "false",
                        "" if pd.isna(t.adhyaya_ordinal)
                        else " ; mbh:inPassage %s" % iri("p", t.parva, int(t.adhyaya_ordinal))))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def report(prop, n, combat, enemies, divine):
    with open("build/groundedness_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 5 proposal report\n\nGenerated by `src/phase5_propose.py`.\n\n")
        f.write("Nothing in this report is asserted. `out/proposed.ttl` is a separate\n"
                "graph and every node in it carries `mbh:status \"proposed\"` and a score.\n\n")
        f.write("- Sentences searched: %d\n- Proposals: %d\n- Distinct: %d\n\n"
                % (n, len(prop), len(prop.drop_duplicates(["subject", "predicate", "object"]))))
        f.write("## By predicate\n\n| predicate | proposals | mean score |\n|---|---|---|\n")
        for p, g in prop.groupby("predicate"):
            f.write("| %s | %d | %.2f |\n" % (p, len(g), g.score.mean()))
        f.write("\n## isEnemyOf is aggregated, not extracted\n\n"
                "%d ordered pairs exchange blows somewhere in the epic. Asserting\n"
                "enmity from one exchange is wrong: Arjuna pierces Bhishma, whom he\n"
                "reveres. A pair is proposed as declared opponents only after blows in\n"
                "at least %d distinct chapters, which yields %d pairs. The per-encounter\n"
                "evidence is deliberately not emitted as triples, because there is no\n"
                "predicate in the fixed inventory for a single combat exchange and the\n"
                "inventory is not expanded without a recorded decision.\n"
                % (len(combat), MIN_ENEMY_CHAPTERS, enemies))
        f.write("\n## Divine progenitors\n\n%d parent edges were reclassified to\n"
                "`hasDivineProgenitor` because the parent is a god.\n" % divine)
        f.write("\n## Calibration is blocked\n\n"
                "The gate asks for a groundedness rate per predicate against a blind\n"
                "calibration set: 300 to 500 sentences re-annotated from scratch by\n"
                "someone who has not seen the LLM output, sampled across parvas with\n"
                "the procedure written down first. That set does not exist and cannot\n"
                "be produced from this side, because the only person available to\n"
                "annotate it has already read the LLM output.\n\n"
                "No groundedness rate is reported. A figure computed against the\n"
                "existing human files would be silver-vs-silver and the hard rules\n"
                "forbid presenting it as an evaluation. A figure computed by the same\n"
                "process that generated the proposals would be circular.\n\n"
                "What is reported instead is a judged precision sample, in\n"
                "`build/phase5_precision.md`, on the same footing as the Phase 4 gate:\n"
                "the judge wrote the patterns, so it measures whether the cited\n"
                "sentence supports the proposal and nothing about agreement.\n")


if __name__ == "__main__":
    main()
