"""Phase 4. Relation extraction by English dependency parsing.

Sanskrit dependency parsing is not available here and the corpus carries no
Sanskrit, so extraction runs on Debroy's English prose with spaCy. Every relation
in this graph is therefore mediated by one translator's word choices. That is
recorded in docs/DECISIONS.md D7 and belongs in the paper.

Fourteen predicates, fixed. Adding to this list is a deliberate decision, not a
code change. The list is a subset of the recorded inventory in
docs/PHASE5_SAMPLING.md and is checked against it by src/validate_outputs.py.
The docstring said twelve while the dict held fourteen; the dict was right and
the count is now taken from it.

Entities are surface forms, not merged identities, because Phase 2 asserted no
merges. A triple therefore links the strings the text used. Phase 2's candidate
links carry the information needed to fold them later.
"""
import collections, json, re, sys
import pandas as pd
import spacy

import nplink
import vocab
from phase5_propose import GODS

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
    "foughtWith": "X and Y exchange blows in a described encounter. Symmetric, and "
                  "it says nothing about who prevailed or about standing enmity.",
    "defeatedBy": "X was beaten by Y in a described encounter and survived it. "
                  "Distinct from slainBy, which is fatal.",
}

# Verb lemma to predicate. Direction is recorded per entry: "agent" means the
# grammatical agent fills the object slot of the triple.
KILL = {"kill", "slay", "slaughter", "behead", "murder"}

# Combat. Measured in build/phase4d_verbs.md as the largest recoverable block in
# the corpus, about 700 clauses with two resolvable person arguments. The verbs
# are the ones Debroy uses for an exchange of blows. "cover", "envelop" and
# "shroud" are in because in this translation they take arrows as the instrument
# and a warrior as the object: "he enveloped Bhima with arrows".
COMBAT = {"pierce", "strike", "attack", "envelop", "counter", "repulse", "grasp",
          "surround", "shroud", "cover", "afflict", "wound", "assail", "resist",
          "oppose", "encounter", "fight", "battle"}
DEFEAT = {"defeat", "vanquish", "subjugate", "conquer", "rout", "overwhelm"}
# Speech verbs that take the addressee as a direct object, and those that take it
# behind "to". This populates the existing speaksTo, which until now came only
# from the vocative addressee of a Phase 3 telling.
SPEAK_DOBJ = {"tell", "ask", "address", "answer", "command", "summon", "question"}
SPEAK_TO = {"speak", "say", "reply", "talk", "call"}
CURSE = {"curse"}
MARRY = {"marry", "wed"}
TEACH = {"teach", "instruct", "train"}
# A bare parentage noun says a parent relation holds and does not say which
# kind, and this is the epic that turns on the difference: Karna is Radha's son
# and Surya's, Yudhishthira is Pandu's and Dharma's. So these map to `bornTo`,
# the inventory's term for birth with the kind unstated, and only an explicit
# marker in the same sentence narrows it. D23.
PARENT_NOUNS = {"son": "bornTo", "daughter": "bornTo",
                "child": "bornTo", "offspring": "bornTo",
                "father": "INV_bornTo", "mother": "INV_bornTo"}
SPOUSE_NOUNS = {"wife": "INV_hasSpouse", "husband": "INV_hasSpouse",
                "consort": "INV_hasSpouse"}
FOSTER = {"foster", "adopted", "adoptive"}
# "A's preceptor is B" says B taught A, so the holder of the genitive is the
# student and the bearer is the teacher. Emitting it uninverted read "You are my
# preceptor" as Drona having been instructed by Arjuna, in 7 of the 41
# instructedBy rows in the coreference sample.
TEACHER_NOUNS = {"preceptor": "INV_instructedBy", "teacher": "INV_instructedBy",
                 "guru": "INV_instructedBy"}

MIN_CONF = {"passive_agent": 0.80, "active_dobj": 0.75, "poss_kinship": 0.85,
            "of_kinship": 0.80, "appos_kinship": 0.70, "noun_role": 0.70,
            "vocative": 0.90, "combat_dobj": 0.75, "combat_passive": 0.80,
            "speech_dobj": 0.75, "speech_to": 0.80, "defeat_dobj": 0.75,
            "defeat_passive": 0.80,
            "speech_passive": 0.80}

# An argument recovered by coreference rather than read directly is discounted,
# because a coref error and a parse error compound. Third person anaphora is
# discounted less than "I" and "you", which resolve to the speaker and the
# addressee of a quoted speech and are a harder problem.
COREF_PENALTY = {3: 0.85}

# An argument recovered by linking a description to the graph is discounted like
# a coreference argument. It is not the same kind of inference: it is refused
# outright unless the graph settles it, so the uncertainty is in the parent edge
# it rests on rather than in the choice of antecedent.
NPLINK_PENALTY = 0.85

# First and second person mentions are not used. "I" and "you" inside a quoted
# speech resolve to the speaker and the addressee of that speech, and LingMess
# has no notion of a diegetic level, so it resolved "me" to the frame narrator
# Sanjaya or Vaishampayana rather than to the character speaking. Measured
# precision on the triples that used one was 0.348 against 0.497 for third
# person, and the errors are not of a kind a confidence discount expresses.
ALLOW_PERSON = {3}


def load_entities():
    """The shared vocabulary, so extraction links only to strings the annotation
    already treated as entities. One definition, in src/vocab.py."""
    persons, places, _, df = vocab.vocabulary()
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


# A coreference mention may cover a phrase whose head is not a person. "Bhimasena
# swiftly killed his four horses" resolves "his" correctly and then offers
# "horses" as an argument, which read as Yudhishthira slain by Bhimasena. The same
# shape produced the army, the maya and the sacrifice as victims. A coreference
# argument is therefore taken only where the argument's own head word can denote a
# person. Plurals are refused outright, because a cluster resolves to one name and
# "they were slain by Partha" is not a fact about one person.
PERSON_HEADS = {
    "son", "daughter", "child", "father", "mother", "brother", "sister", "wife",
    "husband", "consort", "grandfather", "grandson", "uncle", "nephew", "king",
    "queen", "prince", "lord", "ruler", "sovereign", "emperor", "monarch",
    "chief", "leader", "commander", "preceptor", "teacher", "guru", "disciple",
    "student", "one", "sage", "rishi", "maharshi", "brahmarshi", "devarshi",
    "ascetic", "brahmana", "kshatriya", "warrior", "maharatha", "ratha",
    "archer", "hero", "slayer", "destroyer", "charioteer", "suta", "god",
    "goddess", "man", "woman", "lady", "friend", "enemy", "bull", "tiger",
    "rakshasa", "asura", "daitya", "danava", "gandharva", "apsara", "serpent",
    "naga", "yaksha", "vanquisher", "scorcher", "upholder", "granter",
}
PLURAL_PRON = {"they", "them", "their", "theirs", "these", "those", "both"}


def person_head(tok):
    """True if this token can head a phrase that denotes one person."""
    if tok.lower_ in PLURAL_PRON or tok.tag_ in ("NNS", "NNPS"):
        return False
    return tok.pos_ in ("PRON", "PROPN") or tok.lemma_.lower() in PERSON_HEADS


# A kinship noun with a possessor names a person by relation, not by name.
# "Abhimanyu is then killed by Duhshasana's son" resolved that agent to Drona
# and asserted the one slaying CLAUDE.md names as contested. Coreference is
# sometimes right on these, "Drona's son" for Ashvatthama, but the referent is a
# description and picking one antecedent silently is the thing the hard rules
# forbid. The triple is left unasserted rather than guessed.
KIN_HEADS = {"son", "daughter", "child", "father", "mother", "brother",
             "sister", "wife", "husband", "consort", "grandfather", "grandson",
             "uncle", "nephew", "descendant", "offspring"}


def possessed_kin(tok):
    """True for "X's son", "his son", "the son of X"."""
    if tok.lemma_.lower() not in KIN_HEADS:
        return False
    return any(c.dep_ == "poss" for c in tok.children) or \
        any(c.dep_ == "prep" and c.lower_ == "of" for c in tok.children)


MODALS = {"will", "would", "shall", "should", "can", "could", "may", "might",
          "must", "'ll", "'d"}
HYPO = {"if", "whether", "unless", "lest", "though"}


def non_assertive(tok):
    """True if this verb is not asserting that its event happened.

    "I will kill Jayadratha tomorrow", "Had I possessed the ability, I would
    have killed Vrikodara", "you should instruct me", "Is this what Indra
    instructed you?". Future, conditional, counterfactual, optative and
    interrogative clauses were being read as assertions. Coreference made the
    fault visible because it fills in the arguments of quoted speech, which is
    where modals live, but the fault is in the verb patterns and the guard is
    put there.

    A question mark anywhere in the sentence rejects the whole sentence. That is
    blunt, because Debroy's sentences often carry several quoted clauses, and it
    costs recall on the assertive clauses of a sentence that also asks
    something. Precision is the trade this project takes.
    """
    for t in [tok, *tok.ancestors]:
        for c in t.children:
            if c.dep_ == "aux" and c.text.lower() in MODALS:
                return True
            if c.dep_ == "mark" and c.text.lower() in HYPO:
                return True
    return "?" in tok.sent.text


def usable(tok):
    return not in_name_list(tok) and not epithet_head(tok)


def adjacent(bearer, role):
    """True if the bearer abuts the role phrase, ignoring punctuation.

    "Madri, the daughter of the king of Madra" and "Kunti's son Yudhishthira"
    both abut. A name several list members away does not: "The king of Kashi,
    with the great bow, and maharatha Shikhandi, Dhrishtadyumna, Virata and
    Satyaki, Drupada" attached Drupada to that role and made him king of Kashi,
    which then resolved five later "king of Kashi" descriptions to him. The
    parser puts apposition and coordination in the same shape and the distance
    is the only thing that separates them reliably here.
    """
    if role.dep_ in ("attr", "acomp") and bearer.head == role.head and role.head.lemma_ == "be":
        return True
    doc = role.doc
    lo, hi = sorted((bearer.i, role.i))
    # Nothing coordinated may stand between them. The role noun's own subtree
    # swallows the whole list when the list hangs off it, so the subtree edges
    # cannot be used as the boundary.
    if any(doc[i].dep_ == "conj" or doc[i].pos_ == "CCONJ" for i in range(lo + 1, hi)):
        return False
    span = range(role.right_edge.i + 1, bearer.i) if bearer.i > role.i \
        else range(bearer.i + 1, role.left_edge.i)
    return all(doc[i].pos_ in ("PUNCT", "DET", "ADJ", "ADP", "PROPN", "NOUN")
               for i in span)


def role_bearers(role, persons):
    """A single directly named bearer, never a choice across a list or coref."""
    if role.dep_ == "appos" and role.head.dep_ in ("pobj", "dobj", "obj"):
        following = next((t for t in role.doc[role.right_edge.i + 1:]
                          if t.pos_ not in ("PUNCT", "ADV")), None)
        # "towards Bhishma, Krishna's son roared" is a new subject after an
        # introductory clause, not Bhishma's epithet. spaCy attaches it as one.
        if following is not None and following.tag_ in ("VBD", "VBP", "VBZ", "MD"):
            return []
    bearers = [c for c in role.children if c.dep_ == "appos"]
    if role.dep_ == "appos":
        bearers.append(role.head)
    if role.dep_ in ("attr", "acomp") and role.head.lemma_ == "be":
        if non_assertive(role) or any(c.dep_ == "neg" for c in role.head.children):
            return []
        bearers += [c for c in role.head.children if c.dep_ == "nsubj"]
    named = {t.i: t for t in bearers if t.pos_ == "PROPN" and t.text in persons}
    # "Bhishma, Shantanu's son, Karna" offers two named bearers. Neither
    # apposition attachment is reliable enough to choose one silently.
    return list(named.values()) if len(named) == 1 else []


def parent_kind(role, parent):
    """The kinship primitive a parentage noun licenses, or None to stay at `bornTo`.

    Only what the sentence itself marks narrows the kind. A god in the parent
    slot is the one exception that does not need a modifier, because begetting
    is the only parentage the text ever means there: no god fosters and none is
    a legal parent. Everything else stays underspecified, which is an honest
    gap rather than a guess. D23.
    """
    mods = {c.lemma_.lower() for c in role.children}
    if mods & FOSTER:
        return "hasFosterParent"
    if parent in GODS:
        return "hasDivineProgenitor"
    if "biological" in mods:
        return "hasBiologicalParent"
    return None


def ent_of(tok, persons, places, coref=None, table=None):
    """The entity string a token denotes, if any.

    A token is read directly when it is a known name. Otherwise, if Phase 4b
    resolved a coreference mention covering this token, the antecedent name is
    used and the caller is told, so the confidence can be discounted and the
    provenance recorded.
    """
    t = tok.text.strip()
    # A sentence-initial imperative is capitalised and the annotation labelled
    # several of them PERSON: "Tell me", "Look at", "Know me to be behaviour".
    # The name list cannot tell them apart on shape, the parse can.
    if tok.pos_ in ("VERB", "AUX"):
        return None, None, None
    if t in persons:
        return t, "person", None
    if t in places:
        return t, "place", None
    if table is not None:
        name = nplink.resolve(tok, table, persons, places)
        if name:
            return name, "person", "np"
    if coref and person_head(tok) and not possessed_kin(tok):
        if tok.lower_ in {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                          "you", "your", "yours", "yourself", "thou", "thee", "thy"}:
            return None, None, None
        candidates = set()
        for start, end, name, person in coref:
            if person not in ALLOW_PERSON:
                continue
            if tok.idx >= start and tok.idx + len(tok) <= end:
                span = tok.doc.char_span(start, end, alignment_mode="strict")
                if span is not None and span.root == tok:
                    candidates.add((name, person))
        if len(candidates) == 1:
            name, person = next(iter(candidates))
            return name, "person", person
    return None, None, None


def emit(out, subj, pred, obj, pattern, r, evidence, crs=(None, None)):
    """Record one triple.

    `crs` carries, for the subject and the object, the grammatical person of the
    coreference mention each was recovered from, or None if the name was read
    directly. The confidence is discounted once per coref-recovered argument and
    the provenance is kept, so a consumer can filter to directly-read triples.
    """
    if pred.startswith("INV_"):
        pred, subj, obj = pred[4:], obj, subj
        crs = (crs[1], crs[0])
    if subj == obj:
        return
    # Unverified antecedents turned "him" into Adhiratha and "the Pandava"
    # into Pandu. Keep conflict assertions directly named; review retains the
    # rejected passages separately rather than asserting a guessed opponent.
    if pred in {"slainBy", "foughtWith", "defeatedBy"} and any(cr is not None for cr in crs):
        return
    conf = MIN_CONF[pattern]
    for cr in crs:
        if cr == "np":
            conf *= NPLINK_PENALTY
        elif cr is not None:
            conf *= COREF_PENALTY[cr]
    out.append({"subject": subj, "predicate": pred, "object": obj,
                "confidence": round(conf, 3), "pattern": pattern,
                "coref_subject": crs[0] if crs[0] != "np" else None,
                "coref_object": crs[1] if crs[1] != "np" else None,
                "nplink_subject": crs[0] == "np" or None,
                "nplink_object": crs[1] == "np" or None,
                "depth": r["depth"],
                "parva": r["parva"], "adhyaya_ordinal": r["adhyaya_ordinal"],
                "sent_id": r["sent_id"], "cited": r["chapter_cited"],
                "evidence": evidence})


def extract_sentence(doc, r, persons, places, out, coref=None, table=None):
    for tok in doc:
        lemma = tok.lemma_.lower()

        negated = any(c.dep_ == "neg" for c in tok.children) or any(
            c.text.lower() in ("cannot", "never", "not") for c in tok.children) \
            or non_assertive(tok)

        # Passive with an explicit agent: "Bhishma was slain by Arjuna".
        if lemma in KILL and not negated and tok.pos_ in ("VERB", "AUX", "NOUN"):
            victims = [c for c in tok.children if c.dep_ in ("nsubjpass", "nsubj:pass")]
            agents = [g for c in tok.children if c.dep_ == "agent"
                      for g in c.children if g.dep_ == "pobj"]
            for v in victims:
                vs, vk, vs_cr = ent_of(v, persons, places)
                for a in agents:
                    as_, ak, as__cr = ent_of(a, persons, places)
                    if (vs and as_ and vk == "person" and ak == "person"
                            and v.pos_ == a.pos_ == "PROPN" and usable(v) and usable(a)
                            and not simile(doc, v, a)):
                        emit(out, vs, "slainBy", as_, "passive_agent", r, doc.text,
                             (vs_cr, as__cr))
            # Active with a direct object: "Arjuna killed Bhishma".
            subs = [c for c in tok.children if c.dep_ == "nsubj"]
            objs = [c for c in tok.children if c.dep_ in ("dobj", "obj")]
            for s in subs:
                ss, sk, ss_cr = ent_of(s, persons, places)
                for o in objs:
                    os_, ok, os__cr = ent_of(o, persons, places)
                    if (ss and os_ and sk == ok == "person" and s.pos_ == o.pos_ == "PROPN"
                            and usable(s) and usable(o) and not simile(doc, s, o)):
                        emit(out, os_, "slainBy", ss, "active_dobj", r, doc.text,
                             (os__cr, ss_cr))

        # Combat, defeat and address all read the same two shapes, so they share
        # one block. Combat is symmetric and emitted in the order the clause puts
        # it; deduplicating the pair is the splitter's job in Phase 6, per the
        # symmetric leakage rule.
        for verbs, pred, flip in ((COMBAT, "foughtWith", False),
                                  (DEFEAT, "defeatedBy", True),
                                  (SPEAK_DOBJ, "speaksTo", False)):
            if lemma not in verbs or tok.pos_ != "VERB" or negated:
                continue
            tag = {"foughtWith": "combat", "defeatedBy": "defeat",
                   "speaksTo": "speech"}[pred]
            for s_ in [c for c in tok.children if c.dep_ == "nsubj"]:
                ss, sk, ss_cr = ent_of(s_, persons, places, coref, table)
                for o in [c for c in tok.children if c.dep_ in ("dobj", "obj")]:
                    os_, ok, os__cr = ent_of(o, persons, places, coref, table)
                    if (ss and os_ and sk == ok == "person" and usable(s_)
                            and usable(o) and not simile(doc, s_, o)):
                        a, b, ca, cb = ((os_, ss, os__cr, ss_cr) if flip
                                        else (ss, os_, ss_cr, os__cr))
                        emit(out, a, pred, b, tag + "_dobj", r, doc.text, (ca, cb))
            for v in [c for c in tok.children if c.dep_ in ("nsubjpass", "nsubj:pass")]:
                vs, vk, vs_cr = ent_of(v, persons, places, coref, table)
                for ag in [g for c in tok.children if c.dep_ == "agent"
                           for g in c.children if g.dep_ == "pobj"]:
                    as_, ak, as__cr = ent_of(ag, persons, places, coref, table)
                    if (vs and as_ and vk == ak == "person" and usable(v)
                            and usable(ag) and not simile(doc, v, ag)):
                        # "X was told by Y" puts the speaker in the agent slot,
                        # so speech passives invert; defeat passives already read
                        # victim-then-victor, and combat is symmetric.
                        if pred == "speaksTo":
                            emit(out, as_, pred, vs, tag + "_passive", r,
                                 doc.text, (as__cr, vs_cr))
                        else:
                            emit(out, vs, pred, as_, tag + "_passive", r,
                                 doc.text, (vs_cr, as__cr))

        # "Krishna spoke to Arjuna". The addressee sits behind "to".
        if lemma in SPEAK_TO and tok.pos_ == "VERB" and not negated:
            addressees = [g for c in tok.children
                          if c.dep_ == "prep" and c.text.lower() == "to"
                          for g in c.children if g.dep_ == "pobj"]
            for s_ in [c for c in tok.children if c.dep_ == "nsubj"]:
                ss, sk, ss_cr = ent_of(s_, persons, places, coref, table)
                for o in addressees:
                    os_, ok, os__cr = ent_of(o, persons, places, coref, table)
                    if ss and os_ and sk == ok == "person" and usable(o):
                        emit(out, ss, "speaksTo", os_, "speech_to", r, doc.text,
                             (ss_cr, os__cr))

        if lemma in CURSE and tok.pos_ == "VERB" and not negated:
            for s in [c for c in tok.children if c.dep_ == "nsubj"]:
                ss, sk, ss_cr = ent_of(s, persons, places, coref, table)
                for o in [c for c in tok.children if c.dep_ in ("dobj", "obj")]:
                    os_, ok, os__cr = ent_of(o, persons, places, coref, table)
                    if ss and os_ and sk == ok == "person":
                        emit(out, os_, "cursedBy", ss, "active_dobj", r, doc.text,
                             (os__cr, ss_cr))
            for v in [c for c in tok.children if c.dep_ in ("nsubjpass", "nsubj:pass")]:
                vs, vk, vs_cr = ent_of(v, persons, places, coref, table)
                for a in [g for c in tok.children if c.dep_ == "agent"
                          for g in c.children if g.dep_ == "pobj"]:
                    as_, ak, as__cr = ent_of(a, persons, places, coref, table)
                    if vs and as_ and vk == ak == "person":
                        emit(out, vs, "cursedBy", as_, "passive_agent", r, doc.text,
                             (vs_cr, as__cr))

        if lemma in MARRY and tok.pos_ == "VERB" and not negated:
            for s in [c for c in tok.children if c.dep_ == "nsubj"]:
                ss, sk, ss_cr = ent_of(s, persons, places, coref, table)
                for o in [c for c in tok.children if c.dep_ in ("dobj", "obj")]:
                    os_, ok, os__cr = ent_of(o, persons, places, coref, table)
                    if ss and os_ and sk == ok == "person":
                        emit(out, ss, "hasSpouse", os_, "active_dobj", r, doc.text,
                             (ss_cr, os__cr))

        # "instructed Ghatotkacha to engage in the duel" is an order, not
        # teaching, and the predicate definition is teaching. An infinitive
        # complement on the verb marks the command reading, so skip it.
        if lemma in TEACH and tok.pos_ == "VERB" and not negated and not any(
                c.dep_ in ("xcomp", "ccomp") for c in tok.children):
            for s in [c for c in tok.children if c.dep_ == "nsubj"]:
                ss, sk, ss_cr = ent_of(s, persons, places, coref, table)
                for o in [c for c in tok.children if c.dep_ in ("dobj", "obj")]:
                    os_, ok, os__cr = ent_of(o, persons, places, coref, table)
                    if ss and os_ and sk == ok == "person":
                        emit(out, os_, "instructedBy", ss, "active_dobj", r, doc.text,
                             (os__cr, ss_cr))

        # Kinship and role nouns.
        roles = {}
        roles.update(PARENT_NOUNS)
        roles.update(SPOUSE_NOUNS)
        roles.update(TEACHER_NOUNS)
        # usable(tok) guards the role noun itself, not only its arguments. In
        # "Hridika's son, Kripa, Drona's son, Duryodhana, Shakuni" the parser
        # makes "Drona's son" a conj of Kripa, and the block read it as Kripa
        # being Drona's child. Every one of the seven children the graph gave
        # Drona came from that shape and every one was wrong.
        if lemma in roles and tok.pos_ == "NOUN" and usable(tok):
            pred = roles[lemma]
            # possessive: "Kunti's son"
            holders = [c for c in tok.children if c.dep_ == "poss"]
            # "son of Kunti"
            holders += [g for c in tok.children if c.dep_ == "prep" and c.text.lower() == "of"
                        for g in c.children if g.dep_ == "pobj"]
            bearers = role_bearers(tok, persons)
            for h in holders:
                hs, hk, hs_cr = ent_of(h, persons, places)
                if not hs:
                    continue
                for b in bearers:
                    bs, bk, bs_cr = ent_of(b, persons, places)
                    if (bs and bk == hk == "person" and usable(b) and usable(h)
                            and adjacent(b, tok)):
                        pat = "poss_kinship" if any(c.dep_ == "poss" for c in tok.children) else "of_kinship"
                        typed = pred
                        if pred.endswith("bornTo"):
                            kind = parent_kind(tok, bs if pred.startswith("INV_") else hs)
                            if kind:
                                typed = pred.replace("bornTo", kind)
                        emit(out, bs, typed, hs, pat, r, doc.text, (bs_cr, hs_cr))

        # "the king of Hastinapura", "ruler of Panchala"
        if (lemma in ("king", "ruler", "lord", "sovereign")
                and tok.pos_ == "NOUN" and usable(tok)):
            places_ = [g for c in tok.children if c.dep_ == "prep" and c.text.lower() == "of"
                       for g in c.children if g.dep_ == "pobj"]
            bearers = role_bearers(tok, persons)
            for p in places_:
                ps, pk, ps_cr = ent_of(p, persons, places)
                for b in bearers:
                    bs, bk, bs_cr = ent_of(b, persons, places)
                    if (ps and bs and pk == "place" and bk == "person"
                            and usable(b) and adjacent(b, tok)):
                        emit(out, bs, "rules", ps, "noun_role", r, doc.text,
                             (bs_cr, ps_cr))


def load_coref():
    """Phase 4b resolutions, keyed by sentence, as (start, end, name, person)."""
    try:
        c = pd.read_parquet("build/coref.parquet")
    except FileNotFoundError:
        return {}
    # A mention whose text holds the resolved name in a possessive or oblique
    # slot is about the thing possessed, not about the name. "Nakula's horses had
    # been slain by Vrishasena" gave Nakula as the victim, and "the maharatha who
    # was Bharadvaja's son" gave Bharadvaja, which is Drona's father. 8,226 of
    # the 63,181 mentions are of this shape and they were the largest single
    # error class in the sample. Mentions where the name comes from elsewhere in
    # the cluster, which is how "Kunti's son" reaches Yudhishthira, are kept.
    name = c.canonical.map(re.escape)
    poss = [bool(re.search(n + r"\s*[\u2019']s\b", t) or
                 re.search(r"\bof\s+" + n + r"\b", t))
            for n, t in zip(name, c.surface.astype(str))]
    c = c[[not p for p in poss]]
    out = collections.defaultdict(list)
    for t in c.itertuples():
        out[(t.parva, int(t.sent_id))].append(
            (int(t.start), int(t.end), t.canonical, int(t.person)))
    return out


def load_depth():
    """Diegetic depth per sentence, from Phase 3.

    A triple drawn from a sentence at depth 0 is something the text narrates. A
    triple at depth 2 is something a character says inside another character's
    speech. They are not the same kind of claim and the depth travels with the
    triple so that later work can separate them instead of flattening the epic
    into one voice.
    """
    try:
        t = pd.read_parquet("build/tellings.parquet")
    except FileNotFoundError:
        return {}
    d = {}
    for r in t.itertuples():
        k = (r.parva, int(r.sent_id))
        d[k] = min(d.get(k, 99), int(r.depth))
    return d


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    persons, places, df = load_entities()
    coref_all, depth_all = load_coref(), load_depth()
    sents = df.drop_duplicates(subset=["parva", "sent_id"])[
        ["parva", "sent_id", "adhyaya_ordinal", "chapter_cited", "sentence"]]
    # Sorted because depth is carried forward from the previous sentence.
    sents = sents.sort_values(["parva", "sent_id"])
    if limit:
        sents = sents.head(limit)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    rows = sents.to_dict("records")
    # Depth is carried forward from the last sentence that opened a frame, since
    # most sentences are not themselves speech openers but sit inside one. This
    # is an approximation: it assumes a frame stays open until another opener
    # appears, which is what the typography does in practice but is not
    # guaranteed at a chapter boundary.
    last = {}
    for r in rows:
        k = (r["parva"], int(r["sent_id"]))
        if k in depth_all:
            last[r["parva"]] = depth_all[k]
        r["depth"] = last.get(r["parva"], 0)
    # Two passes. The first reads the prose with no link table and produces the
    # directly read graph; the second links descriptions such as "Bharadvaja's
    # son" against it. The corpus is parsed twice rather than held in memory, and
    # the table is never rebuilt from the second pass, so a run reproduces from
    # nothing and one wrong parent edge cannot breed a family. See src/nplink.py.
    texts = [r["sentence"] for r in rows]

    def sweep(table):
        out = []
        for r, doc in zip(rows, nlp.pipe(texts, batch_size=256, n_process=1)):
            coref = coref_all.get((r["parva"], int(r["sent_id"])))
            extract_sentence(doc, r, persons, places, out, coref, table)
        return out

    first = pd.DataFrame(sweep(None))
    table = nplink.build_table(first)
    out = sweep(table)
    print("nplink table: %d parents, %d places; %s"
          % (len(table["children"]), len(table["rulers"]), dict(table["stats"])))

    # speaksTo from Phase 3 tellings: narrator addresses the vocative narratee
    tel = pd.read_parquet("build/tellings.parquet")
    for t in tel.itertuples():
        if t.narratee and t.narrator in persons and t.narratee in persons:
            out.append({"subject": t.narrator, "predicate": "speaksTo",
                        "object": t.narratee, "confidence": MIN_CONF["vocative"],
                        "pattern": "vocative", "coref_subject": None,
                        "coref_object": None, "depth": int(t.depth),
                        "parva": t.parva,
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
        f.write("- Parentage recorded as `bornTo` where the sentence does not say the kind: %d\n\n"
                % int(d.predicate.str.endswith("bornTo").sum()))
        f.write("## By predicate\n\n| predicate | instances | distinct | mean confidence |\n|---|---|---|---|\n")
        for p, g in rel.groupby("predicate"):
            f.write("| %s | %d | %d | %.2f |\n"
                    % (p, len(g), len(g.drop_duplicates(["subject", "object"])),
                       g.confidence.mean()))
        f.write("\n## By pattern\n\n| pattern | instances |\n|---|---|\n")
        for p, g in rel.groupby("pattern"):
            f.write("| %s | %d |\n" % (p, len(g)))
        f.write("\n## Arguments recovered by coreference\n\n"
                "| coref arguments | instances |\n|---|---|\n")
        nc = (rel.coref_subject.notna().astype(int)
              + rel.coref_object.notna().astype(int))
        for k, v in sorted(nc.value_counts().items()):
            f.write("| %d | %d |\n" % (k, v))
        f.write("\n## Diegetic depth of the asserting sentence\n\n"
                "Depth 0 is narrated by the text. Deeper is a claim made inside a\n"
                "character's speech.\n\n| depth | instances |\n|---|---|\n")
        for k, v in sorted(rel.depth.value_counts().items()):
            f.write("| %d | %d |\n" % (k, v))
        f.write("\n## Citation coverage\n\n- Chapter-cited: %d\n- Parva level only: %d\n"
                % (int(rel.cited.sum()), int((~rel.cited).sum())))


if __name__ == "__main__":
    main()
