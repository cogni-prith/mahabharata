"""Noun-phrase entity linking against the asserted graph.

The largest remaining class of unrecoverable relation clauses is the one where an
argument is a description rather than a name: "Bharadvaja's son struck him", "the
king of Madra was slain". Coreference does not reach these, and the Phase 4b
guards explicitly refuse them, because a coreference mention that holds a name in
a possessive slot is about the thing possessed and resolving it to the possessor
was the largest error class in that phase.

A description of this shape is resolvable without guessing when the graph already
answers it. If the graph asserts exactly one child of Bharadvaja, then
"Bharadvaja's son" denotes that child. If it asserts two, the phrase is ambiguous
and nothing is emitted, which is the hard rule against guessing an entity link
applied to an argument slot rather than to a mention.

The table is built only from triples that were read directly off the prose: no
coreference argument, no link from this module. Building it from derived triples
would let one wrong parent edge manufacture a family, and the errors would
compound silently across a re-run. For the same reason this does not iterate to a
fixed point. One pass, from the directly read graph.

Gender is not used. The corpus records no gender and the graph stores parenthood,
not sex, so "Drupada's daughter" and "Drupada's son" resolve the same way and both
are refused where Drupada has more than one asserted child. That costs recall on
exactly the families the text describes most.
"""
import collections

CHILD_NOUNS = {"son", "daughter", "child", "offspring"}
RULER_NOUNS = {"king", "ruler", "lord", "sovereign", "monarch"}


# An edge has to be read off more than one sentence before a description is
# allowed to rest on it. One parse error is otherwise enough to breed a family:
# a single wrong row made "Krishna" the sole asserted parent of Bhishma, and
# every later "Krishna's son" then resolved to Bhishma. That was four of the 250
# judged rows and it is the failure this module was built to avoid.
MIN_EVIDENCE = 2

# Any predicate that puts the object in the parent slot resolves "X's son",
# whatever kind of parentage the sentence happened to mark. Most are `bornTo`,
# because most sentences do not say the kind. D23.
PARENT_PREDS = ("bornTo", "hasBiologicalParent", "hasFosterParent",
                "hasDivineProgenitor")


def build_table(rel):
    """Parent to children, and place to rulers, from the directly read graph."""
    d = rel[rel.coref_subject.isna() & rel.coref_object.isna()]
    if "nplink_subject" in d.columns:
        d = d[d.nplink_subject.isna() & d.nplink_object.isna()]
    seen = collections.defaultdict(set)
    for t in d.itertuples():
        if t.predicate in PARENT_PREDS or t.predicate == "rules":
            kind = "rules" if t.predicate == "rules" else "parent"
            seen[(kind, t.object, t.subject)].add((t.parva, t.sent_id))
    children = collections.defaultdict(set)
    rulers = collections.defaultdict(set)
    for (pred, obj, subj), ev in seen.items():
        if len(ev) < MIN_EVIDENCE:
            continue
        (children if pred == "parent" else rulers)[obj].add(subj)
    return {"children": dict(children), "rulers": dict(rulers),
            "stats": collections.Counter()}


def _holder(tok, persons, places):
    """The name a genitive or "of" phrase hangs on, read directly only."""
    for c in tok.children:
        if c.dep_ == "poss" and c.text.strip() in persons:
            return c.text.strip()
        if c.dep_ == "prep" and c.text.lower() == "of":
            for g in c.children:
                if g.dep_ == "pobj" and g.text.strip() in (persons | places):
                    return g.text.strip()
    return None


def resolve(tok, table, persons, places):
    """The entity this description denotes, or None where it is not settled."""
    lemma = tok.lemma_.lower()
    if lemma in CHILD_NOUNS:
        key, bucket = _holder(tok, persons, places), "children"
    elif lemma in RULER_NOUNS:
        key, bucket = _holder(tok, persons, places), "rulers"
    else:
        return None
    if key is None:
        return None
    cands = table[bucket].get(key)
    if not cands:
        table["stats"]["holder_not_in_graph"] += 1
        return None
    if len(cands) > 1:
        table["stats"]["ambiguous"] += 1
        return None
    name = next(iter(cands))
    if name == key:
        return None
    table["stats"]["resolved_" + bucket] += 1
    return name
