"""Checks for the chat retriever. No network, no API key, no passage index.

The passage half is skipped deliberately: it loads a sentence encoder and
34 MB of vectors, and it is already covered by src/test_semantic.py. What is
new here is name resolution and the rule that an appellation never becomes an
identity, and both run on two small tables.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat import Retriever, build_context, citation


def test_exact(r):
    hits = [h for h in r.resolve("who killed Karna") if h["name"] == "Karna"]
    assert hits, "Karna is in the graph and must resolve"
    assert hits[0]["via"] == "exact" and hits[0]["resolved"]


def test_misspelling(r):
    """The whole reason rapidfuzz is here. Cosine over MiniLM does not do this."""
    for typo, want in [("Arjun", "Arjuna"), ("Yudhistira", "Yudhishthira"),
                       ("Bhisma", "Bhishma"), ("Drupadi", "Droupadi")]:
        got = [h["name"] for h in r.resolve("tell me about " + typo)]
        assert want in got, "%r did not reach %s, got %r" % (typo, want, got)


def test_dropped_apostrophe(r):
    """"whos arjuns father" is the realistic typed question, and norm_surface
    only strips a possessive that kept its apostrophe."""
    got = [h["name"] for h in r.resolve("whos arjuns father")]
    assert "Arjuna" in got, got


def test_spelling_variants_are_not_merged(r):
    """Yudhisthira (29 mentions) and Yudhishthira (2,063) are separate entities
    and nothing in this project says they are one person. A misspelling that
    reaches both must report both, and the context must say so."""
    res = r.resolve("tell me about Yudhistira")
    names = {h["name"] for h in res}
    assert {"Yudhisthira", "Yudhishthira"} <= names, names
    ctx = build_context(res, r.graph_facts(res), {"passages": [], "status": "unavailable"})
    assert "do not merge them" in ctx


def test_annotation_artifacts_excluded(r):
    """"Tell" is declared a PERSON on 2 mentions. It must not swallow the verb."""
    assert not [h for h in r.resolve("tell me about war") if h["name"] == "Tell"]


def test_no_guess(r):
    """A name the corpus does not hold must return nothing, not its nearest neighbour."""
    got = r.resolve("what did Napoleon Bonaparte do")
    assert not got, "invented a resolution for a name outside the corpus: %r" % got


def test_appellation_is_never_resolved(r):
    """The hard rule. Radheya may retrieve Karna's records and may not assert him.

    Radheya is also a node in its own right, because Phase 2 merged nothing, so
    it resolves to itself. What must not happen is Karna arriving as resolved.
    """
    hits = r.resolve("who is Radheya")
    assert hits, "Radheya is in build/appellations.csv and must retrieve candidates"
    karna = [h for h in hits if h["name"] == "Karna"]
    assert karna, "Radheya did not retrieve its candidate referent"
    for h in karna:
        assert h["via"] == "appellation"
        assert not h["resolved"], "an appellation was reported as resolved"
        assert h["appellation"] == "Radheya" and 0 < h["confidence"] <= 1
    assert {h["name"] for h in hits if h["resolved"]} == {"Radheya"}, \
        "something other than the appellation itself came back resolved"


def test_ambiguous_appellation_keeps_every_candidate(r):
    """Achyuta has five candidates and the best scores 0.25. Silently taking it
    is exactly the failure CLAUDE.md forbids, so all five must survive."""
    hits = r.resolve("who is Achyuta")
    assert len(hits) >= 5, "collapsed an ambiguous appellation to %d candidate(s)" % len(hits)
    assert len({h["name"] for h in hits}) == len(hits)


def test_facts_unfiltered_by_predicate(r):
    """Karna's records must come back whichever side of the triple he sits on."""
    facts = r.graph_facts(r.resolve("Karna"))
    assert facts
    assert any(f["object"] == "Karna" for f in facts), "only subject-side rows returned"
    assert len({f["predicate"] for f in facts}) > 1, "looks predicate-filtered"


def test_rare_predicate_survives_the_budget(r):
    """speaksTo is 1,247 of 2,800 assertions at the highest confidence in the
    set. Without a per-predicate turn it fills the block and the answer to
    "who killed Karna" never appears in the context at all."""
    facts = r.graph_facts(r.resolve("who killed Karna"))
    slain = [f for f in facts if f["predicate"] == "slainBy"]
    assert slain, "slainBy did not survive the fact budget"
    assert any(f["subject"] == "Karna" for f in slain)


def test_coref_pronoun_slayers_are_gone(r):
    """The raw extractor output says Karna was slain by Nakula, Vrisha, Vijaya
    and Savyasachi. Each of those came from a pronoun subject that coreference
    attached to Karna: "He was slain by the great-souled Nakula". The reviewed
    file applies the current guards and withholds them.

    Arjuna and Kiriti survive because both are read from a directly named
    argument, which is the distinction the guard is drawing.
    """
    slain = {f["object"] for f in r.graph_facts(r.resolve("who killed Karna"))
             if f["predicate"] == "slainBy" and f["subject"] == "Karna"}
    assert "Arjuna" in slain
    assert not ({"Nakula", "Vrisha"} & slain), "a coref-pronoun slayer came back: %r" % slain


def test_reads_the_reviewed_relations(r):
    """Reading the raw file is how the pronoun slayers got in. If this flips
    back, every guard in review_viewer_graph.py stops reaching an answer."""
    assert r.reviewed, "build/reviewed_relations.parquet was not used"
    assert (r.rel.status == "asserted").all(), "withheld rows leaked into the fact set"


def test_abhimanyu_slayer_stays_absent(r):
    """The contested fact. The coref guard removed every (Abhimanyu, slainBy, *)
    row and the chat layer must not reintroduce one."""
    facts = r.graph_facts(r.resolve("who killed Abhimanyu"))
    bad = [f for f in facts if f["subject"] == "Abhimanyu" and f["predicate"] == "slainBy"]
    assert not bad, "a slayer for Abhimanyu came back: %r" % bad


def test_every_fact_line_carries_provenance(r):
    res = r.resolve("Karna and Arjuna")
    ctx = build_context(res, r.graph_facts(res), {"passages": [], "status": "unavailable"})
    lines = [l for l in ctx.splitlines() if l.startswith("  (")]
    assert lines
    for l in lines:
        assert l.endswith("uncited") or "] " in l or any(
            c.isdigit() for c in l.rsplit(",", 1)[-1]), "no citation on %r" % l


def test_context_warns_before_it_reports(r):
    """The unresolved block must come first, so the model reads the caveat
    before the records it qualifies."""
    res = r.resolve("who is Radheya")
    ctx = build_context(res, r.graph_facts(res), {"passages": [], "status": "unavailable"})
    assert ctx.index("UNRESOLVED NAMES AND EPITHETS") < ctx.index("ASSERTED FACTS")
    assert "Candidate referents, not identities" in ctx
    assert "reached only through the unresolved name Radheya" in ctx


def test_facts_are_json(r):
    """An uncited row has no chapter and pandas spells that NaN, which is not
    JSON. The endpoint serializes with allow_nan=False, so a NaN here is a 400
    on any question that touches an uncited relation."""
    import json
    facts = r.graph_facts(r.resolve("tell me about Yudhistira"))
    json.dumps(facts, allow_nan=False)
    for f in facts:
        assert f["adhyaya"] is None or isinstance(f["adhyaya"], int)
        assert f["cited"] or f["adhyaya"] is None


def test_every_appellation_is_json(r):
    """The NaN bug twice over. Eleven prior_only rows carry no confidence, and
    a NaN reaching the endpoint is a 400 on that question. Sweeping all 121
    appellations costs a second and catches the next one too."""
    import json
    for appellation in sorted(r.app.appellation.unique()):
        res = r.resolve("who is " + appellation)
        facts = r.graph_facts(res)
        payload = {"resolutions": res, "facts": facts,
                   "coreference": r.coreference_notes(facts)}
        try:
            json.dumps(payload, allow_nan=False)
        except ValueError as e:
            raise AssertionError("%r is not JSON serializable: %s" % (appellation, e))


def test_corpus_prior_is_not_coreference_evidence(r):
    """prior_only rows have zero apposition sightings and come from the prior
    that build/canonical_id_audit.md blames for the 295-row Bharadvaja to Drona
    collapse. They must not appear as evidence that two names are one person."""
    prior = r.app[r.app.status == "prior_only"]
    assert len(prior), "expected prior_only rows in build/appellations.csv"
    assert "Bharadvaja" in set(prior.appellation), "the known collapse should be in this set"
    for appellation in prior.appellation:
        facts = r.graph_facts(r.resolve("who is " + appellation))
        for n in r.coreference_notes(facts):
            assert n["appellation"] != appellation, \
                "%r reached the coreference notes on the corpus prior alone" % appellation


def test_prior_only_resolution_says_it_has_no_evidence(r):
    """It may still surface as a lead, but never carrying a fabricated number."""
    hits = [h for h in r.resolve("who is Suyodhana") if h["via"] == "appellation"]
    assert hits and all(h["confidence"] is None for h in hits)
    ctx = build_context(hits, [], {"passages": [], "status": "unavailable"})
    assert "NO apposition evidence" in ctx and "corpus prior" in ctx


def test_epithet_records_are_reachable(r):
    """The corpus files Arjuna's parentage under his epithets, not his name.
    bornTo Pandu is stated six times of Dhananjaya and once each of Savyasachi
    and Phalguna, and hasDivineProgenitor Indra of Dhananjaya. Under Arjuna
    himself there is only Kunti, which is why the answer used to say no father
    was recorded. It was recorded; retrieval could not reach it."""
    facts = r.graph_facts(r.resolve("who was father of arjuna"))
    parents = {(f["subject"], f["object"]) for f in facts
               if f["predicate"] in ("bornTo", "hasDivineProgenitor")}
    # On a graph folded by Phase 4f the Dhananjaya rows sit under Arjuna, so
    # either form shows Pandu is reachable; neither means it is not.
    assert {("Dhananjaya", "Pandu"), ("Arjuna", "Pandu")} & parents, parents
    assert any(o in {"Indra", "Shakra"} for _, o in parents), parents
    # And every one of them is marked with the name it was recorded under.
    for f in facts:
        if f["subject"] == "Dhananjaya":
            assert f["unresolved_via"], "an epithet record arrived unlabelled"


def test_epithet_expansion_is_never_resolved(r):
    """Widening retrieval must not become an identity claim."""
    epithets = [h for h in r.resolve("who was father of arjuna") if h["via"] == "epithet"]
    assert epithets and not any(h["resolved"] for h in epithets)
    assert {"Dhananjaya", "Partha"} <= {h["name"] for h in epithets}
    for h in epithets:
        assert h["basis"] in ("apposition", "prior")
        assert h["of"] == "Arjuna"


def test_known_prior_collapse_stays_out(r):
    """Bharadvaja to Drona is the 295-row failure the corpus prior is famous
    for here. It is prior_only, so the epithet widening must not use it."""
    assert "Bharadvaja" not in {e["appellation"] for e in r.epithets_of.get("Drona", [])}


def test_citation():
    assert citation("07_Drona", 31, True) == "07_Drona 31"
    assert citation("07_Drona", 31, False) == "uncited"
    assert citation("07_Drona", None, True) == "uncited"


if __name__ == "__main__":
    r = Retriever(passages=False)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t() if t.__code__.co_argcount == 0 else t(r)
        print("ok  %s" % t.__name__)
    print("\n%d checks passed" % len(tests))
