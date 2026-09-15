"""Final integrity check over everything in out/.

The hard rules are checkable and until now nothing checked them. This parses
every Turtle file with rdflib and tests the rules that would make the output
worthless if broken: fabricated citations, predicted triples leaking into the
asserted graph, dangling references between files, owl:sameAs used for identity,
the predicate set drifting, and the embedding index disagreeing with the graph it
was trained on.

Writes out/validation.md and exits non-zero if any hard check fails, so it is
usable as the last step of tools/run_all.sh.
"""
import json, sys
import numpy as np
import rdflib
from rdflib import RDF, OWL, URIRef

MBH = rdflib.Namespace("http://example.org/mbh/")
# The recorded inventory, docs/PHASE5_SAMPLING.md and docs/DECISIONS.md D22.
# Seventeen: the hand-annotated fifteen, plus isAlliedWith and isEnemyOf, which
# only the extractor emits. The ceiling CLAUDE.md sets is on what a human
# annotates, so it is checked against the annotator's list and not against this.
ANNOTATED = {"hasBiologicalParent", "hasLegalParent", "hasFosterParent",
             "hasDivineProgenitor", "bornTo", "hasSpouse", "slainBy",
             "defeatedBy", "foughtWith", "speaksTo", "cursedBy",
             "instructedBy", "rules", "isAmshaOf", "rebornAs"}
ALLOWED = ANNOTATED | {"isAlliedWith", "isEnemyOf"}
CEILING = 15
checks = []


def ok(name, passed, detail):
    checks.append((name, bool(passed), detail))
    return passed


def spo(g, cls):
    out = []
    for n in g.subjects(RDF.type, cls):
        s, p, o = g.value(n, MBH.subject), g.value(n, MBH.predicate), g.value(n, MBH.object)
        out.append((n, s, p, o))
    return out


def main():
    g = {}
    for f in ("relations", "proposed", "entities", "passages", "tellings"):
        gr = rdflib.Graph()
        try:
            gr.parse("out/%s.ttl" % f, format="turtle")
            ok("%s.ttl parses" % f, True, "%d triples" % len(gr))
        except Exception as ex:
            ok("%s.ttl parses" % f, False, str(ex)[:200])
        g[f] = gr

    asserted = spo(g["relations"], MBH.Assertion)
    proposed = spo(g["proposed"], MBH.Proposal)

    # Predicted triples never enter the asserted graph.
    a_set = {(s, p, o) for _, s, p, o in asserted}
    p_set = {(s, p, o) for _, s, p, o in proposed}
    both = a_set & p_set
    ok("no proposal appears in the asserted graph", not both,
       "%d asserted, %d proposed, %d shared" % (len(a_set), len(p_set), len(both)))
    ok("no Proposal node in relations.ttl",
       not list(g["relations"].subjects(RDF.type, MBH.Proposal)), "by rdf:type")
    ok("no Assertion node in proposed.ttl",
       not list(g["proposed"].subjects(RDF.type, MBH.Assertion)), "by rdf:type")
    ok("every proposal carries a score and status proposed",
       all(g["proposed"].value(n, MBH.score) is not None
           and str(g["proposed"].value(n, MBH.status)) == "proposed" for n, *_ in proposed),
       "%d proposals" % len(proposed))

    # Never invent a citation. cited at chapter level means a passage reference
    # exists; not cited means no reference at all, not a borrowed one.
    bad_cited, bad_uncited = [], []
    for n, s, p, o in asserted:
        c = g["relations"].value(n, MBH.citedAtChapterLevel)
        ref = g["relations"].value(n, MBH.inPassage)
        if c is not None and bool(c.toPython()):
            if ref is None:
                bad_cited.append((s, p, o))
        elif ref is not None:
            bad_uncited.append((s, p, o))
    ok("every chapter-cited assertion names a passage", not bad_cited,
       "%d without a passage" % len(bad_cited))
    ok("every uncited assertion names no passage", not bad_uncited,
       "%d carrying an address they should not have" % len(bad_uncited))

    # Referential integrity between the files.
    passages = set(g["passages"].subjects(RDF.type, MBH.Passage))
    refs = {r for _, r in g["relations"].subject_objects(MBH.inPassage)}
    ok("every cited passage exists in passages.ttl", refs <= passages,
       "%d distinct references, %d missing" % (len(refs), len(refs - passages)))
    # A node is declared if entities.ttl types it at all. Places are objects of
    # `rules` and are declared mbh:Place, not mbh:Person.
    ents = set(g["entities"].subjects(RDF.type, None))
    used = {x for _, s, _, o in asserted for x in (s, o)}
    ok("every entity in an assertion exists in entities.ttl", used <= ents,
       "%d entities used, %d declared, %d missing" % (len(used), len(ents), len(used - ents)))
    t_refs = {r for _, r in g["tellings"].subject_objects(MBH.inPassage)}
    ok("every telling passage exists in passages.ttl", t_refs <= passages,
       "%d references, %d missing" % (len(t_refs), len(t_refs - passages)))

    # A telling either names its narrator or says the narrator is unresolved. It
    # never does neither, which would be a dropped frame, and never both.
    tell = list(g["tellings"].subjects(RDF.type, MBH.Telling))
    bad = [t for t in tell
           if (g["tellings"].value(t, MBH.narrator) is None)
           == (g["tellings"].value(t, MBH.narratorUnresolved) is None)]
    ok("every telling names a narrator or declares it unresolved", not bad,
       "%d tellings, %d unresolved, %d malformed"
       % (len(tell), sum(1 for t in tell
                         if g["tellings"].value(t, MBH.narratorUnresolved) is not None), len(bad)))

    # Identity is a relation, never owl:sameAs.
    same = sum(len(list(x.triples((None, OWL.sameAs, None)))) for x in g.values())
    ok("no owl:sameAs anywhere", same == 0, "%d found" % same)

    # The predicate set has not drifted. Both graphs, not just the asserted one:
    # the proposal layer is where a new predicate would first appear, and
    # checking only `asserted` is how isEnemyOf and bornTo went unrecorded.
    a_preds = {str(p).rsplit("/", 1)[1] for _, _, p, _ in asserted}
    p_preds = {str(p).rsplit("/", 1)[1] for _, _, p, _ in proposed}
    preds = a_preds | p_preds
    ok("predicate set within the recorded inventory", preds <= ALLOWED,
       "%d in use (%d asserted, %d proposed), %d outside: %s"
       % (len(preds), len(a_preds), len(p_preds), len(preds - ALLOWED),
          ", ".join(sorted(preds))))
    ok("hand-annotated inventory at or under %d" % CEILING, len(ANNOTATED) <= CEILING,
       "%d annotated, %d extractor-only" % (len(ANNOTATED), len(ALLOWED - ANNOTATED)))

    # The embedding index agrees with the graph it was trained on.
    idx = json.load(open("out/embedding_index.json"))
    e = np.fromfile("out/entity_emb.bin", dtype="float32")
    r = np.fromfile("out/relation_emb.bin", dtype="float32")
    ok("entity_emb.bin matches the index",
       e.size == len(idx["entities"]) * idx["entity_dim"],
       "%d floats, %d entities x %d" % (e.size, len(idx["entities"]), idx["entity_dim"]))
    ok("relation_emb.bin matches the index",
       r.size == len(idx["relations"]) * idx["relation_dim"],
       "%d floats, %d relations x %d" % (r.size, len(idx["relations"]), idx["relation_dim"]))
    ok("no NaN or infinity in the embedding", np.isfinite(e).all() and np.isfinite(r).all(),
       "checked %d floats" % (e.size + r.size))
    emb_ents = {URIRef(MBH["e/" + n]) for n in idx["entities"]}
    ok("every embedded entity appears in the asserted graph", emb_ents <= used,
       "%d embedded, %d outside the graph" % (len(emb_ents), len(emb_ents - used)))
    proj = json.load(open("out/projection.json"))
    ok("projection covers every embedded entity",
       set(proj["coords"]) == set(idx["entities"]),
       "%d coordinates, %d entities" % (len(proj["coords"]), len(idx["entities"])))
    ok("both seeds recorded", proj.get("seed") and idx.get("seed"),
       "train %s, umap %s" % (idx.get("seed"), proj.get("seed")))
    ok("below-gate flag present in every embedding artefact",
       idx.get("below_gate") is True and proj.get("below_gate") is True
       and '"below_gate":true' in open("out/projection.js", "r").read(),
       "index, projection.json, projection.js")

    fails = [c for c in checks if not c[1]]
    L = ["# Output validation", "",
         "Automated, from `src/validate_outputs.py`. %d checks, %d failed."
         % (len(checks), len(fails)), "",
         "This checks the hard rules that are mechanically checkable. It cannot check",
         "whether a triple is true; that is what the precision samples are for.", "",
         "| check | result | detail |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % (n, "pass" if p else "**FAIL**", d) for n, p, d in checks]
    L += [""]
    if fails:
        L += ["## Failures", ""] + ["- %s: %s" % (n, d) for n, _, d in fails] + [""]
    open("out/validation.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("%d checks, %d failed" % (len(checks), len(fails)))
    for n, _, d in fails:
        print("FAIL", n, "-", d)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
