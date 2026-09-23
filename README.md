# Mahabharata Knowledge Graph and Embedding

Representing the Mahabharata mathematically. Every entity and every relation
becomes a vector in a learned space, trained on a citation-anchored symbolic
graph extracted from an annotated English translation of the BORI critical
edition.

The symbolic graph is not the deliverable. It is the training data for the
deliverable. The deliverable is a trained embedding, queryable three ways, with
the graph neighbourhood and the vector space shown together.

`docs/KG_SYSTEM.pdf` is the full technical documentation. Build it with
`cd docs && pdflatex KG_SYSTEM.tex`.

## State of the work, stated first

Two facts govern how every number in this repository should be read.

**The Phase 6 entry gate fails on all three conditions and the model was
trained anyway, on explicit instruction.** The metrics below describe how well a
model fits 1,369 triples. None of them is a finding about the Mahabharata, and
every artifact carries a `below_gate` flag that the interface displays.

| gate condition | required | measured |
|---|---|---|
| distinct asserted triples | at least 10,000 | 1,444 |
| entities of degree 5 or more | at least 500 | 137 |
| largest relation share | at most 40% | 50.6% (`speaksTo`) |

**Entity resolution is unfinished.** All 121 appellations are unresolved and
nothing is merged. "Achyuta" has five candidate referents and the best scores
0.25. Spelling variants are therefore separate entities: `Yudhisthira` at 29
mentions and `Yudhishthira` at 2,063 are distinct nodes, and nothing here
asserts they are the same person.

Three tasks are blocked on human judgement and no script can produce them: the
Phase 2 entity census, the Phase 5 blind calibration set, and the Phase 4
relation sample beyond what is already judged.

## What exists

| phase | output | state |
|---|---|---|
| 1. Mentions and citation spine | 125,262 mentions, 99.79% at `(parvan, adhyaya)` | pass |
| 2. Entity resolution | 2,995 persons, 872 places, 121 appellations, **0 merged** | blocked on a human |
| 3. Speech frames | 5,465 tellings to depth 6 | pass |
| 4. Relation extraction | 1,774 asserted triples, 10 predicates, precision 0.850 | pass on a judged sample |
| 5. Propose and verify | 26 proposals, none asserted | blocked on a human |
| 6. Embedding | RotatE, 561 entities, MRR 0.175 | trained below gate |
| 7. Projection and interface | UMAP, static page, three query modes | two of three work |
| 8. Divergence analysis | ranked list produced | negative result |

Phase 8 is the experiment rather than the display. For each entity it compares
the top ten graph neighbours against the top ten embedding neighbours. The
result is negative and reported as such: mean overlap 0.028, and 616 of 683
entities tie at exactly zero, so the ranking orders a tie.

## Repository layout

```
src/          the pipeline, phase by phase, plus the retrieval and chat layer
out/          deliverables: Turtle graphs, embeddings, projection, the interface
docs/         written decisions, dataset findings, status, system documentation
ontology/     the OWL 2 RL ontology and its SHACL shapes
tools/        run_all.sh and the viewer bundle source
annotations/  human judgement and the curated lineage table, tracked because neither is regenerable
ner/          the earlier NER phase that produced the paper, kept for reference
paper/        NER.tex and its figures
build/        intermediates, gitignored, regenerable by definition
```

The corpus is not in this repository. It is read-only input of 300 MB or more
and lives outside version control. `docs/DATASET.md` records its measured
properties, including several that are not what the filenames suggest.

## Running it

```
.venv/bin/pip install -r requirements-semantic.txt -r requirements-chat.txt
.venv/bin/python src/validate_outputs.py
bash tools/run_all.sh
```

`validate_outputs.py` runs 25 checks over `out/` and exits non-zero on any
failure. One currently fails: `(Ghatotkacha, bornTo, Hidimba)` appears in both
the asserted and the proposed graph, so it reports 24 of 25.

### The interface

```
.venv/bin/python src/serve_explorer.py --port 8765
```

Four modes at `http://127.0.0.1:8765/index.html`.

**Graph model** shows the extracted graph, the UMAP projection and an entity
inspector. It runs from `file://` with no server and makes no network request.

**Text and concepts** searches 16,339 source passages with a local sentence
encoder fused with lexical ranking. Results are quoted source text, never a
generated answer.

**Ask** answers a typed question from the graph and the passages together, with
the evidence drawn beside the answer. Spelling and grammar do not have to be
right. This is the only component that sends anything off the machine, it is
opt-in behind `GROQ_API_KEY`, and without a key it returns the retrieved
evidence and generates nothing. The decision record is D27 in
`docs/DECISIONS.md`.

**Lineage** draws a family tree and a generation-by-generation graph from the
curated table in `annotations/lineage/`, and answers kinship questions such as
parents, siblings, descendants, or how two people are related. The table is
hand-authored and uncited. Only 36 of its 254 edges are also stated in the
text, and those carry the chapter. The rest are marked curated. Nothing is
generated and it runs from `file://`. The decision record is D28.

## The rules the code keeps

These are not style preferences. Each one exists because violating it produced a
wrong output that was found and traced.

**Never invent a citation.** A triple with no supporting span is marked
`uncited` and stays visible as uncited. Fabricated provenance is worse than
absent provenance because it cannot be detected downstream.

**Never guess an entity link.** Where a referent is genuinely ambiguous the
system emits a candidate set with confidences, never a single silent choice.

**Leave contested facts unasserted.** Abhimanyu has no single recorded slayer.
The edge is omitted and a note explains why.

**Predicted triples never enter the asserted graph.** Embedding output is a
hypothesis. It lives in a separate named graph, always carries a score, and is
labelled `predicted` in every view.

**Identity is a relation, never `owl:sameAs`.** A shared name is not a resolved
person.

**Generated files are never hand-edited.** If an output is wrong, the generator
is what gets fixed.

## Two caveats that belong in any write-up

**Every relation in this graph is mediated by one translator's word choices.**
The corpus is English throughout. There is no Sanskrit, no critical apparatus
and no variant data anywhere in it.

**The review pass trades recall for precision, and the trade is visible.**
1,145 records are withheld by the extractor guards, and withheld is not
disproved. `(Karna, slainBy, Nakula)` is withheld and false.
`(Duryodhana, slainBy, Bhimasena)` is withheld and true. Both came from a
pronoun subject resolved by coreference, and no guard can tell them apart from
the sentence alone.

## Note on this repository

It is private, and the reason matters. `out/semantic.trig` and the viewer data
files carry 16,339 verbatim sentences of a copyrighted translation. That is
fine as a working artifact and is not fine as a public release. Making this
repository public would require deciding what to do about those files first,
and git history keeps them even after a later commit removes them.
