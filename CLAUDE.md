# Mahabharata Knowledge Graph Embedding

Representing the Mahabharata mathematically. Every entity and every relation
becomes a vector in a learned space, trained on a citation-anchored symbolic
graph extracted from an annotated corpus of the BORI critical edition in English
translation.

The symbolic graph is not the deliverable. It is the training data for the
deliverable. The deliverable is a trained embedding, queryable three ways, with
the graph neighbourhood and the vector space shown together.

## Orientation

The input is an annotated English-translation corpus. The intermediate output is
an RDF graph in which every triple carries a passage anchor at
`(parvan, adhyaya)` and a diegetic level. The final output is a knowledge graph
embedding over that graph, a 2D projection of it, and a static interface.

"Mathematically" is meant literally. A force-directed layout is a drawing
algorithm with no semantics and is not what this project produces. An embedding
is a trained model in which distance carries meaning and vector arithmetic is
defined.

### The three query modes

1. **Nearest neighbours in vector space.** Similarity, not adjacency. The two
   differ and the difference is the point.
2. **Link prediction.** Score every candidate tail for an incomplete triple.
   `(Duryodhana, isAlliedWith, ?)` ranks entities the text never links.
3. **Analogy by vector arithmetic.** `Arjuna - Kunti + Gandhari` should land near
   Duryodhana. This demo has to work.

### The scientific result to aim at

Karna's vector should sit near the Pandava cluster although every symbolic edge
places him with the Kauravas. The symbolic graph cannot express that; the
embedding can, and it is visible on a 2D projection.

The contribution is the generalised, measurable form of that observation, not
the screenshot. For every entity, compute the overlap between its top-k graph
neighbours and its top-k embedding neighbours by cosine similarity, then rank
entities by lowest overlap. That ranked list is a finding about the epic's social
structure produced by a method rather than by handpicking a good example. This is
Phase 8.

## The corpus

`data/` does not exist. Do not look for it. The corpus is two Google Drive export
folders at the repo root, both containing a single same-named subfolder. Paths
contain spaces, so quote them.

| Path | Contents |
|---|---|
| `Mahabharata LLM annotated-20260801T163642Z-1-001/Mahabharata LLM annotated/` | 18 BIO token CSVs, LLM-annotated |
| `Mahabharat_Human_Annotated_Data-20260801T163709Z-1-001/Mahabharat_Human_Annotated_Data/` | 8 BIO token CSVs, human pass |
| `mahabharata_ner_by_sentence_master (2).csv` | 66,802 sentences, punctuation and nested quotes intact |
| `mahabharata_ner_entities_master (2).csv` | 123,994 entity rows |
| `parva_pdfs/*.pdf` | 20 PDFs, the only source of chapter boundaries |

Scope is all 18 parvas. Harivamsha is excluded: it has no token-level annotation.

Confirmed properties of this corpus, measured and recorded in `docs/DATASET.md`.
Do not re-derive these from filenames or assume otherwise:

- **Eight parvas have a human pass, not twelve.** Two of the eight are unrevised
  machine output: Mahaprasthanika is byte-identical to the LLM file and Mausala
  differs by one token.
- **The human pass changed 1.25% of tokens** (4,780 of 382,001). The human layer
  is therefore not independent of the LLM layer. This governs Phase 5.
- **The `gold_hand` tier in the master CSV is mislabelled.** Its parvas are
  Sabha, Virata and Stri, none of which has a human file. Never use it for a
  score.
- **No Sanskrit anywhere.** Zero Devanagari, zero IAST, in the corpus and in the
  PDFs. English translation only.
- **No critical apparatus and no variant data.**
- **No shloka number is printed inline anywhere.** Front matter gives per-chapter
  shloka counts, which do not locate a verse.
- **The substrates join on normalized sentence text, never on index.** See the
  next section.
- **`shanti_parva.csv` is not confined to Shanti Parva.** It shares 1,635 long
  sentences with `anushasana_tagged.csv` and ends with Mahaprasthanika text.
  Re-derive its parva labels; do not believe them.

### The substrate join

The master CSVs carry the citation spine and the speech frames, because they
retain punctuation and nested quotes. The BIO CSVs carry token spans. Joining
them is not optional and the key is not obvious.

**Join on normalized sentence text.** Normalize as `preprocess_pdf.py` does:
strip possessives, replace all punctuation with whitespace, drop digit-only
tokens, collapse whitespace, lowercase. Measured coverage is 95.1% to the LLM
files and 95.7% to the human files, with no parva below 89.3%.

**Do not join on `sent_id + 1 == sentence_no`.** That join scores roughly 96% on
fourteen parvas and exactly 0% on Karna, Shalya, Mahaprasthanika and
Svargarohana, which it drops silently as absent rows rather than errors. The
master CSV is a sparse subset that kept its original indices while the BIO files
re-segmented independently. No fixed offset repairs it. A sample of five parvas
will not catch this. Validate any join across all 18 and print per-parva rates.

Every mention carries a `tier` field recording which substrate it came from, so
later analysis can filter to human-touched records without re-deriving
provenance.

## Citation

The citation unit is `(parvan, adhyaya)`. Shloka-level citation is impossible
from this corpus and no shloka-aligned Sanskrit text is being acquired.

Chapter granularity is not uniform, and the difference is load-bearing:

| Parvas | Chapter heads recoverable |
|---|---|
| 6 through 18 | 1,054 of 1,057 |
| Udyoga | 28 of 197 |
| Adi, Sabha, Aranyaka, Virata | 0 of 654 |

Where no chapter head exists, the passage is anchored at `(parvan)` with
`adhyaya` null and `cited: false` at chapter level. The null stays null. It is
not interpolated from position, not estimated from shloka counts, and not carried
over from a neighbouring passage.

Anchors are keyed on `(parva, ordinal)`. Printed global chapter numbers contain
typos in the source, confirmed in Shalya, which prints `Chapter 1126(7)` where
the sequence requires 1226. Retain the printed number as an observed attribute,
never as the key.

## Hard rules

Not style preferences. Violating these makes the output worthless.

**Never invent a citation.** A triple with no supporting span does not get an
address. It gets `cited: false` and stays visible as uncited. Fabricated
provenance is worse than no provenance because it cannot be detected downstream.

**Never guess an entity link.** "Krishna" resolves to Vasudeva Krishna, Arjuna,
Draupadi or Krishna Dvaipayana depending on context. Where the referent is
genuinely ambiguous, emit the mention with a candidate set and a confidence,
never a single silent choice.

**Leave contested facts unasserted.** Abhimanyu has no single recorded slayer.
Omit the edge and carry a note explaining why. Do the same anywhere the text does
not settle it. An honest gap beats a plausible guess.

**Read the source before asserting anything about it.** Never describe the
dataset, a schema, or any file from memory or inference. Open it.

**Nothing goes public without sign-off.** No public repo, no demo link, no blog
post, no dataset release, no putting the interface on a personal site. This
includes pushing to any remote that is not explicitly approved.

**Ask before expanding the predicate set.**

### Additional hard rules for the embedding work

**Predicted triples never enter the asserted graph.** They live in a separate
named graph, always carry a score, and are labelled `predicted` in every view. An
embedding output is a hypothesis, not an assertion about the text.

**Never evaluate on hand-authored seed data.** If a seed graph is ever added it
is a schema fixture and a smoke test, nothing else.

**Record the random seed** for every training run and every UMAP projection. A
projection you cannot reproduce is a screenshot, not a result.

**Report the negative sampling strategy and ratio.** KGE numbers are not
comparable across sampling regimes and omitting it makes the evaluation
unreadable.

**Never train before the Phase 6 entry gate passes.** Embeddings trained on too
little data produce noise that looks like results.

## Modelling invariants

Settled. If one needs revisiting, raise it and record the change in
`docs/DECISIONS.md`. Do not quietly change it in code.

**Named graphs, not RDF-star.** One named graph per `Telling` span, one for
predicted triples. Quad support in SPARQL is universal; RDF-star support is not.
This also carries the distinction between what the text asserts and what a
character claims. Reason within a graph, query across graphs.

**OWL 2 RL, not DL.** RL is rule-based and forward-chainable, so transitive
closure materializes in seconds at corpus scale. A DL tableau reasoner buys
nothing here.

**Causal predicates stay atomic.** Never declare `causes` transitive. An RL
reasoner would materialize the closure and discard every intermediate step, which
destroys the only thing causal analysis needs. Walk them at query time with
`causes+`.

**Identity is a relation, never `owl:sameAs`.** Keep `isAmshaOf`, `rebornAs`,
`transformedInto` and `disguisedAs` strictly separate.

**Store kinship primitives only.** `hasBiologicalParent`, `hasLegalParent`,
`hasFosterParent`, `hasDivineProgenitor`. Derive siblings, cousins and uncles at
query time with an inequality filter. The OWL 2 property chain for siblings makes
everyone their own sibling.

**Keep the hand-annotated relation inventory to ten or twelve predicates** with
hard written definitions. Entity agreement on this corpus reached 0.831.
Relation agreement typically lands between 0.5 and 0.65 and "is this an alliance"
will be worse. Automated extraction handles the long tail with confidence scores.
Do not let the list drift past twelve without a deliberate, recorded decision.

## Schema

There is no `ontology/mahabharata.ttl` in this repo and no seed graph. The schema
is derived minimally from the predicates named in this file, recorded in
`docs/DECISIONS.md`. If a validated OWL 2 RL ontology is added later, adopt it;
`IdentityConflationShape` and `PatronymicShape` bear directly on Phase 2.

## Build plan

Each phase has an input, an output and a gate. Do not start a phase before its
gate passes. Report at every gate and wait.

### Phase 1. Mention layer and citation spine

Every annotated span becomes a `Mention` anchored to a `Passage` at
`(parvan, adhyaya)` where recoverable, `(parvan)` otherwise.

- Out: `build/mentions.parquet`, `out/passages.ttl`
- Gate: report how many mentions resolve to chapter level, how many to parva
  level only, and the substrate join match rate.

### Phase 2. Entity resolution. This is construction, not validation.

`canonical_id` is unusable as given: 90.8% verbatim echoes, only 19 entities with
more than one surface form, and it makes exactly the errors the hard rules
forbid. Treat it as a weak prior, never as truth.

Build the appellation table: patronymics, matronymics, epithets, toponymics.
Cross-check derived patronymics against the kinship graph. If a mention carries
Radheya but no parent edge reaches Radha, one of the two is wrong.

The Bharadvaja to Drona collapse (295 rows) and Krishna Dvaipayana to Krishna
(72 rows) are not one-off bugs. They are automated canonicalisation collapsing a
patronymic epithet onto its bearer. Quantify that failure mode across the whole
corpus and write it up. It is a publishable result and costs almost nothing to
measure while this phase is already running.

- Out: `out/entities.ttl`, `build/appellations.csv`, `build/canonical_id_audit.md`
- Gate: manual audit of 200 sampled resolutions with precision reported, plus the
  canonicalisation failure counts.

### Phase 3. Speech frames and diegetic depth

The master CSVs preserve nested quotes; 4,687 sentences show two-level nesting and
all frame narrators are present. Build `Telling` nodes with narrator, narratee and
nesting depth, one named graph per telling span. Sauti inside Vaishampayana
inside Sanjaya inside Krishna.

Impossible from the BIO CSVs, where `preprocess_pdf.py` stripped the quotes.

This is the cheapest structural win available and the layer that makes the graph
epistemically honest.

- Out: `out/tellings.ttl`
- Gate: nesting depth reconstructs for the known frame levels. Spot-check the
  Gita and one embedded tale.

### Phase 4. Relation extraction by English dependency parsing

The corpus being pure English is not a loss. Sanskrit dependency parsing is thin
and would have been the hardest part of this project. Run Stanza or spaCy
dependency parsing on the Debroy prose, extract subject-verb-object, and map verbs
to predicates in English.

Every relation in this graph is mediated by one translator's word choices. That
caveat is recorded in `docs/DECISIONS.md` and belongs in the paper, not buried.

- Out: `out/relations.ttl`, every triple with a confidence and a
  `(parvan, adhyaya)` citation
- Gate: precision on a 300-triple manual sample, broken down by predicate.

### Phase 5. Propose and verify

An LLM proposes candidate triples freely. Verification is redesigned because the
annotation layer is contaminated.

**The annotation is an index, not evidence.** It changed 1.25% of tokens against
the LLM output, so verifying LLM proposals against it would measure a model
against a lightly edited copy of its own output. Use the annotation only to
retrieve candidate sentences. Verify against Debroy's prose itself, judging the
proposition against the text rather than against the labels.

**Calibrate on a blind set.** Re-annotate 300 to 500 sentences from scratch,
blind to the LLM output, sampled across parvas. Write the sampling procedure down
before sampling. Use that set and only that set to report agreement and calibrate
the verifier.

Never report a number computed against the existing human files as an independent
evaluation. Where such a number is useful for another reason, label it
silver-vs-silver.

- Out: `out/proposed.ttl`, `build/groundedness_report.md`
- Gate: groundedness rate per predicate against the blind calibration set. No
  proposed triple enters the main graph before this.

### Phase 6. Embedding. The centre of the project.

Train a knowledge graph embedding over the triples from Phases 1 to 5 using
PyKEEN, which supplies the model and the evaluation protocol together.

**Use RotatE.** Not TransE. The predicate set contains symmetric relations
(`isAlliedWith`, `isEnemyOf`, `hasSpouse`), antisymmetric ones (`slainBy`,
`hasBiologicalParent`) and composition (parent applied twice is grandparent).
TransE cannot represent symmetry at all: it forces the relation vector to zero,
collapsing every ally onto their ally. RotatE handles symmetry, antisymmetry,
inversion and composition. ComplEx is an acceptable second choice.

**Entry gate, and it is hard.** Check and report before training:

- At least 10,000 triples after Phase 5. Below roughly 5,000 the vectors are not
  meaningful.
- At least 500 entities of degree 5 or higher. Report the full degree
  distribution.
- No single relation type exceeding 40% of all triples. If `belongsToClan` is
  half the data, the model learns that and little else.

If any condition fails, stop and report. Do not train something meaningless and
present its numbers.

**Two leakage traps, both the standard way this goes wrong:**

- *Inverse leakage.* If `(a, hasBiologicalParent, b)` is in train and a derived
  `(b, hasChild, a)` is in test, the model has seen the answer. Exclude all
  derived inverse relations from the evaluation split.
- *Symmetric leakage.* If `isAlliedWith(a, b)` is in train and
  `isAlliedWith(b, a)` is in test, same problem. Deduplicate symmetric pairs
  before splitting, not after.

Use filtered ranking. Report MRR and Hits at 1, 3 and 10.

- Out: `out/entity_emb.bin`, `out/relation_emb.bin`, `out/eval.md`
- Gate: entry conditions met, leakage checks documented, metrics reported.

### Phase 7. Projection and interface

UMAP to 2D over the entity embeddings. Write coordinates to
`out/projection.json` with the random seed recorded so the layout reproduces.

One static HTML page. No server, no framework, no backend.

- A query bar accepting an entity name, an incomplete triple, or an analogy
  expression.
- Left pane: the graph neighbourhood of the focus entity, actual edges.
- Right pane: the embedding scatter, focus highlighted, coloured by cluster or
  faction.
- Below: ranked link-prediction results with scores, every row labelled
  `predicted` and `uncited`.

Scoring runs client side. A few thousand entities at 200 dimensions is about 4 MB
as float32, which ships alongside the page and scores with a small matmul in
JavaScript.

- Out: `out/index.html`, `out/projection.json`
- Gate: the three query modes work end to end on real data.

### Phase 8. Divergence analysis

The measurable form of the Karna observation. Per-entity overlap between top-k
graph neighbours and top-k embedding neighbours, ranked by lowest overlap.

- Out: `build/divergence.md`
- Gate: the ranked list is produced and the top 20 inspected by hand for whether
  they are interesting or artifacts of low degree.

## Conventions

**Commands must be single-line.** No backslash continuations or multi-line
heredocs in anything handed over to run. Bracketed paste mangles them.

**Directories.** The two corpus folders and the master CSVs are read-only input.
`build/` is intermediates and is gitignored. `out/` is deliverables. `docs/` is
written decisions. `src/` is code. `tools/` is generators.

**Generated files are never hand-edited.** If output is wrong, fix the generator.

**Turtle over JSON for anything that is an assertion.** JSON is fine for
intermediates in `build/`.

**Every extraction script writes a run report** to `build/` with input counts,
output counts and rejection reasons. Silent drops are the main way these
pipelines go wrong, and the substrate join is the proof: a plausible index join
dropped four entire parvas without raising anything.

**Prose style for anything written for humans.** Plain and direct. No em-dashes,
no buzzword triplets, no marketing register.

**Git.** Not a repository yet. Ask before initialising. If it happens, `build/`
is gitignored and there is no remote.

## Anti-goals

- Do not build a co-occurrence character network. Those exist, they are not
  grounded, and they are not this.
- Do not present a force-directed layout as the embedding. They are different
  things and conflating them misrepresents the work.
- Do not train embeddings before the Phase 6 entry gate passes.
- Do not expand to 80 predicates because a schema can express them.
- Do not optimise NER metrics. That question is closed and the answer was
  memorization.
- Do not add a web framework or a backend.
- Do not proceed past a gate without reporting.
