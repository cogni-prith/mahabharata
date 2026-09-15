# Project status

Written 2026-09-13, updated 2026-09-15. One line per phase, the gate as the plan
states it, and what is actually on disk. Read `out/validation.md` for the
mechanical checks and `docs/DECISIONS.md` for why anything is the way it is.

The deliverable is a trained embedding of the Mahabharata. It exists, it is
queryable three ways, and **it was trained below its own entry gate on explicit
instruction, which is D19.** That is the single most important fact about the
current state and every output repeats it.

## Gate ledger

| phase | gate | state |
|---|---|---|
| 1. Mentions and citation spine | report chapter-level resolution and the join rate | **pass**. 125,262 mentions, 99.79% at `(parvan, adhyaya)`, 1,995 of 1,995 BORI chapters recovered. `build/phase1_report.md` |
| 2. Entity resolution | manual audit of 200 sampled resolutions, plus canonicalisation failure counts | **blocked on a human**. Failure counts done: 43 rewrites the corpus prior makes, 20 unsupported over 4,507 mentions, 9 shared appellations over 3,845. The audit is now a census of all 246 candidate links over 121 appellations, not a 200-row sample, ruled on through `build/annot_entities.html`, unjudged. 2,995 entities and 872 places declared, 121 appellations left unresolved, **0 merged** |
| 3. Speech frames and diegetic depth | nesting reconstructs, spot-check the Gita and one embedded tale | **pass**, and only now. 5,465 tellings to depth 6. The two spot-checks are in `build/phase3_spotcheck.md` and run as asserts in `src/phase3_spotcheck.py` |
| 4. Relation extraction | precision on a 300-triple manual sample, by predicate | **pass on a 240-row sample**, 0.838 overall. By provenance: direct 0.959, coref 0.761, noun-phrase-linked 0.830. `build/phase4_precision.md` |
| 5. Propose and verify | groundedness per predicate against a blind calibration set | **blocked on a human**. 26 proposals exist, held in `out/proposed.ttl`, none asserted. The blind set is drawn and frozen at 419 rows, sheet id `59a1d3129190`, annotated through `build/annot_blind.html`, unjudged. No groundedness number can be reported until it comes back |
| 6. Embedding | entry conditions met, leakage documented, metrics reported | **entry conditions FAIL on all three**, trained anyway on instruction. Leakage documented, metrics reported, seeds recorded. `out/eval.md`, D19 |
| 7. Projection and interface | the three query modes work end to end on real data | **two of three work, one does not.** Nearest neighbours and link prediction return sensible ranked output. The analogy demo does not land. `out/phase7_check.md` |
| 8. Divergence | ranked list produced, top 20 inspected by hand | **pass, with a negative result.** `build/divergence.md`, inspected in `docs/PHASE8_INSPECTION.md`. 616 of 683 entities tie at zero overlap, so the ranking orders a tie |

## Where the Phase 6 gate stands

| condition | required | measured |
|---|---|---|
| distinct asserted triples | >= 10,000 | 1,771 |
| entities of degree >= 5 | >= 500 | 162 |
| largest relation share | <= 40% | 42.2%, `speaksTo` |

This is not a missing-effort problem and no further pattern list will close it.
`build/phase4d_verbs.md` measures the ceiling: 2,641 clauses corpus-wide where a
verb has two resolvable person arguments in an assertive clause. Three other
sources were measured and rejected with numbers rather than argued about: clan
membership fired 38 times in 12,000 candidate sentences, alias folding removes
entities faster than it raises degree, and dialogue turns only add to the one
predicate already over the ceiling. D17 and D18 carry the detail.

Reaching the gate needs a different source of triples. The two candidates are a
verified Phase 5 (an LLM proposing freely, verified against the prose, which needs
the blind calibration set first) and resolving the appellations so that degree
consolidates onto fewer entities.

## What the embedding actually does

- **Nearest neighbours: works, and does something the graph cannot.** It groups an
  entity with its own epithets unprompted. Measured in `build/alias_candidates.md`:
  an apposition-evidenced alias reaches the cosine top 10 for 15.7% of 185 pairs
  against 1.5% by chance. Useful as a candidate generator for Phase 2, not as a
  resolver.
- **Link prediction: works mechanically.** `(Yudhishthira, speaksTo, ?)` returns
  eight already-asserted neighbours in its top eight, which says the model fits
  the training graph, not that it generalises.
- **Analogy: does not work.** `Arjuna - Kunti + Gandhari` puts Duryodhana at rank
  48 of 680; the best form tried reaches rank 11. It has to land at rank 1.
- **The Karna hypothesis has its sign backwards for this graph.** The graph puts
  Karna among Pandavas because his edges are fights; the vector puts him among
  Kauravas because his fights are the Kauravas' fights. Restate the hypothesis
  before testing it again.

## What is on disk

`out/`: `passages.ttl`, `entities.ttl`, `tellings.ttl`, `relations.ttl`,
`proposed.ttl`, `entity_emb.bin`, `relation_emb.bin`, `embedding_index.json`,
`projection.json`, `projection.js`, `index.html`, `eval.md`, `phase7_check.md`,
`validation.md`. `build/` holds the intermediates and every run report and is
gitignored. `tools/run_all.sh` is the pipeline in order, with the three steps that
need a human marked in place.

`annotations/` is tracked and holds human judgement only, because `build/` is
gitignored as regenerable and a judgement is not. It carries the 314 judged
Phase 4 rows today and is where the two annotation exports install. The
annotation pages themselves are generated into `build/` by
`src/annot_pages.py`: two static HTML files, no server and no backend, with
`src/test_annotation.py` covering the export validator, a Chrome round trip, and
a structural check that the blind page carries none of the key's columns. D22.

`src/validate_outputs.py` runs 25 checks over `out/` and exits non-zero on any
failure: every Turtle file parses, no proposal appears in the asserted graph, every
chapter-cited assertion names a passage and every uncited one names none, no
dangling entity or passage reference, every telling either names a narrator or
declares it unresolved, no `owl:sameAs`, the predicate set inside the recorded
inventory across the asserted graph and the proposed one both, and the embedding
artefacts agreeing with the graph and carrying their seeds and their below-gate
flag. All 25 pass. The predicate check used to read the asserted graph only,
which made `bornTo` and `isEnemyOf` invisible to it because they are emitted by
the proposal layer; it now reports 11 predicates in use, 9 asserted and 2
proposed, with 0 outside the inventory. D22 Fault 3.

## A bad edge the interface exposed, and the fix

The rebuilt page answered `who killed Abhimanyu` with one asserted edge,
`(Abhimanyu, slainBy, Drona)`, cited to `07_Drona 31`. Its evidence sentence is
"Abhimanyu is then killed by Duhshasana's son." The agent of that sentence is a
possessed kinship noun, not a name, and coreference resolved it to Drona. The
citation was real and the sentence was real; the entity link was wrong, and it
was wrong on the one fact CLAUDE.md names as contested.

The fault: when the argument of a clause is "X's son" or "his brother", the
referent is a description rather than a name, and resolving it to a single
antecedent is exactly the silent choice the hard rules forbid.

`ent_of` in `src/phase4_relations.py` now refuses a coreference argument whose
mention head is a possessed kinship noun, guarded by `possessed_kin`, with the
Abhimanyu sentence and four other cases as asserts in
`src/test_possessed_kin.py`. The whole pipeline from Phase 4 onward was rerun.
The extractor drops from 3,060 rows to 2,935 and from 1,845 distinct triples to
1,771; coreference-argument rows drop from 1,212 to 1,087;
`(Abhimanyu, slainBy, *)` is now empty. Precision on the judged sample is
unaffected at 0.838, because the judged sets key on the triple rather than on a
row number. The retrained model is 683 entities, 9 relations, MRR 0.1453 and
Hits@10 0.3353, still below the gate and still labelled as such.

Training is deterministic. A seeded run on this GPU reproduced only to 4e-07,
because the gradient scatter into the embedding reduces in block-completion
order, and that was enough to move 638 of 683 entities in the UMAP layout and
drop the projection's top-10 neighbour overlap against its own rerun to 0.32.
`src/phase6_train.py` now sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch
imports and calls `torch.use_deterministic_algorithms(True)`, and two full
train-then-project cycles at seed 20260917 are bit-identical in
`out/entity_emb.bin` and in both coordinate sets, with the metrics unchanged.
D22 Fault 7.

## The four things a human has to do

Nothing in the list below can be produced by running a script, and the first two
are the difference between a demo and a result.

1. **The blind calibration set for Phase 5.** Open `build/annot_blind.html` in a
   browser, judge 419 passages, export, and install the export with
   `.venv/bin/python src/annot_pages.py --import PATH`. `docs/PHASE5_SAMPLING.md`
   is the procedure, written before sampling: three strata over 11,550 eligible
   sentences, seed 20260914, sheet id `59a1d3129190`, all 18 parvas represented.
   Stratum A is 200 allocated by predicate so the per-predicate gate is
   reachable, stratum B is 200 allocated by parva over the sentences the pipeline
   produced nothing from, stratum C is a census of all 19 proposal-bearing
   sentences. The page shows the sentence and nothing else; the pipeline's own
   answer is held in `build/blind_calibration_key.csv`. **Do not open the key
   before annotating.** `src/phase5_calibrate.py` will not touch it until a
   complete submitted review exists and its sheet id matches. Without the
   annotations there is no honest groundedness number and no verified proposals,
   which is the only route to the triple count the gate needs.
2. **The Phase 2 entity census**, through `build/annot_entities.html`: 246
   candidate links over 121 appellations, every one of them, not a sample. Each
   row prints its citation, up to 8 apposition sightings and the passage, so the
   ruling is a reading rather than a hunt. 11 rows are `prior_only`, have no
   apposition span and stay visibly uncited. This census also carries the ruling
   on the 17-pair Phase 8b alias shortlist, all 17 of which are in it; the
   200-row sample was missing 2 of them. Expect some to stay unmerged: Richika
   lands on Bhargava, which is every descendant of Bhrigu and not one man. A
   `yes` is a reading of one passage and still merges nothing.
3. **A decision on whether to retrain once 1 and 2 land.** The current model is
   below the gate and labelled as such. If the annotations close the triple gap
   the run is one command, `tools/run_all.sh` from Phase 6.
4. **A ruling on `bornTo`.** The annotator's page lists fifteen predicates: the
   recorded fourteen plus `bornTo`, which the proposal layer already emits and
   the recorded inventory never named. That is a +1 against "ask before
   expanding the predicate set" and it was implemented under a stated
   assumption, not approved. The reasoning and the alternative are in D22.
   Nothing has been annotated, so reversing it costs one regenerated page.

Nothing has been published. There is no git remote. No sign-off has been given or
asked for.
