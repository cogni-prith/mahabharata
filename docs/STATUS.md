# Project status

Written 2026-09-13. One line per phase, the gate as the plan states it, and what
is actually on disk. Read `out/validation.md` for the mechanical checks and
`docs/DECISIONS.md` for why anything is the way it is.

The deliverable is a trained embedding of the Mahabharata. It exists, it is
queryable three ways, and **it was trained below its own entry gate on explicit
instruction, which is D19.** That is the single most important fact about the
current state and every output repeats it.

## Gate ledger

| phase | gate | state |
|---|---|---|
| 1. Mentions and citation spine | report chapter-level resolution and the join rate | **pass**. 125,262 mentions, 99.79% at `(parvan, adhyaya)`, 1,995 of 1,995 BORI chapters recovered. `build/phase1_report.md` |
| 2. Entity resolution | manual audit of 200 sampled resolutions, plus canonicalisation failure counts | **blocked on a human**. Failure counts done: 43 rewrites the corpus prior makes, 20 unsupported over 4,507 mentions, 9 shared appellations over 3,845. Sample drawn at `build/entity_audit_sample.csv`, seed 20260912, unjudged. 2,995 entities and 872 places declared, 121 appellations left unresolved, **0 merged** |
| 3. Speech frames and diegetic depth | nesting reconstructs, spot-check the Gita and one embedded tale | **pass**, and only now. 5,465 tellings to depth 6. The two spot-checks are in `build/phase3_spotcheck.md` and run as asserts in `src/phase3_spotcheck.py` |
| 4. Relation extraction | precision on a 300-triple manual sample, by predicate | **pass on a 260-row sample**, 0.808 overall. By provenance: direct 0.959, coref 0.725, noun-phrase-linked 0.800. `build/phase4_precision.md` |
| 5. Propose and verify | groundedness per predicate against a blind calibration set | **blocked on a human**. 26 proposals exist, held in `out/proposed.ttl`, none asserted. The blind 300 to 500 sentence set does not exist, so no groundedness number can be reported |
| 6. Embedding | entry conditions met, leakage documented, metrics reported | **entry conditions FAIL on all three**, trained anyway on instruction. Leakage documented, metrics reported, seeds recorded. `out/eval.md`, D19 |
| 7. Projection and interface | the three query modes work end to end on real data | **two of three work, one does not.** Nearest neighbours and link prediction return sensible ranked output. The analogy demo does not land. `out/phase7_check.md` |
| 8. Divergence | ranked list produced, top 20 inspected by hand | **pass, with a negative result.** `build/divergence.md`, inspected in `docs/PHASE8_INSPECTION.md`. 618 of 688 entities tie at zero overlap, so the ranking orders a tie |

## Where the Phase 6 gate stands

| condition | required | measured |
|---|---|---|
| distinct asserted triples | >= 10,000 | 1,845 |
| entities of degree >= 5 | >= 500 | 169 |
| largest relation share | <= 40% | 42.6%, `speaksTo` |

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
  an apposition-evidenced alias reaches the cosine top 10 for 18.4% of 185 pairs
  against 1.5% by chance. Useful as a candidate generator for Phase 2, not as a
  resolver.
- **Link prediction: works mechanically.** `(Yudhishthira, speaksTo, ?)` returns
  eight already-asserted neighbours in its top eight, which says the model fits
  the training graph, not that it generalises.
- **Analogy: does not work.** `Arjuna - Kunti + Gandhari` puts Duryodhana at rank
  29 of 685; the best form tried reaches rank 8. It has to land at rank 1.
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

`src/validate_outputs.py` runs 25 checks over `out/` and exits non-zero on any
failure: every Turtle file parses, no proposal appears in the asserted graph, every
chapter-cited assertion names a passage and every uncited one names none, no
dangling entity or passage reference, every telling either names a narrator or
declares it unresolved, no `owl:sameAs`, the predicate set inside the recorded
inventory, and the embedding artefacts agreeing with the graph and carrying their
seeds and their below-gate flag. All 25 pass.

## The three things a human has to do

Nothing in the list below can be produced by running a script, and the first two
are the difference between a demo and a result.

1. **The blind calibration set for Phase 5.** 300 to 500 sentences re-annotated
   from scratch, blind to the LLM output, sampled across parvas with the procedure
   written down before sampling. Without it there is no honest groundedness number
   and no verified proposals, which is the only route to the triple count the gate
   needs.
2. **The 200-row Phase 2 entity audit**, then a ruling on the 20-pair alias
   shortlist in `build/alias_candidates.md`. Each pair needs its cited passage
   read. Expect some to stay unmerged: two of the twenty collapse onto Bhargava,
   which is every descendant of Bhrigu and not one man.
3. **A decision on whether to retrain.** The current model is below the gate and
   labelled as such. Once 1 and 2 land, the gate may be reachable, and the run is
   one command.

Nothing has been published. There is no git remote. No sign-off has been given or
asked for.
