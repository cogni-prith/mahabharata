# Phase 5. The blind calibration set: sampling procedure

Written before the sample was drawn, which is the point of writing it down. The
sampler `src/phase5_blind_sample.py` implements exactly what is below and
nothing else. If the procedure changes, it changes here first and the sample is
redrawn with a new seed recorded.

This document was revised on 2026-09-14 and the set redrawn, because the first
draw could not produce the number the gate asks for. What changed and why is in
`docs/DECISIONS.md` as D22. The short form: stratum A was allocated by parva,
which put eight of eleven predicates under n=10 and made a per-predicate rate
uncomputable; the row id was assigned before the shuffle, so it encoded the
stratum; and the proposal layer was 0.7% of stratum A, so the Phase 5 gate would
have been a blind re-audit of Phase 4 wearing Phase 5's name.

## Why this set exists

The Phase 5 gate is a groundedness rate per predicate. It cannot be computed
against the existing human files: the human pass changed 1.25% of tokens
against the LLM output it was edited from, so an agreement number between them
measures a model against a lightly edited copy of itself. Any number computed
that way is labelled silver-vs-silver and is not this.

This set is annotated from scratch, blind to the LLM output, by a human. It is
the only set any reported agreement or groundedness number may be computed
against.

## Population

Sentences from `mahabharata_ner_by_sentence_master (2).csv`, all 18 parvas.
A sentence is eligible when all of the following hold:

1. It resolves to a chapter, meaning `adhyaya_ordinal` is not null. A sentence
   with no chapter anchor cannot be cited at `(parvan, adhyaya)` and is not
   useful as calibration for a citation-anchored graph.
2. It is at least 60 characters long. Shorter sentences in this translation are
   vocatives and interjections, not propositions.
3. It carries at least two distinct PERSON surfaces in the annotation layer.

Condition 3 uses the annotation, which the plan permits: the annotation is
allowed as a retrieval index and forbidden as evidence. The set is blind in the
sense that matters, which is that the annotator never sees a label, a span, an
extracted triple or a proposal.

Eligible sentences are deduplicated on the content key before any draw, and the
key is a hash of the normalized sentence rather than of the verbatim one. The
judgement is keyed on it, so two rows sharing a key would share a judgement.
Normalizing first matters: 13,192 sentences meet conditions 1 and 3, of which
13,174 are distinct after normalization, so 18 rows are duplicates that a
verbatim comparison would have kept apart and a key lookup would then have
collided. The sampler asserts uniqueness rather than trusting it. Stratum C rows
sort first in the deduplication, so a census row is never the copy that is
dropped.

## Strata

Three strata, because groundedness, recall and the proposal layer are three
different questions and pooling any two gives a number that reads as an
evaluation and is not.

- **Stratum A, 200 sentences.** Phase 4 asserted at least one triple from this
  sentence. Reading these gives the groundedness rate: of what the pipeline
  claims, how much the prose supports.
- **Stratum B, 200 sentences.** The pipeline produced nothing from this
  sentence, although it meets conditions 1 to 3 and therefore looks extractable.
  Reading these gives the recall side: of what the prose states, how much the
  pipeline never saw.
- **Stratum C, a census.** Every sentence from which Phase 5 proposed a triple.
  There are 19 of them, so this is a census and not a sample.

Stratum C exists because Phase 5's gate decides whether proposals may be
asserted, and proposals cover 19 sentences against Phase 4's 2,764. A 200-row
stratum A drawn over the union would be expected to contain 1.4
proposal-bearing sentences, so the reported figure would be 99.3% a measurement
of Phase 4. A and C are reported separately and are never pooled.

Condition 2, the 60-character floor, is **waived for stratum C**. The floor
exists to keep vocatives out of a random draw, and a census is not a draw. Two
of the 19 fall under it, at 45 and 57 characters, and both state a birth in as
many words: "Surasa gave birth to the nagas and Kadru to the pannagas." Excluding
them would drop 10.5% of the layer the gate is about on a criterion aimed at a
different problem. Conditions 1 and 3 still hold for all 19. The waiver and its
count are printed in `build/blind_calibration_report.md`.

Eleven of the 19 also carry a Phase 4 assertion. They are stratum C and not
stratum A, because a sentence belongs to one stratum, and stratum A is drawn
from the asserted pool with these 11 removed. Putting them in both would let 11
non-randomly chosen rows into the stratum A denominator, selected precisely by
having attracted a proposal.

The 7 `isEnemyOf` proposals are outside every stratum by construction. They
carry `sent_id = -1` because `src/phase5_propose.py` emits them from combat
aggregated over four or more chapters, so no single sentence grounds one. They
are reported as a named count in the calibration report and no rate is computed
over them.

## Allocation within a stratum

**Stratum A is allocated by predicate, not by parva.** The gate is a rate per
predicate, so the draw has to guarantee the cell sizes the gate needs. Parva
allocation does not: measured over the eligible population, a parva-proportional
200 gives expected counts of speaksTo 84, hasBiologicalParent 50, foughtWith 48,
slainBy 8.0, rules 4.2, hasSpouse 3.2, defeatedBy 2.7, cursedBy 1.5 and
instructedBy 1.4. Eight of eleven predicates fall below n=10 and no per-predicate
rate survives.

Phase 4 asserts nine predicates. Stratum A takes an equal share of 200 across
those nine, capped at the number of eligible sentences carrying each, rarest
predicate drawn first so a scarce predicate is never crowded out by a common
one, and the remainder redistributed to the predicates that still have supply.
A sentence carrying several predicates counts for all of them and is drawn once,
so a realised cell runs above its allocation rather than below it. The
per-predicate eligible counts and the realised cell sizes are printed by the
sampler and recorded in `build/blind_calibration_report.md`.

The realised draw, sentences carrying each predicate:

| predicate | eligible | drawn | was expected under parva allocation |
|---|---|---|---|
| speaksTo | 779 | 35 | 84 |
| hasBiologicalParent | 518 | 42 | 50 |
| foughtWith | 299 | 32 | 48 |
| slainBy | 54 | 31 | 8.0 |
| rules | 21 | 21 | 4.2 |
| hasSpouse | 18 | 18 | 3.2 |
| defeatedBy | 15 | 15 | 2.7 |
| cursedBy | 13 | 13 | 1.5 |
| instructedBy | 6 | 6 | 1.4 |

Eight of the nine now reach n=13 or better, against one of nine before.

**`instructedBy` cannot reach n=10 at any sample size**, because the whole
eligible population is 6. That is a property of the extraction, not of the
draw. The calibration report prints a count for it and not a rate, and the
threshold is stated there. Knowing this before the draw is the point of writing
the procedure down first.

**Stratum B is allocated by parva**, proportional to each parva's share of that
stratum's eligible population, rounded down, then a floor of 5 sentences per
parva so no parva is unrepresented, then the remainder given to the largest
parvas by eligible count. Stratum B has no predicates to balance, so parva
coverage is the only thing to spread.

**Stratum C is a census** and is not allocated.

Drawn without replacement with `random.Random(SEED)`.

**SEED = 20260914.** Recorded here, in the sampler, in the key file and in the
draw report. The previous draw at SEED = 20260913 is retired and its sheet and
key are deleted. No annotation existed against it, so nothing was lost.

The draw is 200 + 200 + 19 = **419 rows**, over all 18 parvas, with
**sheet id `59a1d3129190`**.

## The sheet is frozen once drawn

`tools/run_all.sh` runs the sampler on every pass, and the eligible frame depends
on three parquet files that earlier steps rewrite. At the same seed, an extractor
change silently draws a different 419 sentences. D21 is the standing proof: the
possessed-kinship guard took the asserted triples from 1,845 to 1,771.

So the sampler refuses to overwrite an existing sheet and says why. A redraw is
a person deleting the sheet and the key on purpose, after changing this document
and recording a new seed. The sheet carries a `sheet_id`, the first 12 hex of a
SHA-1 over its sorted row keys, and every judgement file carries the same value,
so a judgement made against a retired sheet is detectable rather than silently
rescored.

## What the annotator sees

`build/annot_blind.html`, generated from the sheet by `src/annot_pages.py`. It
is a single self-contained page opened from the filesystem, with one sentence on
screen at a time, in the sheet's own shuffled order, with no sort, no filter and
no search. Judgements are exported as `annotations/phase5_blind.json`.

`build/blind_calibration_sheet.csv` is the generator's input and is no longer
what a human reads. Columns:

| column | meaning |
|---|---|
| `key` | the content key, first 10 hex of BLAKE2b over the normalized sentence |
| `parva`, `adhyaya` | the citation, so the passage can be looked up |
| `sentence` | the text, verbatim, punctuation and nested quotes intact |

There is no positional row id. In the retired draw the id was assigned over a
list built stratum A first and stratum B second, and only then shuffled, so
`c000` to `c199` were one stratum and `c200` to `c399` the other. Sorting the
sheet by id in a spreadsheet handed over the design. A content key cannot carry
that information, which is why the fix is to delete the id rather than to
reorder its assignment: a second recorded seed is still a recorded seed, and
anyone holding the repo can replay a shuffle.

Per sentence the annotator records one row state, which is never inferred from
an empty field:

| state | meaning |
|---|---|
| `undecided` | the default. Nobody visited, or visited and left. Not typeable |
| `triples` | one or more triples committed |
| `none` | read, and it states no relation in the inventory |

`unclear` is a property of an individual triple and not of the row, because
rule 5 attaches it to a slot. A row with two triples, one of them carrying a
description rather than a name, has to say which.

No label, no span, no extracted triple and no stratum appears in the sheet or in
the page. `build/blind_calibration_key.csv` holds the key, the stratum, the
sentence and the pipeline's own output for each sentence. **Do not open the key
file before annotating.**

## The predicate inventory to annotate with

Fifteen. The recorded fourteen, plus `bornTo`.

`bornTo` is added because it is the only predicate the pipeline emits that the
annotator could not previously write, and every one of the 19 stratum C rows
uses it. Without it stratum C scores 0.000 by vocabulary mismatch rather than by
the text, which would be a number that is not honest. It is recorded as part of
D22 rather than taken quietly, because CLAUDE.md caps the hand-annotated
inventory and D17 already recorded one expansion.

Nothing outside this list; if the sentence states something the list cannot
express, write it in the note rather than inventing a predicate.

| predicate | direction and meaning |
|---|---|
| `hasBiologicalParent` | `Child \| hasBiologicalParent \| Parent` |
| `hasLegalParent` | the socially recognised parent where the text separates them |
| `hasFosterParent` | the raising parent, Radha and Adhiratha for Karna |
| `hasDivineProgenitor` | the god who fathered, Surya for Karna |
| `bornTo` | `Child \| bornTo \| Parent`, where the text marks a birth without saying of what kind |
| `hasSpouse` | either order, the text rarely marks one |
| `slainBy` | `Victim \| slainBy \| Killer`, only where the sentence says the killing happened |
| `defeatedBy` | `Loser \| defeatedBy \| Winner`, defeat short of death |
| `foughtWith` | symmetric, a fight took place |
| `speaksTo` | `Speaker \| speaksTo \| Addressee` |
| `cursedBy` | `Cursed \| cursedBy \| Curser` |
| `instructedBy` | `Student \| instructedBy \| Teacher` |
| `rules` | `Ruler \| rules \| Place` |
| `isAmshaOf` | a portion or incarnation of a divinity |
| `rebornAs` | a rebirth, not an identity |

Five of these fifteen, `hasLegalParent`, `hasFosterParent`, `hasDivineProgenitor`,
`isAmshaOf` and `rebornAs`, are not emitted by the extractor at all. They stay on
the sheet because stratum B measures what the pipeline never saw, and an
annotator restricted to what the extractor already emits cannot report a miss of
a kind the extractor does not handle. On stratum A they can only produce
unmatched rows, which the calibration report counts in a named bucket rather than
scoring as extraction failures.

## Rules for the annotator

These mirror the project's hard rules and exist so the calibration set is not
contaminated by the same faults it is meant to measure.

1. **Annotate what this sentence states**, not what the epic elsewhere
   establishes. "Arjuna wielded Gandiva" is not evidence that Kunti is his
   mother even though you know she is.
2. **A future, conditional, hypothetical or questioned event is not an
   assertion.** "I will kill Jayadratha tomorrow" states no slaying.
3. **A simile is not an event.** "Slay him as Vritra was destroyed by Indra"
   states nothing about Vritra in this sentence.
4. **Quoted speech still counts**, and the speaker is not the narrator. Record
   the relation the speaker asserts, and note if you think the speaker is
   unreliable.
5. **Where the referent is a description rather than a name**, for example
   "Duhshasana's son", write the description in the slot and tick `unclear` on
   that triple. Do not resolve it to a name from memory.
6. **Where the epic itself does not settle it**, for example who slew
   Abhimanyu, leave it out and say so in the note. A gap is a result.
7. Use the surface name as written in this translation, `Droupadi` and not
   `Draupadi`.
8. **Ignore extraction debris and annotate the sentence.** 27 of the 419 rows
   carry a fused chapter head such as `926(66)` or a stray footnote digit, which
   is the drop-cap problem CLAUDE.md describes arriving in the sheet at 6.4%.
   Every one of them still contains a readable proposition. Where the debris has
   destroyed a name, use rule 5. There is no separate mark for a damaged row:
   excluding 6.4% of the sheet on a non-random criterion would move the
   denominator further than the debris does.

## After the sheet comes back

`src/phase5_calibrate.py` reads `annotations/phase5_blind.json` and, only then,
`build/blind_calibration_key.csv`. It refuses to run, and says so, when no blind
judgement file exists, so the key cannot be unsealed by running the pipeline
early. The guard is on the blind judgement file specifically and not on the
`annotations/` directory, which the entity audit also writes into.

It computes, per predicate and with the strata kept apart at every line:

- **Stratum A groundedness**: of the triples Phase 4 asserted on these
  sentences, the share the annotator also wrote. Matched on the predicate and on
  both names after `phase2_entities.norm_surface`, with `hasSpouse` and
  `foughtWith` matched unordered.
- **Stratum C groundedness**, the same computation over the 19 proposal-bearing
  sentences, reported separately and never pooled with A.
- **Stratum B**: a count and a predicate mix, not a rate. Stratum B is defined
  as the sentences the pipeline produced nothing from, so every annotator triple
  on it is a miss by construction and a rate there is 100% by definition. The
  honest form is how many triples per 200 sentences, and of what kind.
- Unmatched rows in named buckets: a surface mismatch where the predicate and
  one name agree, a sentence the annotator read as stating no relation, a triple
  the annotator wrote nothing matching for, and a predicate the annotator used
  that the extractor cannot emit. A triple that matched but which the annotator
  marked `unclear` is reported on its own line above that table, not inside it,
  because it matched and has already counted as grounded.

Counts, not rates, below n=10, with the threshold stated in the report. The
number that reaches the gate is the per-predicate stratum A rate; everything
else is context for reading it.

It writes `build/phase5_calibration.md`. That is deliberately not
`build/groundedness_report.md`, which `src/phase5_propose.py` writes about the
proposal step itself: two scripts sharing one output name would have replaced
one measurement with the other and nothing would have said so.
